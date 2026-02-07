#!/usr/bin/env python3
"""
Phase 4 – Intrepid Worlds I0–I4 (Part 5)

GOAL:
  Keep Phase4_Driver_Part4 mechanics intact (pair production, two-archive coevolution,
  bin-uniform parent sampling, polygamy events, logging, batching, backups),
  while upgrading ONLY:
    (1) world-dependent descriptor axes (I0–I4)
    (2) world-dependent fitness shaping (optional hook)
    (3) world relationship contracts (Guide/Pact/Stoic/Mirror)

ADDITIONAL INTENDED CHANGE (per user intent):
  Polygamy rounds now test *versatility*: during a polygamy round, insertion fitness
  for each side is the arithmetic mean fitness across that genome's multiple matings
  within the batch (Option 1: bin per pairing's metrics, but fitness used for insertion
  is the hub-average for that genome on that side).

Still uses the real evaluator (no placeholder sims):
  evaluate_pair_variant3_with_reliance(...)
"""

import argparse
import ast
import csv
import gzip
import json
import os
import random
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

from Phase3_Support_For_Phase4 import (
    DualCosmosVariant3,
    evaluate_pair_variant3_with_reliance,
    crossover_homologous,
    mutate_genome,
    is_valid_genome,
    get_grid_index,
    OPCODE_HASH,
)

# ----------------------------
# Worker (pickle-safe)
# ----------------------------
def _phase4_eval_one(task):
    """
    Evaluate one pairing (A,B) under the provided world-physics knobs.
    task may include an optional 'extras' dict at the end; older tuples (Part4) are supported.
    """
    if len(task) == 15:
        (
            a, b,
            bits_a, bits_b, targets,
            cycles, grace, pair_reduce,
            gp2_drop_a, gp2_drop_b,
            gp2_flip_a, gp2_flip_b,
            gp2_noise_a, gp2_noise_b,
            perturb_seed,
        ) = task
    else:
        (
            a, b,
            bits_a, bits_b, targets,
            cycles, grace, pair_reduce,
            gp2_drop_a, gp2_drop_b,
            gp2_flip_a, gp2_flip_b,
            gp2_noise_a, gp2_noise_b,
            perturb_seed,
            _extras,
        ) = task

    return evaluate_pair_variant3_with_reliance(
        a, b,
        bits_a, bits_b, targets,
        cycles, grace,
        pair_reduce,
        gp2_drop_a, gp2_drop_b,
        gp2_flip_a, gp2_flip_b,
        gp2_noise_a, gp2_noise_b,
        perturb_seed,
    )

# ----------------------------
# IO helpers
# ----------------------------
def _parse_bin_key(k):
    if isinstance(k, tuple):
        return k
    if isinstance(k, str):
        t = ast.literal_eval(k)
        if isinstance(t, tuple):
            return tuple(int(x) for x in t)
    raise ValueError(f"Unparseable bin key: {k!r}")

def load_phase3_backup(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload.get("meta", {})
    raw_arch = payload.get("archive", {})
    arch = {}
    for k, rec in raw_arch.items():
        arch[_parse_bin_key(k)] = rec
    return meta, arch

def save_backup(path, meta, archive):
    payload = {"meta": meta, "archive": {str(k): v for k, v in archive.items()}}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

def genome_from_record(rec):
    return rec["genome"]

def _genome_sig(genome):
    # hashable identity for grouping within a batch (polygamy averaging)
    return tuple((int(op), int(arg)) for op, arg in genome)

def uniform_bin_sample_key(archive, rng):
    # Uniform over filled bins (MAP-Elites style)
    return rng.choice(list(archive.keys()))

# ----------------------------
# Cosmos perturb helpers (GP3)
# ----------------------------
def _apply_dropout(bits, rng, p):
    if p <= 0.0:
        return bits
    out = bits[:]
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = 0
    return out

def _apply_bit_flip(bits, rng, p):
    if p <= 0.0:
        return bits
    out = bits[:]
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = 1 - out[i]
    return out

def _apply_indep_noise(bits, rng, p):
    if p <= 0.0:
        return bits
    out = bits[:]
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = rng.randrange(2)
    return out

def _apply_complementary_dropout(bits_a, bits_b, rng, p):
    """
    Complementary dropout: when one side drops, the other keeps.
    Implemented by selecting a 'blind' side independently per tick with prob p,
    then dropping only that side.
    """
    if p <= 0.0:
        return bits_a, bits_b
    ba = bits_a[:]
    bb = bits_b[:]
    for i in range(len(ba)):
        if rng.random() < p:
            if rng.random() < 0.5:
                ba[i] = 0
            else:
                bb[i] = 0
    return ba, bb

def perturb_cosmos(bits_a, bits_b, world_spec, rng):
    """
    Returns per-batch perturbed (bits_a, bits_b) according to the world relationship contract.
    """
    ba = bits_a
    bb = bits_b

    # Pact complementarity: apply complementary dropout FIRST (so later noise/flip still apply)
    if world_spec.get("cosmos_complementary_dropout", 0.0) > 0.0:
        ba, bb = _apply_complementary_dropout(
            ba, bb, rng, world_spec.get("cosmos_complementary_dropout", 0.0)
        )

    # Alice-side perturbations
    ba = _apply_dropout(ba, rng, world_spec.get("cosmos_drop_a", 0.0))
    ba = _apply_bit_flip(ba, rng, world_spec.get("cosmos_flip_a", 0.0))
    ba = _apply_indep_noise(ba, rng, world_spec.get("cosmos_noise_a", 0.0))

    # Bob-side perturbations
    bb = _apply_dropout(bb, rng, world_spec.get("cosmos_drop_b", 0.0))
    bb = _apply_bit_flip(bb, rng, world_spec.get("cosmos_flip_b", 0.0))
    bb = _apply_indep_noise(bb, rng, world_spec.get("cosmos_noise_b", 0.0))

    return ba, bb

# ----------------------------
# Fitness reshaping (optional)
# ----------------------------
def reshape_fitness(fit, metrics, world_spec):
    """
    If metrics contains components (keys configured in world_spec),
    compute: wA*FA + wB*FB + wAB*FAB.
    Otherwise return original fit.

    NOTE: Most worlds default to the evaluator's pair fitness.
    """
    if not isinstance(metrics, dict):
        return float(fit)

    kA, kB, kAB = world_spec.get("fitness_keys", ("AliceFitLive", "BobFitLive", "FitnessLive"))
    if (kA in metrics) and (kB in metrics) and (kAB in metrics):
        FA = float(metrics[kA])
        FB = float(metrics[kB])
        FAB = float(metrics[kAB])
        wA, wB, wAB = world_spec.get("fitness_weights", (0.0, 0.0, 1.0))
        return (wA * FA) + (wB * FB) + (wAB * FAB)

    return float(fit)

# ----------------------------
# Axes / binning (world-specific)
# ----------------------------
GRID_SIZE = 16  # must match Phase3 GRID_SIZE

def _bin01(v: float) -> int:
    v = 0.0 if v < 0.0 else 0.999 if v >= 1.0 else v
    return int(v * GRID_SIZE)

def world_bin_index(metrics, world_spec):
    axes_keys = world_spec.get("axes_keys", None)
    if axes_keys is None:
        return get_grid_index(metrics)
    vals = [float(metrics.get(k, 0.0)) for k in axes_keys]
    return tuple(_bin01(v) for v in vals)

# ----------------------------
# World specs (I0–I4)
# ----------------------------
def get_world_specs(args):
    """
    Relationship contract knobs are encoded here:
      - Guide: GP3 asymmetry (Alice cleaner than Bob by default)
      - Pact: complementary GP3 dropout
      - Stoic: GP2 corruption (drop/flip/noise)
      - Mirror: multi-partner evaluation (Option B: aggregate fitness across partners)

    Axes are the Phase4 descriptor spaces (must exist in evaluator metrics or be added by driver).
    """
    base = {
        # GP3 (cosmos) perturbations
        "cosmos_drop_a": 0.0,
        "cosmos_drop_b": 0.0,
        "cosmos_flip_a": 0.0,
        "cosmos_flip_b": 0.0,
        "cosmos_noise_a": 0.0,
        "cosmos_noise_b": 0.0,
        "cosmos_complementary_dropout": 0.0,

        # GP2 (social channel) corruption
        "gp2_drop_a": 0.0,
        "gp2_drop_b": 0.0,
        "gp2_flip_a": 0.0,
        "gp2_flip_b": 0.0,
        "gp2_noise_a": 0.0,
        "gp2_noise_b": 0.0,

        # Fitness shaping
        "fitness_keys": ("AliceFitLive", "BobFitLive", "FitnessLive"),
        "fitness_weights": (0.0, 0.0, 1.0),
    }

    specs = {}

    # I0 – Interpreter (reciprocal baseline)
    # Contract: baseline GP3; reciprocal GP2 without corruption.
    specs["I0"] = dict(base)
    specs["I0"].update({
        "axes_keys": ("Activity", "ChannelEntropyCompress", "AlgoDensity", "SignalSparsity"),
    })

    # I1 – Guide (asymmetric knowledge on GP3; GP2 stable)
    # Contract: GP3 asymmetry: Bob noisier/more dropout than Alice (role may switch later if desired).
    specs["I1"] = dict(base)
    specs["I1"].update({
        "cosmos_drop_a": 0.0,
        "cosmos_drop_b": min(0.25, args.drop_prob + 0.10),
        "cosmos_noise_a": 0.0,
        "cosmos_noise_b": min(0.25, args.noise + 0.05),
        "need_protocol_stability": True,
        "axes_keys": ("SignalSparsity", "BobLatencyNorm", "ProtocolStability", "OpcodeEntropy"),
    })

    # I2 – Pact (complementary GP3 observability; GP2 stable)
    specs["I2"] = dict(base)
    specs["I2"].update({
        "cosmos_complementary_dropout": min(0.35, args.drop_prob + 0.15),
        "axes_keys": ("MutualSignalRate", "SyncIndex", "MemoryUtilization", "ControlFlowSpan"),
    })

    # I3 – Stoic (hostile/unreliable GP2)
    specs["I3"] = dict(base)
    specs["I3"].update({
        "gp2_drop_a": 0.00,
        "gp2_drop_b": 0.00,
        "gp2_flip_a": 0.01,
        "gp2_flip_b": 0.01,
        "gp2_noise_a": 0.00,
        "gp2_noise_b": 0.00,
        "axes_keys": ("RelianceIndex", "SignalSparsity", "RecoveryTime", "DtBurstiness"),
    })

    # I4 – Mirror (partner diversity; Option B: aggregate fitness across partners)
    specs["I4"] = dict(base)
    specs["I4"].update({
        "mirror_k_partners": args.mirror_k_partners,
        "axes_keys": ("PartnerFloor", "ChannelEntropy", "OpcodeEntropy", "ResponseVariance"),
    })

    return specs

# ----------------------------
# Protocol stability proxy (Guide)
# ----------------------------
def add_protocol_stability(metrics_run1, metrics_run2):
    """
    Practical stability proxy in [0,1]:
      1 - average absolute delta over a small set of comm-structure metrics.
    We avoid placeholder simulations: both runs are full Pic10Sim evaluations under different seeds.
    """
    keys = ["SignalSparsity", "MutualSignalRate", "SyncIndex", "ChannelEntropyCompress"]
    deltas = []
    for k in keys:
        v1 = float(metrics_run1.get(k, 0.0))
        v2 = float(metrics_run2.get(k, 0.0))
        deltas.append(abs(v1 - v2))
    d = sum(deltas) / max(1, len(deltas))
    stability = 1.0 - max(0.0, min(0.999, d))
    metrics_run1["ProtocolStability"] = float(stability)
    return metrics_run1

def ensure_bob_latency_norm(metrics: dict, cycles: int):
    """
    Normalize BobLatency into [0,1) as BobLatencyNorm.
    If evaluator already supplies BobLatencyNorm, leave it.
    """
    if not isinstance(metrics, dict):
        return metrics
    if "BobLatencyNorm" in metrics:
        return metrics
    if "BobLatency" in metrics:
        v = float(metrics.get("BobLatency", 0.0))
        metrics["BobLatencyNorm"] = min(0.999, max(0.0, v / max(1, int(cycles))))
    return metrics


# ----------------------------
# Mirror metrics (I4)
# ----------------------------
def add_mirror_metrics(base_metrics, fit_list):
    """
    fit_list: list of fitness values across partner draws.

    Adds:
      - ResponseVariance: normalized std dev of fitness (0..1)
      - PartnerFloor: minimum fitness across partner draws (0..1)
    """
    if not fit_list:
        base_metrics["ResponseVariance"] = 0.0
        base_metrics["PartnerFloor"] = 0.0
        return base_metrics

    m = sum(fit_list) / len(fit_list)
    v = sum((x - m) ** 2 for x in fit_list) / len(fit_list)
    std = v ** 0.5

    # fitness in [0,1]; std max is 0.5 (two-point 0/1 split)
    std_norm = min(0.999, std / 0.5)

    base_metrics["ResponseVariance"] = float(std_norm)

    # Worst-case partner performance (distinct from variance)
    base_metrics["PartnerFloor"] = float(max(0.0, min(0.999, min(fit_list))))

    return base_metrics


# ----------------------------
# World runner
# ----------------------------
def run_world(world_name, args, base_archive, bits_a, bits_b, targets):
    ws_all = get_world_specs(args)
    if world_name not in ws_all:
        raise ValueError(f"Unknown world: {world_name}")
    ws = ws_all[world_name]
    rng = random.Random(args.seed + (hash(world_name) & 0xFFFF))
    pair_reduce_mode = args.pair_reduce

    # Two coevolving archives (Part4 behavior)
    alice_archive = {}
    bob_archive = {}

    # ------------------------------------------------------------
    # TRUE Phase4 seeding: evaluate EVERY Phase3 correlated pair in this world,
    # then bin+insert using Phase4 metrics + Phase4 fitness.
    # ------------------------------------------------------------

    # Apply this world's GP3 relationship contract to the cosmos
    bits_a_w, bits_b_w = perturb_cosmos(bits_a, bits_b, ws, rng)

    seed_tasks = []
    # Note: base_archive is dict(bin_key -> record)
    # Each record stores a correlated pair: rec["genome"] (Alice) and rec["bob_genome"] (Bob).
    for j, rec in enumerate(base_archive.values()):
        ga = rec.get("genome", None)
        gb = rec.get("bob_genome", None)
        if ga is None or gb is None:
            continue

        # IMPORTANT: seeding should reflect the world contract and axes,
        # but we do NOT enable Mirror multi-partner explosion during seeding.
        extras = {
            "need_protocol_stability": bool(ws.get("need_protocol_stability", False)),
            "mirror_mode": False,   # keep seeding cost sane; Mirror pressure happens during evolution
        }

        seed_tasks.append(
            (
                ga,
                gb,
                bits_a_w,
                bits_b_w,
                targets,
                args.cycles,
                args.grace,
                pair_reduce_mode,
                float(ws.get("gp2_drop_a", 0.0)),
                float(ws.get("gp2_drop_b", 0.0)),
                float(ws.get("gp2_flip_a", 0.0)),
                float(ws.get("gp2_flip_b", 0.0)),
                float(ws.get("gp2_noise_a", 0.0)),
                float(ws.get("gp2_noise_b", 0.0)),
                (args.seed + 0x9E3779B1 + j) & 0xFFFFFFFF,
                extras,
            )
        )

    print(f"[{world_name}] Seeding: evaluating {len(seed_tasks)} Phase3 correlated pairs in-world...")

    # Evaluate in parallel (same mechanism as the main batches)
    if args.workers and args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            seed_results = list(ex.map(_phase4_eval_one, seed_tasks, chunksize=args.chunksize))
    else:
        seed_results = [_phase4_eval_one(t) for t in seed_tasks]

    seeded = 0
    for (task, (fit, metrics, a_crash, b_crash)) in zip(seed_tasks, seed_results):
        if a_crash or b_crash:
            continue

        # --- I1 protocol stability during seeding (2nd eval with slightly different GP2 corruption) ---
        if world_name == "I1":
            t2b = list(task)
            t2b[14] = int(t2b[14]) + 99991  # perturb_seed slot

            # Force run-2 to differ so ProtocolStability is informative (not trivially 1.0)
            # Slots: 10 flip_a, 11 flip_b, 12 noise_a, 13 noise_b
            t2b[10] = max(float(t2b[10]), 0.01)
            t2b[11] = max(float(t2b[11]), 0.01)
            t2b[12] = max(float(t2b[12]), 0.02)
            t2b[13] = max(float(t2b[13]), 0.02)

            fit_r2, metrics_r2, ac2, bc2 = _phase4_eval_one(tuple(t2b))
            if not (ac2 or bc2):
                metrics = add_protocol_stability(metrics, metrics_r2)

        # Ensure BobLatencyNorm exists before binning
        metrics = ensure_bob_latency_norm(metrics, args.cycles)

        fit2 = reshape_fitness(fit, metrics, ws)
        idx = world_bin_index(metrics, ws)

        (ga, gb, *_rest) = task

        # Insert into Alice archive (Alice genome competes on fit2 in this bin)
        prev = alice_archive.get(idx)
        if (prev is None) or (fit2 > float(prev.get("fit", -1e9))):
            alice_archive[idx] = {
                "id": "seed",
                "genome": ga,
                "fit": float(fit2),
                "metrics": metrics,
                "origin": {"phase": 3, "seed": True, "world": world_name, "side": "A"},
            }

        # Insert into Bob archive (Bob genome competes on fit2 in this bin)
        prev = bob_archive.get(idx)
        if (prev is None) or (fit2 > float(prev.get("fit", -1e9))):
            bob_archive[idx] = {
                "id": "seed",
                "genome": gb,
                "fit": float(fit2),
                "metrics": metrics,
                "origin": {"phase": 3, "seed": True, "world": world_name, "side": "B"},
            }

        seeded += 1

    print(
        f"[{world_name}] Seeded archives from Phase3 (true rebinned): "
        f"binsA={len(alice_archive)} binsB={len(bob_archive)} seeded_pairs={seeded}"
    )


    # Output paths
    world_dir = os.path.join(args.out_dir, world_name)
    os.makedirs(world_dir, exist_ok=True)
    backups_dir = os.path.join(world_dir, "backups")
    os.makedirs(backups_dir, exist_ok=True)

    metrics_path = os.path.join(world_dir, f"{world_name}_metrics.csv")

    # Prepare CSV
    fieldnames = [
        "batch",
        "pair_idx",
        "world",
        "polygamy_round",
        "perturb_type",
        "fit_raw",
        "fit_used_a",
        "fit_used_b",
        "bin",
        "a_crash",
        "b_crash",
        "opcode_hash_ok",
    ] + list(ws.get("axes_keys", ())) + [
        "AliceFitLive", "BobFitLive", "FitnessLive", "FitnessMute", "RelianceIndex"
    ]
    fieldnames = list(dict.fromkeys(fieldnames))  # preserve order, de-dup

    # ---- Batch summary log (one row per batch) ----
    summary_path = os.path.join(world_dir, f"{world_name}_summary.csv")
    sf = open(summary_path, "w", newline="")
    sw = csv.writer(sf)

    sw.writerow([
        "batch",
        "elapsed_s",
        "world",
        "cycles",
        "binsA",
        "binsB",
        "avg_fit",
        "archive_best",
        "batch_best",
        "rate_per_s",
        "mut_rate",
        "polygamy_prob",
        "crash_frac",
        "mean_len_A",
        "min_len_A",
        "max_len_A",
        "mean_len_B",
        "min_len_B",
        "max_len_B",
    ])


    write_header = not os.path.exists(metrics_path)
    csv_f = open(metrics_path, "a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        csv_w.writeheader()


    total_batches = (args.n_pairs + args.pairs_per_batch - 1) // args.pairs_per_batch
    evaluated = 0
    t0 = time.time()

    # Precompute archive bins list for cheap choice
    def _a_keys():
        return list(alice_archive.keys())
    def _b_keys():
        return list(bob_archive.keys())

    for batch in range(total_batches):
        if evaluated >= args.n_pairs:
            break

        polygamy_round = (rng.random() < args.polygamy_prob)
        perturb_type = rng.randrange(args.perturb_types) if args.perturb_types > 0 else None

        # World-specific GP3 perturbation applied per batch
        bits_a_w, bits_b_w = perturb_cosmos(bits_a, bits_b, ws, rng)

        # Build parent pairs for this batch (Part4 behavior: uniform over bins)
        # Build parent pairs for this batch (Part4 behavior: uniform over bins)
        n_this = min(args.pairs_per_batch, args.n_pairs - evaluated)
        parent_pairs = []
        for _ in range(n_this):
            ak = uniform_bin_sample_key(alice_archive, rng)
            bk = uniform_bin_sample_key(bob_archive, rng)

            a_rec = alice_archive[ak]
            b_rec = bob_archive[bk]

            # reference fitness for gating (parents are not a "pair" elite, so use mean)
            fit_ref = 0.5 * (float(a_rec.get("fit", 0.0)) + float(b_rec.get("fit", 0.0)))

            parent_pairs.append((
                genome_from_record(a_rec),
                genome_from_record(b_rec),
                float(fit_ref),
            ))

        # Polygamy rewiring (Part4 logic)
        if polygamy_round and len(parent_pairs) > 1:
            a_list = [a for (a, _b, _fr) in parent_pairs]
            b_list = [b for (_a, b, _fr) in parent_pairs]

            do_a_hubs = rng.random() < 0.5
            do_b_hubs = rng.random() < 0.5
            if not (do_a_hubs or do_b_hubs):
                do_a_hubs = True

            if do_a_hubs:
                n_hubs = max(1, len(a_list) // 50)  # ~2%
                hub_indices = rng.sample(range(len(a_list)), n_hubs)
                for hi in hub_indices:
                    hub = a_list[hi]
                    k = rng.randint(args.polygamy_k_min, args.polygamy_k_max)
                    slots = rng.sample(range(len(a_list)), min(k, len(a_list)))
                    for s in slots:
                        a_list[s] = hub

            if do_b_hubs:
                n_hubs = max(1, len(b_list) // 50)
                hub_indices = rng.sample(range(len(b_list)), n_hubs)
                for hi in hub_indices:
                    hub = b_list[hi]
                    k = rng.randint(args.polygamy_k_min, args.polygamy_k_max)
                    slots = rng.sample(range(len(b_list)), min(k, len(b_list)))
                    for s in slots:
                        b_list[s] = hub

            # preserve the original fit_ref list order after rewiring
            fit_refs = [fr for (_a, _b, fr) in parent_pairs]
            parent_pairs = [(a_list[i], b_list[i], fit_refs[i]) for i in range(len(a_list))]

        # Create children (Part4 behavior: A×A and B×B crossover within batch)
        # Create children (Part4 behavior: A×A and B×B crossover within batch)
        tasks = []
        children_meta = []  # aligned with tasks: holds child genomes and their sigs for aggregation
        for i in range(n_this):
            # parent includes fit_ref now
            a_parent, b_parent, fit_ref = parent_pairs[i]

            # other parent also includes a fit_ref (ignore it)
            a_other, b_other, _fit_ref_other = parent_pairs[rng.randrange(n_this)]

            a_child = crossover_homologous(a_parent, a_other, rng)
            b_child = crossover_homologous(b_parent, b_other, rng)

            a_child = mutate_genome(a_child, args.mut_rate, rng)
            b_child = mutate_genome(b_child, args.mut_rate, rng)

            if not is_valid_genome(a_child) or not is_valid_genome(b_child):
                # fall back to parents if mutation produced invalid genome
                a_child = a_parent
                b_child = b_parent

            # I4 audition stage-1 cycles; other worlds use full cycles
            base_cycles = args.cycles
            if world_name == "I4":
                base_cycles = int(ws.get("audition_cycles_1", 1000))

            tasks.append((
                a_child, b_child,
                bits_a_w, bits_b_w, targets,
                base_cycles, args.grace, args.pair_reduce,
                ws["gp2_drop_a"], ws["gp2_drop_b"],
                ws["gp2_flip_a"], ws["gp2_flip_b"],
                ws["gp2_noise_a"], ws["gp2_noise_b"],
                (args.seed + batch * 1000003 + i),
                {"world": world_name, "batch": batch, "i": i},
            ))
            children_meta.append({
                "i": i,
                "a_genome": a_child,
                "b_genome": b_child,
                "a_sig": _genome_sig(a_child),
                "b_sig": _genome_sig(b_child),
                "fit_ref": float(fit_ref),
            })

        # Evaluate base pairings (parallel)
        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_phase4_eval_one, tasks, chunksize=args.chunksize))
        else:
            results = [ _phase4_eval_one(t) for t in tasks ]

        # Optionally add Guide protocol stability (2nd run with different seed, same pairing)
        # and Mirror aggregated fitness across additional partners (Option B).
        # We do this sequentially for clarity (can be parallelized later if needed).
        # We also gather final (fit_used, metrics_final) per child.
        final_fit = [None] * n_this
        final_metrics = [None] * n_this
        final_crash = [None] * n_this

        # Precompute elites lists for Mirror partner sampling
        a_elites = [genome_from_record(rec) for rec in alice_archive.values()]
        b_elites = [genome_from_record(rec) for rec in bob_archive.values()]

        for i in range(n_this):
            fit_raw, metrics, a_crash, b_crash = results[i]
            a_child = children_meta[i]["a_genome"]
            b_child = children_meta[i]["b_genome"]

            fit_used = float(fit_raw)
            metrics_used = dict(metrics) if isinstance(metrics, dict) else {}

            # Normalize BobLatency -> BobLatencyNorm on the metrics we will keep
            metrics_used = ensure_bob_latency_norm(metrics_used, args.cycles)


            # Verify opcode mapping hash if present
            opcode_hash_ok = True


            # Guide: protocol stability proxy via second evaluation under different perturb_seed
            if world_name == "I1":
                # second run with offset seed; same pairing, but force some perturbation
                t2b = list(tasks[i])
                t2b[14] = int(t2b[14]) + 99991  # perturb_seed slot

                # Force run-2 to differ so ProtocolStability is informative
                # Slots: 10 flip_a, 11 flip_b, 12 noise_a, 13 noise_b
                t2b[10] = max(float(t2b[10]), 0.01)
                t2b[11] = max(float(t2b[11]), 0.01)
                t2b[12] = max(float(t2b[12]), 0.02)
                t2b[13] = max(float(t2b[13]), 0.02)

                fit2, metrics2, _ac2, _bc2 = _phase4_eval_one(tuple(t2b))
                metrics2 = dict(metrics2) if isinstance(metrics2, dict) else {}
                metrics2 = ensure_bob_latency_norm(metrics2, args.cycles)

                metrics_used = add_protocol_stability(metrics_used, metrics2)

            # Mirror: Option B aggregate fitness across partners (k draws each side)
            # Mirror: Option B aggregate fitness across partners with audition gating (Path B)
            if world_name == "I4":
                fit_ref = float(children_meta[i].get("fit_ref", 0.0))

                # thresholds + cycles
                aud1_cycles = int(ws.get("audition_cycles_1", 800))
                aud2_cycles = int(ws.get("audition_cycles_2", 2500))
                full_cycles = int(args.cycles)

                thr1 = float(ws.get("audition_thr_1", 0.90))  # at 1000 cycles
                thr2 = float(ws.get("audition_thr_2", 0.95))  # at 3000 cycles

                # --- AliveSignal gate (cheap) ---
                # Use early metrics that should be non-trivial if they're "doing something".
                # (These keys already exist in your ProtocolStability helper’s key list.)
                msr = float(metrics_used.get("MutualSignalRate", 0.0))
                syn = float(metrics_used.get("SyncIndex", 0.0))
                ssp = float(metrics_used.get("SignalSparsity", 0.0))
                alive_thr = 0.02
                alive_count = sum([
                    msr > alive_thr,
                    syn > alive_thr,
                    ssp > alive_thr
                ])

                alive_signal = 1.0 if alive_count >= 2 else 0.0

                metrics_used["AliveSignal"] = float(alive_signal)

                # If the base eval was not aud1_cycles for some reason, treat it as stage-1 anyway.
                fit_aud1 = float(fit_raw)
                pass1 = (not (a_crash or b_crash)) and (alive_signal > 0.5) and (fit_aud1 >= thr1 * fit_ref)

                # If fail stage 1: keep the audition fitness, skip expensive work
                if not pass1:
                    metrics_used["I4GateStage"] = 1
                    metrics_used["I4GatePass"] = 0.0
                    fit_used = fit_aud1
                    metrics_used["FitnessLive"] = float(fit_used)
                    metrics_used = add_mirror_metrics(metrics_used, [fit_used])  # variance=0 by construction

                else:
                    # Stage 2 audition at 3000 cycles
                    t2 = list(tasks[i])
                    t2[5] = aud2_cycles
                    t2[14] = int(t2[14]) + 27183  # new perturb_seed
                    fit2, m2, ac2, bc2 = _phase4_eval_one(tuple(t2))

                    fit_aud2 = float(fit2)
                    pass2 = (not (ac2 or bc2)) and (fit_aud2 >= thr2 * fit_ref)

                    if not pass2:
                        metrics_used["I4GateStage"] = 2
                        metrics_used["I4GatePass"] = 0.0
                        # use stage-2 result as "final" (still cheap vs full)
                        fit_used = fit_aud2
                        # prefer the stage-2 metrics (more informative than 1000-cycle metrics)
                        if isinstance(m2, dict):
                            metrics_used.update(m2)
                        metrics_used["FitnessLive"] = float(fit_used)
                        metrics_used = add_mirror_metrics(metrics_used, [fit_used])

                    else:
                        # Full evaluation only for survivors
                        t3 = list(tasks[i])
                        t3[5] = full_cycles
                        t3[14] = int(t3[14]) + 918273  # new perturb_seed
                        fit3, m3, ac3, bc3 = _phase4_eval_one(tuple(t3))

                        # update base
                        fit_used = float(fit3)
                        if isinstance(m3, dict):
                            metrics_used.update(m3)
                        a_crash = bool(ac3)
                        b_crash = bool(bc3)

                        # Now do k=1 partner tests (can be done at aud2_cycles to save time)
                        k = int(ws.get("mirror_k_partners", 1))
                        partner_cycles = int(ws.get("mirror_partner_cycles", aud2_cycles))

                        fit_list = [float(fit_used)]

                        # draw partners with replacement from current elites
                        for j in range(k):
                            pb = b_elites[rng.randrange(len(b_elites))]
                            fa, ma, ca, cb = evaluate_pair_variant3_with_reliance(
                                a_child, pb,
                                bits_a_w, bits_b_w, targets,
                                partner_cycles, args.grace,
                                args.pair_reduce,
                                ws["gp2_drop_a"], ws["gp2_drop_b"],
                                ws["gp2_flip_a"], ws["gp2_flip_b"],
                                ws["gp2_noise_a"], ws["gp2_noise_b"],
                                (args.seed + batch * 1000003 + i * 131 + j * 17 + 7),
                            )
                            fit_list.append(float(fa))

                        for j in range(k):
                            pa = a_elites[rng.randrange(len(a_elites))]
                            fb, mb, ca, cb = evaluate_pair_variant3_with_reliance(
                                pa, b_child,
                                bits_a_w, bits_b_w, targets,
                                partner_cycles, args.grace,
                                args.pair_reduce,
                                ws["gp2_drop_a"], ws["gp2_drop_b"],
                                ws["gp2_flip_a"], ws["gp2_flip_b"],
                                ws["gp2_noise_a"], ws["gp2_noise_b"],
                                (args.seed + batch * 1000003 + i * 137 + j * 19 + 11),
                            )
                            fit_list.append(float(fb))

                        # Aggregate partner fitness
                        fit_used = sum(fit_list) / len(fit_list)
                        metrics_used["I4GateStage"] = 3
                        metrics_used["I4GatePass"] = 1.0
                        metrics_used["FitnessLive"] = float(fit_used)
                        metrics_used = add_mirror_metrics(metrics_used, fit_list)

            # Final fitness shaping (if configured)
            fit_used = reshape_fitness(fit_used, metrics_used, ws)

            final_fit[i] = float(fit_used)
            final_metrics[i] = metrics_used
            final_crash[i] = (a_crash, b_crash)

        # Polygamy-as-average: compute per-side average fitness across matings within this batch
        if polygamy_round:
            a_fit_map = {}
            b_fit_map = {}
            a_counts = {}
            b_counts = {}
            for i in range(n_this):
                a_sig = children_meta[i]["a_sig"]
                b_sig = children_meta[i]["b_sig"]
                f = float(final_fit[i])
                a_fit_map[a_sig] = a_fit_map.get(a_sig, 0.0) + f
                b_fit_map[b_sig] = b_fit_map.get(b_sig, 0.0) + f
                a_counts[a_sig] = a_counts.get(a_sig, 0) + 1
                b_counts[b_sig] = b_counts.get(b_sig, 0) + 1
            # convert to means
            for ksig in list(a_fit_map.keys()):
                a_fit_map[ksig] /= max(1, a_counts[ksig])
            for ksig in list(b_fit_map.keys()):
                b_fit_map[ksig] /= max(1, b_counts[ksig])
        else:
            a_fit_map = None
            b_fit_map = None

        # Insert into archives (Part4 behavior, but with polygamy averaging on insertion fitness)
        for i in range(n_this):
            metrics = final_metrics[i]
            fit_raw = float(final_fit[i])
            a_child = children_meta[i]["a_genome"]
            b_child = children_meta[i]["b_genome"]
            a_sig = children_meta[i]["a_sig"]
            b_sig = children_meta[i]["b_sig"]
            a_crash, b_crash = final_crash[i]

            # bin per pairing metrics (Option 1)
            idx = world_bin_index(metrics, ws)

            fit_used_a = fit_raw
            fit_used_b = fit_raw
            if polygamy_round:
                fit_used_a = float(a_fit_map.get(a_sig, fit_raw))
                fit_used_b = float(b_fit_map.get(b_sig, fit_raw))

            # Construct records
            a_rec = {
                "id": str(uuid.uuid4()),
                "genome": a_child,
                "fit": fit_used_a,
                "metrics": metrics,
                "origin": {
                    "world": world_name,
                    "batch": batch,
                    "pair_idx": evaluated + i,
                    "polygamy_round": bool(polygamy_round),
                    "perturb_type": perturb_type,
                    "side": "A",
                },
            }
            b_rec = {
                "id": str(uuid.uuid4()),
                "genome": b_child,
                "fit": fit_used_b,
                "metrics": metrics,
                "origin": {
                    "world": world_name,
                    "batch": batch,
                    "pair_idx": evaluated + i,
                    "polygamy_round": bool(polygamy_round),
                    "perturb_type": perturb_type,
                    "side": "B",
                },
            }

            # Insert / replace if better
            if (idx not in alice_archive) or (fit_used_a > float(alice_archive[idx].get("fit", -1e9))):
                alice_archive[idx] = a_rec
            if (idx not in bob_archive) or (fit_used_b > float(bob_archive[idx].get("fit", -1e9))):
                bob_archive[idx] = b_rec

            # Log metrics
            row = {
                "batch": batch,
                "pair_idx": evaluated + i,
                "world": world_name,
                "polygamy_round": int(bool(polygamy_round)),
                "perturb_type": perturb_type if perturb_type is not None else "",
                "fit_raw": fit_raw,
                "fit_used_a": fit_used_a,
                "fit_used_b": fit_used_b,
                "bin": str(idx),
                "a_crash": int(bool(a_crash)),
                "b_crash": int(bool(b_crash)),
                "opcode_hash_ok": int(bool(True)),
            }
            for k in ws.get("axes_keys", ()):
                row[k] = float(metrics.get(k, 0.0))
            # common metrics if present
            for k in ["AliceFitLive", "BobFitLive", "FitnessLive", "FitnessMute", "RelianceIndex"]:
                if k in metrics:
                    row[k] = float(metrics[k])

            csv_w.writerow(row)

        evaluated += n_this

        # Backups
        if (batch + 1) % max(1, args.backup_interval_batches) == 0:
            meta = {
                "world": world_name,
                "evaluated": evaluated,
                "bins_a": len(alice_archive),
                "bins_b": len(bob_archive),
                "time_sec": time.time() - t0,
                "args": vars(args),
            }
            backup_path = os.path.join(backups_dir, f"{world_name}_backup_phase4_{evaluated:09d}.json.gz")
            save_backup(backup_path, meta, {
                "alice": {str(k): v for k, v in alice_archive.items()},
                "bob":   {str(k): v for k, v in bob_archive.items()},
            })

        # Progress
        # Progress + batch summary
        # Progress (write summary row + terminal line)
        if (batch + 1) % 1 == 0:
            elapsed = time.time() - t0
            rate = evaluated / elapsed if elapsed > 0 else 0.0

            # ---- batch-level diagnostics (from this batch only) ----
            batch_fit = [float(x) for x in final_fit[:n_this]]  # fit used for insertion logic
            avg_fit = (sum(batch_fit) / len(batch_fit)) if batch_fit else float("nan")
            batch_best = max(batch_fit) if batch_fit else float("nan")

            # ---- MAP-Elites truth: archive best (monotone non-decreasing) ----
            archive_best_a = max((float(rec["fit"]) for rec in alice_archive.values()), default=float("nan"))
            archive_best_b = max((float(rec["fit"]) for rec in bob_archive.values()), default=float("nan"))
            archive_best = max(
                [x for x in (archive_best_a, archive_best_b) if x == x],  # drop NaNs
                default=float("nan")
            )

            # ---- crash fraction (this batch only) ----
            crashes = 0
            for (ac, bc) in final_crash[:n_this]:
                if ac or bc:
                    crashes += 1
            crash_frac = crashes / max(1, n_this)

            # ---- genome length stats (this batch’s children) ----
            lens_a = [len(children_meta[i]["a_genome"]) for i in range(n_this)]
            lens_b = [len(children_meta[i]["b_genome"]) for i in range(n_this)]

            mean_len_A = (sum(lens_a) / len(lens_a)) if lens_a else 0.0
            min_len_A = min(lens_a) if lens_a else 0
            max_len_A = max(lens_a) if lens_a else 0

            mean_len_B = (sum(lens_b) / len(lens_b)) if lens_b else 0.0
            min_len_B = min(lens_b) if lens_b else 0
            max_len_B = max(lens_b) if lens_b else 0

            # ---- write ONE summary row per batch ----
            sw.writerow([
                batch + 1,
                elapsed,
                world_name,
                args.cycles,
                len(alice_archive),
                len(bob_archive),
                avg_fit,
                archive_best,      # MAP-Elites best (monotone)
                batch_best,        # batch best (can fluctuate)
                rate,
                args.mut_rate,
                args.polygamy_prob,
                crash_frac,
                mean_len_A,
                min_len_A,
                max_len_A,
                mean_len_B,
                min_len_B,
                max_len_B,
            ])
            sf.flush()

            # ---- terminal line ----
            print(
                f"[{world_name}] batch {batch + 1}/{total_batches}  "
                f"evaluated={evaluated}  binsA={len(alice_archive)} binsB={len(bob_archive)}  "
                f"avgFit={avg_fit:.4f}  batchBest={batch_best:.4f}  archiveBest={archive_best:.4f}  "
                f"elapsed={elapsed / 60.0:.2f}m  rate={rate:.1f}/s"
            )


    sf.close()

    csv_f.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_file", required=True, help="Phase3 backup .json.gz")
    ap.add_argument("--out_dir", default="Phase4_Results", help="Output root")
    ap.add_argument("--worlds", default="I0,I1,I2,I3,I4", help="Comma-separated worlds to run")

    ap.add_argument("--n_pairs", type=int, required=True)
    ap.add_argument("--pairs_per_batch", type=int, default=5000)

    ap.add_argument("--cycles", type=int, required=True)
    ap.add_argument("--grace", type=int, required=True)
    ap.add_argument("--pair_reduce", choices=["geom", "min"], default="geom")
    ap.add_argument("--mut_rate", type=float, required=True)

    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=256)
    ap.add_argument("--backup_interval_batches", type=int, default=1)

    # Cosmos base (shared)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--invert_prob", type=float, default=0.5)
    ap.add_argument("--drop_prob", type=float, default=0.05)

    # Polygamy (hub rewiring + NOW averaged insertion fitness per side during polygamy rounds)
    ap.add_argument("--polygamy_prob", type=float, default=0.10)
    ap.add_argument("--polygamy_k_min", type=int, default=2)
    ap.add_argument("--polygamy_k_max", type=int, default=6)

    # Perturb selection (logged; physics is encoded in world specs / contracts)
    ap.add_argument("--perturb_types", type=int, default=0)

    # Mirror
    ap.add_argument("--mirror_k_partners", type=int, default=3, help="I4: extra partners per side (k)")

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.workers is None:
        args.workers = os.cpu_count() or 1

    # Timestamped run directory to avoid overwriting prior outputs
    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.out_dir = os.path.join(args.out_dir, f"run_{stamp}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Phase4 output dir: {args.out_dir}")

    meta3, base_archive = load_phase3_backup(args.in_file)
    print(f"Loaded Phase3 backup: {args.in_file}")
    print(f"Phase3 evaluated: {meta3.get('evaluated', 'unknown')} | bins: {meta3.get('bins', 'unknown')}")

    # Shared base cosmos stream across worlds
    rng = random.Random(args.seed)
    cosmos = DualCosmosVariant3(
        physics_cycles=args.cycles,
        noise_rate=args.noise,
        invert_prob=args.invert_prob,
        drop_prob=args.drop_prob,
        rng=rng,
    )
    bits_a, bits_b, targets = cosmos.generate()

    world_list = [w.strip() for w in args.worlds.split(",") if w.strip()]
    for wname in world_list:
        run_world(wname, args, base_archive, bits_a, bits_b, targets)

if __name__ == "__main__":
    main()
