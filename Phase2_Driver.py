#!/usr/bin/env python3
"""
Phase 2 - Part 4: Cross-world mixing + evaluation + combined Map-Elites archive
with SOCIAL CHANNEL on GP2 (LIVE partners, one-way) + 100-batch shuffling.

PARALLELIZATION ONLY CHANGE:
- The child evaluation loop is executed in parallel in CHUNKS.
- Archive insertion is still applied IN THE SAME i-order as the serial loop.
- All other logic is untouched.
"""

import argparse
import gzip
import json
import os
import random
import time
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple

import concurrent.futures

import Pic10Sim


# ----------------------------
# Frozen ISA assumptions
# ----------------------------
N_OPCODES = 41
MAX_LINES = 256
GRID_SIZE = 16

OPCODE_LIST = [
    'ADDWF_W', 'ADDWF_F', 'ANDWF_W', 'ANDWF_F',
    'CLRF', 'CLRW', 'COMF_W', 'COMF_F',
    'DECF_W', 'DECF_F', 'DECFSZ_W', 'DECFSZ_F',
    'INCF_W', 'INCF_F', 'INCFSZ_W', 'INCFSZ_F',
    'IORWF_W', 'IORWF_F', 'MOVF_W', 'MOVF_F',
    'MOVWF', 'MOVLW', 'NOP', 'RLF_W', 'RLF_F',
    'RRF_W', 'RRF_F', 'SUBWF_W', 'SUBWF_F',
    'SWAPF_W', 'SWAPF_F', 'XORWF_W', 'XORWF_F',
    'BCF', 'BSF', 'BTFSC', 'BTFSS',
    'CALL', 'GOTO', 'RETLW', 'DELAY_MACRO'
]
OPCODE_HASH = hashlib.sha256(("\n".join(OPCODE_LIST)).encode("utf-8")).hexdigest()[:16]


# ----------------------------
# Cosmos (same as Phase 1)
# ----------------------------
SILENCE = [0, 0, 0, 0, 0, 0, 0, 0]
HUM     = [1, 1, 1, 1, 0, 0, 0, 0]
DUET    = [1, 1, 0, 0, 1, 1, 0, 0]
WARBLE  = [1, 0, 1, 0, 1, 0, 1, 0]

SIGNALS = [(0, SILENCE), (1, HUM), (2, DUET), (3, WARBLE)]
GAP_MIN = 4
GAP_MAX = 20

class Cosmos:
    def __init__(self, physics_cycles: int, noise_rate: float, invert_prob: float, rng: random.Random):
        self.cycles = physics_cycles
        self.noise = noise_rate
        self.inv_p = invert_prob
        self.rng = rng
        self.rts_state = 0

    def generate(self) -> Tuple[List[int], List[int]]:
        bits: List[int] = []
        targets: List[int] = []

        while len(bits) < self.cycles:
            target, pattern = self.rng.choice(SIGNALS)
            invert = (self.rng.random() < self.inv_p)

            n_packets = self.rng.randint(15, 25)
            burst_len = n_packets * 8

            for _ in range(burst_len):
                if len(bits) >= self.cycles:
                    break

                b = pattern[len(bits) % 8]
                if invert:
                    b = 1 - b

                if self.rng.random() < self.noise:
                    self.rts_state ^= 1
                b ^= self.rts_state

                bits.append(b)
                targets.append(target)

            gap = self.rng.randint(GAP_MIN, GAP_MAX)
            for _ in range(gap):
                if len(bits) >= self.cycles:
                    break
                bits.append(0)
                targets.append(0)

        return bits, targets


# ----------------------------
# Genome validity
# ----------------------------
def is_valid_instruction(inst: Any) -> bool:
    if not isinstance(inst, list) or len(inst) != 2:
        return False
    op_id, operand = inst
    if not isinstance(op_id, int) or not isinstance(operand, int):
        return False
    if op_id < 0 or op_id >= N_OPCODES:
        return False
    if operand < 0 or operand > 255:
        return False
    return True

def is_valid_genome(genome: Any) -> bool:
    return isinstance(genome, list) and len(genome) >= 1 and all(is_valid_instruction(x) for x in genome)


# ----------------------------
# Genetics (same kernel as Part2/Part3)
# ----------------------------
def random_instruction(rng: random.Random) -> List[int]:
    return [rng.randrange(N_OPCODES), rng.randrange(256)]

def mutate_genome(parent: List[List[int]], rate: float, rng: random.Random) -> List[List[int]]:
    child = [list(inst) for inst in parent]
    for i in range(len(child)):
        if rng.random() < rate:
            if rng.random() < 0.5:
                child[i][0] = rng.randrange(N_OPCODES)
            else:
                child[i][1] = rng.randrange(256)
    if rng.random() < rate * 0.5:
        if len(child) < MAX_LINES and rng.random() < 0.5:
            idx = rng.randrange(len(child) + 1)
            child.insert(idx, random_instruction(rng))
        elif len(child) > 1:
            idx = rng.randrange(len(child))
            child.pop(idx)
    return child

def crossover_homologous(p1: List[List[int]], p2: List[List[int]], rng: random.Random) -> List[List[int]]:
    roll = rng.random()
    if roll < 0.85:
        cut = rng.randrange(0, min(len(p1), len(p2)) + 1)
        child = [list(x) for x in p1[:cut]] + [list(x) for x in p2[cut:]]
        return child[:MAX_LINES]
    if roll < 0.97:
        if len(p1) < 2:
            return [list(x) for x in p2[:MAX_LINES]]
        a = rng.randrange(len(p1))
        b = rng.randrange(len(p1))
        if a > b:
            a, b = b, a
        if a == b:
            b = min(len(p1), a + 1)
        block = [list(x) for x in p1[a:b]]
        insert_at = rng.randrange(len(p2) + 1)
        child = [list(x) for x in p2[:insert_at]] + block + [list(x) for x in p2[insert_at:]]
        return child[:MAX_LINES]
    L = max(len(p1), len(p2))
    child: List[List[int]] = []
    for i in range(L):
        if i < len(p1) and i < len(p2):
            child.append(list(p1[i]) if rng.random() < 0.5 else list(p2[i]))
        elif i < len(p1):
            child.append(list(p1[i]))
        else:
            child.append(list(p2[i]))
        if len(child) >= MAX_LINES:
            break
    if len(child) == 0:
        child = [random_instruction(rng)]
    return child


# ----------------------------
# MAP binning (same axes)
# ----------------------------
def get_grid_index(metrics: Dict[str, float]) -> Tuple[int, int, int, int]:
    vals = [
        float(metrics["Activity"]),
        float(metrics["Responsiveness"]),
        float(metrics["ChannelEntropy"]),
        float(metrics["AlgoDensity"]),
    ]
    idxs = []
    for v in vals:
        v = 0.0 if v < 0.0 else 0.999 if v >= 1.0 else v
        idxs.append(int(v * GRID_SIZE))
    return tuple(idxs)  # type: ignore


# ----------------------------
# Evaluation: LIVE Bob -> Alice GP2 (one-way)
# ----------------------------
def evaluate_alice_with_live_bob(
    alice_genome: List[List[int]],
    bob_genome: List[List[int]],
    cosmos_bits: List[int],
    cosmos_targets: List[int],
    physics_cycles: int,
    grace_after_switch: int
) -> Tuple[float, Dict[str, float], bool, bool]:
    """
    Runs Bob and Alice in lockstep for physics_cycles.
      Bob: GP2=0, GP3=cosmos
      Alice: GP2 = (Bob_out & 1), GP3=cosmos

    Returns:
      (fitness, metrics, alice_crashed, bob_crashed)
    """
    alice = Pic10Sim.Pic10Sim()
    alice.reset()
    alice.load(alice_genome, OPCODE_LIST)

    bob = Pic10Sim.Pic10Sim()
    bob.reset()
    bob.load(bob_genome, OPCODE_LIST)

    outputs: List[int] = []
    correct = 0
    valid = 0
    toggles = 0
    last_out = None

    grace = 0
    last_target = None

    bob_crashed = False
    last_bob_gp0 = 0  # if Bob crashes, it stays 0

    for t in range(physics_cycles):
        bit3 = cosmos_bits[t]
        target = cosmos_targets[t]

        # Step Bob first (one-way coupling)
        if not bob_crashed:
            bob_out, bob_crash_now = bob.emulate_cycle(gp2_input=0, gp3_input=bit3)
            if bob_crash_now:
                bob_crashed = True
                last_bob_gp0 = 0
            else:
                last_bob_gp0 = int(bob_out) & 0x01

        # Step Alice with GP2 from Bob GP0
        alice_out, alice_crash_now = alice.emulate_cycle(gp2_input=last_bob_gp0, gp3_input=bit3)
        if alice_crash_now:
            return 0.0, {}, True, bob_crashed

        alice_out = int(alice_out)
        outputs.append(alice_out)

        if last_out is not None and alice_out != last_out:
            toggles += 1
        last_out = alice_out

        if last_target is None:
            last_target = target

        if target != last_target:
            grace = grace_after_switch
            last_target = target

        if grace > 0:
            grace -= 1
            continue

        valid += 1
        if alice_out == target:
            correct += 1

    fitness = (correct / valid) if valid > 0 else 0.0

    import zlib
    raw = bytes(outputs)
    comp = zlib.compress(raw, level=9)
    entropy = (len(comp) / len(raw)) if len(raw) > 0 else 0.0

    metrics = {
        "Activity": float(toggles / physics_cycles),
        "Responsiveness": float(fitness),
        "ChannelEntropy": float(entropy),
        "AlgoDensity": float(len(alice_genome) / MAX_LINES),
    }
    return float(fitness), metrics, False, bob_crashed


# ----------------------------
# Archive loading
# ----------------------------
def load_archive_file(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def extract_elites(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    archive = payload.get("archive", {})
    if not isinstance(archive, dict):
        raise ValueError("payload['archive'] is not a dict")
    elites: List[Dict[str, Any]] = []
    for _, rec in archive.items():
        if isinstance(rec, dict) and "genome" in rec:
            elites.append(rec)
    return elites


# ----------------------------
# PARALLELIZATION ONLY: worker function
# ----------------------------
def _eval_task(task: Tuple[List[List[int]], List[List[int]], List[int], List[int], int, int]):
    child, bob, cosmos_bits, cosmos_targets, cycles, grace = task
    return evaluate_alice_with_live_bob(child, bob, cosmos_bits, cosmos_targets, cycles, grace)

def _eval_task_list(tasks: List[Tuple[List[List[int]], List[List[int]], List[int], List[int], int, int]]):
    return [_eval_task(t) for t in tasks]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing backup_Wk_25000.json.gz files")
    ap.add_argument("--out_file", default="phase2_social_live_archive_25000.json.gz",
                    help="Output combined Phase-2 archive (.json.gz)")
    ap.add_argument("--n_children", type=int, default=20000,
                    help="Number of cross-world children to generate/evaluate")
    ap.add_argument("--mut_rate", type=float, default=0.10,
                    help="Mutation rate applied after crossover")
    ap.add_argument("--seed", type=int, default=888888,
                    help="RNG seed for Phase2")
    ap.add_argument("--cycles", type=int, default=12000,
                    help="PHYSICS_CYCLES for evaluation")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="NOISE_RATE for cosmos coherence")
    ap.add_argument("--invert_prob", type=float, default=0.5,
                    help="Polarity inversion probability per burst")
    ap.add_argument("--grace", type=int, default=16,
                    help="Grace window after target switch")

    ap.add_argument("--seed_with_parents", action="store_true",
                    help="Seed archive with parent elites (GP2=0 baseline, no social coupling)")

    ap.add_argument("--children_per_batch", type=int, default=50,
                    help="Size of a Phase2 'batch' for shuffling purposes (default 50)")
    ap.add_argument("--shuffle_interval_batches", type=int, default=100,
                    help="Shuffle live partner assignments every N batches (default 100)")

    # Parallel knobs
    ap.add_argument("--workers", type=int, default=None,
                    help="Default = cpu_count-2 (clamped).")
    ap.add_argument("--chunk_size", type=int, default=200,
                    help="Children evaluated per submitted chunk (default 200).")

    args = ap.parse_args()

    cpu = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else max(1, cpu - 2)
    workers = max(1, min(workers, max(1, cpu - 2)))  # clamp to cpu-2

    rng = random.Random(args.seed)
    run_uuid = str(uuid.uuid4())
    ts = datetime.now().isoformat(timespec="seconds")

    print("Phase2-Part4 starting (LIVE Bob -> Alice GP2, one-way, 100-batch shuffling) [PARALLEL]")
    print(f"  in_dir: {args.in_dir}")
    print(f"  out_file: {args.out_file}")
    print(f"  n_children: {args.n_children}")
    print(f"  mut_rate: {args.mut_rate}")
    print(f"  cycles: {args.cycles}  noise: {args.noise}  invert_prob: {args.invert_prob}  grace: {args.grace}")
    print(f"  children_per_batch: {args.children_per_batch}  shuffle_interval_batches: {args.shuffle_interval_batches}")
    print(f"  workers: {workers}  chunk_size: {args.chunk_size}")
    print(f"  run_uuid: {run_uuid}")
    print("")

    # -----------------------------
    # Results directory + filenames
    # -----------------------------
    base_results_dir = "Phase2_Results"
    os.makedirs(base_results_dir, exist_ok=True)

    safe_ts = ts.replace(":", "-")
    run_dir = os.path.join(base_results_dir, f"run_{safe_ts}_{run_uuid[:8]}")
    os.makedirs(run_dir, exist_ok=True)

    backups_dir = os.path.join(run_dir, "backups")
    os.makedirs(backups_dir, exist_ok=True)

    out_file_path = os.path.join(run_dir, os.path.basename(args.out_file))

    metrics_path = os.path.join(run_dir, "Metrics_Phase2.csv")
    metrics_f = open(metrics_path, "w", encoding="utf-8")
    metrics_f.write(
        "step,batch,elapsed_s,pop_bins,inserted,valid,crashes,crash_frac,bob_crashes,"
        "bob_crash_frac_valid,rate_per_s,best_fit\n"
    )
    metrics_f.flush()

    print(f"[results] run_dir      : {run_dir}")
    print(f"[results] out_file     : {out_file_path}")
    print(f"[results] metrics_csv  : {metrics_path}")
    print(f"[results] backups_dir  : {backups_dir}")
    print("")

    # -----------------------------
    # Load world archives (parents)
    # -----------------------------
    world_elites: Dict[int, List[Dict[str, Any]]] = {}
    source_meta: Dict[int, Any] = {}

    for w in range(10):
        fname = os.path.join(args.in_dir, f"backup_W{w}_25000.json.gz")
        payload = load_archive_file(fname)
        source_meta[w] = payload.get("meta", {})
        elites = extract_elites(payload)

        filtered = []
        for rec in elites:
            g = rec.get("genome", None)
            if is_valid_genome(g):
                filtered.append(rec)
        if not filtered:
            raise ValueError(f"W{w}: no valid parent genomes")

        world_elites[w] = filtered
        print(f"W{w}: usable parents={len(filtered)}")

    # -----------------------------
    # Build shared cosmos (once)
    # -----------------------------
    cosmos_rng = random.Random(rng.randrange(10**9))
    cosmos = Cosmos(args.cycles, args.noise, args.invert_prob, cosmos_rng)
    cosmos_bits, cosmos_targets = cosmos.generate()

    # -----------------------------
    # Combined archive + counters
    # -----------------------------
    archive: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}

    start = time.time()
    crashes = 0
    valid = 0
    inserted = 0
    bob_crashes = 0
    best_fit_so_far = 0.0

    # -----------------------------
    # Optional: seed with parents
    # -----------------------------
    if args.seed_with_parents:
        print("\nSeeding archive with parent elites (GP2=0 baseline, no social coupling)...")
        seeded = 0
        silent_bob = [[OPCODE_LIST.index("NOP"), 0]]

        for w in range(10):
            for rec in world_elites[w]:
                g = rec["genome"]
                if not is_valid_genome(g):
                    continue

                fit, mets, alice_crashed, bob_crashed = evaluate_alice_with_live_bob(
                    g, silent_bob, cosmos_bits, cosmos_targets, args.cycles, args.grace
                )
                if alice_crashed or not mets:
                    continue

                idx = get_grid_index(mets)
                if idx not in archive or fit > archive[idx]["fitness"]:
                    archive[idx] = {
                        "genome": g,
                        "fitness": float(fit),
                        "metrics": mets,
                        "origin": {"type": "parent", "world": w, "gp2_mode": "held_low_live_baseline"}
                    }
                    best_fit_so_far = max(best_fit_so_far, float(fit))

                seeded += 1

        print(f"Seeded from {seeded} parent evals. bins={len(archive)}")

        seed_backup_path = os.path.join(backups_dir, "backup_phase2_seed.json.gz")
        meta_seed = {
            "phase2_part": 4,
            "run_uuid": run_uuid,
            "timestamp": ts,
            "note": "archive after seeding with parents (GP2=0 baseline)",
            "params": {
                "in_dir": args.in_dir,
                "n_children": args.n_children,
                "mut_rate": args.mut_rate,
                "cycles": args.cycles,
                "noise": args.noise,
                "invert_prob": args.invert_prob,
                "grace": args.grace,
                "children_per_batch": args.children_per_batch,
                "shuffle_interval_batches": args.shuffle_interval_batches,
                "seed_with_parents": True,
                "workers": workers,
                "chunk_size": args.chunk_size,
            },
            "source_meta": source_meta,
        }
        with gzip.open(seed_backup_path, "wt", encoding="utf-8") as f:
            json.dump({"meta": meta_seed, "archive": {str(k): v for k, v in archive.items()}}, f)

        elapsed0 = time.time() - start
        metrics_f.write(
            f"0,0,{elapsed0:.3f},{len(archive)},{inserted},{valid},{crashes},0.0,"
            f"{bob_crashes},0.0,0.0,{best_fit_so_far:.6f}\n"
        )
        metrics_f.flush()

    print("\nGenerating children with LIVE Bob->Alice coupling...")

    # -----------------------------
    # Live partner assignment table (unchanged)
    # -----------------------------
    def make_partner_table() -> List[Tuple[int, List[List[int]]]]:
        table = []
        for _ in range(args.children_per_batch):
            w_bob = rng.randrange(10)
            bob = rng.choice(world_elites[w_bob])["genome"]
            table.append((w_bob, bob))
        return table

    partner_table = make_partner_table()
    batches_until_shuffle = args.shuffle_interval_batches

    # -----------------------------
    # Backup policy (unchanged)
    # -----------------------------
    BACKUP_INTERVAL_BATCHES = 50
    checkpoint_every_children = args.children_per_batch * BACKUP_INTERVAL_BATCHES
    next_checkpoint_step = checkpoint_every_children

    # -----------------------------
    # Chunk builder: produces per-child slot_records + a task list for workers
    # (Workers evaluate only valid tasks; invalid slots are None)
    # -----------------------------
    def build_chunk(i_start: int, i_end: int):
        nonlocal partner_table, batches_until_shuffle

        tasks: List[Tuple[List[List[int]], List[List[int]], List[int], List[int], int, int]] = []
        slot_records: List[Any] = []

        for i in range(i_start, i_end + 1):
            slot = (i - 1) % args.children_per_batch

            if slot == 0:
                current_batch = ((i - 1) // args.children_per_batch) + 1
                if batches_until_shuffle <= 0:
                    partner_table = make_partner_table()
                    batches_until_shuffle = args.shuffle_interval_batches
                    print(f"  [SHUFFLE] batch={current_batch} partner table reshuffled")
                batches_until_shuffle -= 1

            wa = rng.randrange(10)
            wb = rng.randrange(10)
            while wb == wa:
                wb = rng.randrange(10)

            p1 = rng.choice(world_elites[wa])["genome"]
            p2 = rng.choice(world_elites[wb])["genome"]

            child = mutate_genome(crossover_homologous(p1, p2, rng), args.mut_rate, rng)
            if not is_valid_genome(child):
                slot_records.append(None)
                continue

            w_bob, bob = partner_table[slot]
            if not is_valid_genome(bob):
                slot_records.append(None)
                continue

            origin = {
                "type": "hybrid_social_live_one_way",
                "world_a": wa,
                "world_b": wb,
                "bob_world": w_bob,
                "gp2_source": "bob_gp0_live",
                "pairing": "fixed_within_shuffle_window",
                "shuffle_interval_batches": args.shuffle_interval_batches,
                "children_per_batch": args.children_per_batch,
                "bob_crashed": None,
            }

            slot_records.append({"child": child, "origin": origin})

            # Worker task includes cosmos + params (no initializer needed)
            tasks.append((child, bob, cosmos_bits, cosmos_targets, args.cycles, args.grace))

        return tasks, slot_records

    # -----------------------------
    # PARALLEL execution:
    # We keep many chunk futures inflight, collect as completed,
    # but APPLY in strict chunk order to preserve serial semantics.
    # -----------------------------
    completed = 0
    next_i = 1

    chunk_id = 0
    next_apply_chunk = 0

    inflight = {}        # chunk_id -> future
    chunk_slots = {}     # chunk_id -> slot_records
    chunk_i_range = {}   # chunk_id -> (i_start, i_end)

    max_inflight = max(4, workers * 4)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:

        # helper to submit one chunk
        def submit_one():
            nonlocal next_i, chunk_id
            if next_i > args.n_children:
                return False
            i_start = next_i
            i_end = min(args.n_children, i_start + args.chunk_size - 1)
            tasks, slots = build_chunk(i_start, i_end)

            fut = ex.submit(_eval_task_list, tasks)  # returns list[(fit, mets, alice_crashed, bob_crashed)]
            inflight[chunk_id] = fut
            chunk_slots[chunk_id] = slots
            chunk_i_range[chunk_id] = (i_start, i_end)

            chunk_id += 1
            next_i = i_end + 1
            return True

        # prime pipeline
        while len(inflight) < max_inflight and next_i <= args.n_children:
            submit_one()

        # store completed results until we can apply in order
        completed_results = {}  # chunk_id -> results list

        while inflight or completed_results:
            # pull any finished futures
            done, _ = concurrent.futures.wait(
                inflight.values(),
                timeout=0.1,
                return_when=concurrent.futures.FIRST_COMPLETED
            )

            for fut in done:
                # locate its chunk_id
                cid = None
                for k, v in inflight.items():
                    if v is fut:
                        cid = k
                        break
                if cid is None:
                    continue
                results = fut.result()
                completed_results[cid] = results
                del inflight[cid]

            # refill pipeline
            while len(inflight) < max_inflight and next_i <= args.n_children:
                submit_one()

            # apply any available chunks in strict order
            while next_apply_chunk in completed_results:
                results = completed_results.pop(next_apply_chunk)
                slots = chunk_slots.pop(next_apply_chunk)
                chunk_slots.pop(next_apply_chunk, None)
                chunk_i_range.pop(next_apply_chunk, None)

                r_idx = 0
                for slot in slots:
                    completed += 1

                    if slot is None:
                        crashes += 1
                    else:
                        fit, mets, alice_crashed, bob_crashed = results[r_idx]
                        r_idx += 1

                        if alice_crashed or not mets:
                            crashes += 1
                        else:
                            valid += 1
                            if bob_crashed:
                                bob_crashes += 1

                            idx = get_grid_index(mets)
                            origin = slot["origin"]
                            origin["bob_crashed"] = bool(bob_crashed)

                            elite = {"genome": slot["child"], "fitness": float(fit), "metrics": mets, "origin": origin}

                            if idx not in archive or float(fit) > float(archive[idx]["fitness"]):
                                archive[idx] = elite
                                inserted += 1
                                if float(fit) > best_fit_so_far:
                                    best_fit_so_far = float(fit)

                    # ---- METRICS/PRINT (every 1000 children) ----
                    if completed % 1000 == 0:
                        elapsed = time.time() - start
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        crash_frac = (crashes / completed) if completed > 0 else 0.0
                        bob_crash_frac_valid = (bob_crashes / max(1, valid)) if valid > 0 else 0.0

                        print(
                            f"{completed:7d}/{args.n_children} | bins={len(archive):5d} | inserted={inserted:5d} "
                            f"| crash_frac={crash_frac:.3f} | bob_crash_frac(valid)={bob_crash_frac_valid:.3f} "
                            f"| best_fit={best_fit_so_far:.6f} | rate={rate:.1f}/s"
                        )

                        current_batch = ((completed - 1) // args.children_per_batch) + 1
                        metrics_f.write(
                            f"{completed},{current_batch},{elapsed:.3f},{len(archive)},{inserted},{valid},{crashes},{crash_frac:.6f},"
                            f"{bob_crashes},{bob_crash_frac_valid:.6f},{rate:.3f},{best_fit_so_far:.6f}\n"
                        )
                        if completed % 10000 == 0:
                            metrics_f.flush()

                    # ---- CHECKPOINT BACKUP ----
                    if completed >= next_checkpoint_step:
                        current_batch = ((completed - 1) // args.children_per_batch) + 1
                        backup_path = os.path.join(backups_dir, f"backup_phase2_{completed}.json.gz")

                        meta_ckpt = {
                            "phase2_part": 4,
                            "run_uuid": run_uuid,
                            "timestamp": ts,
                            "checkpoint": {
                                "step": completed,
                                "batch": current_batch,
                                "note": f"checkpoint every {BACKUP_INTERVAL_BATCHES} batches",
                            },
                            "params": {
                                "in_dir": args.in_dir,
                                "n_children": args.n_children,
                                "mut_rate": args.mut_rate,
                                "seed": args.seed,
                                "cycles": args.cycles,
                                "noise": args.noise,
                                "invert_prob": args.invert_prob,
                                "grace": args.grace,
                                "children_per_batch": args.children_per_batch,
                                "shuffle_interval_batches": args.shuffle_interval_batches,
                                "seed_with_parents": bool(args.seed_with_parents),
                                "workers": workers,
                                "chunk_size": args.chunk_size,
                            },
                            "results_so_far": {
                                "bins": len(archive),
                                "inserted": inserted,
                                "valid": valid,
                                "crashes": crashes,
                                "best_fit": best_fit_so_far,
                            },
                            "source_meta": source_meta,
                        }

                        with gzip.open(backup_path, "wt", encoding="utf-8") as f:
                            json.dump({"meta": meta_ckpt, "archive": {str(k): v for k, v in archive.items()}}, f)

                        next_checkpoint_step += checkpoint_every_children

                next_apply_chunk += 1

    # Final summary
    elapsed = time.time() - start
    crash_frac = crashes / args.n_children if args.n_children > 0 else 0.0
    bob_crash_frac_all = bob_crashes / args.n_children if args.n_children > 0 else 0.0
    bob_crash_frac_valid = bob_crashes / max(1, valid) if valid > 0 else 0.0
    rate = args.n_children / elapsed if elapsed > 0 else 0.0

    print("\nDone.")
    print(f"  evaluated: {args.n_children}")
    print(f"  valid: {valid}")
    print(f"  crashes/invalid: {crashes} (crash_frac={crash_frac:.3f})")
    print(f"  bob_crashed among valid evals: {bob_crashes} (bob_crash_frac_all={bob_crash_frac_all:.3f})")
    print(f"  bins in archive: {len(archive)}")
    print(f"  best_fit: {best_fit_so_far:.6f}")
    print(f"  elapsed: {elapsed:.1f}s\n")

    final_batch = ((args.n_children - 1) // args.children_per_batch) + 1 if args.n_children > 0 else 0
    metrics_f.write(
        f"{args.n_children},{final_batch},{elapsed:.3f},{len(archive)},{inserted},{valid},{crashes},{crash_frac:.6f},"
        f"{bob_crashes},{bob_crash_frac_valid:.6f},{rate:.3f},{best_fit_so_far:.6f}\n"
    )
    metrics_f.flush()
    metrics_f.close()

    meta_out = {
        "phase2_part": 4,
        "run_uuid": run_uuid,
        "timestamp": ts,
        "input_contract": "GP3=cosmos_input, GP2=LIVE Bob GP0 (one-way), GP0/GP1=outputs",
        "opcode_list_hash": hashlib.sha256("\n".join(OPCODE_LIST).encode("utf-8")).hexdigest()[:16],
        "source_archives": {"in_dir": args.in_dir, "files": [f"backup_W{w}_25000.json.gz" for w in range(10)]},
        "source_meta": source_meta,
        "params": {
            "n_children": args.n_children,
            "mut_rate": args.mut_rate,
            "seed": args.seed,
            "cycles": args.cycles,
            "noise": args.noise,
            "invert_prob": args.invert_prob,
            "grace": args.grace,
            "children_per_batch": args.children_per_batch,
            "shuffle_interval_batches": args.shuffle_interval_batches,
            "seed_with_parents": bool(args.seed_with_parents),
            "workers": workers,
            "chunk_size": args.chunk_size,
        },
        "results": {
            "bins": len(archive),
            "valid": valid,
            "crashes": crashes,
            "crash_frac": crash_frac,
            "bob_crashes": bob_crashes,
            "bob_crash_frac_all": bob_crash_frac_all,
            "inserted": inserted,
            "best_fit": best_fit_so_far,
        }
    }

    with gzip.open(out_file_path, "wt", encoding="utf-8") as f:
        json.dump({"meta": meta_out, "archive": {str(k): v for k, v in archive.items()}}, f)

    print(f"Wrote Phase2-Part4 archive: {out_file_path}")




if __name__ == "__main__":
    main()
