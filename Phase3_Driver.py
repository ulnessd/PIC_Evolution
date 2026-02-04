#!/usr/bin/env python3
"""
Phase 3 Nursery (Variant 3 "Intermittent Truth"):

- Reciprocal live coupling: Alice <-> Bob on GP2 using partner GP0 bit
- Cosmos does NOT wait: instruction cost dt from Pic10Sim.last_dt advances world-time
- Variant-3 physics on GP3:
    * shared target stream
    * independent noise (fog) per agent
    * intermittent dropout (one agent blind on some ticks)
- Reliance Index axis:
    RI = max(0, f_live - f_mute)
    where f_mute is evaluated with GP2 forced low both ways, SAME cosmos stream.

Archive axes (4D):
  - Activity
  - ChannelEntropy
  - AlgoDensity
  - RelianceIndex

Fitness used for selection within bin:
  - pair fitness (default: geometric mean of individual fitnesses)

Parallelization:
  - Evaluate children in parallel in CHUNKS (ProcessPoolExecutor)
  - Apply archive insertions in strict i-order (serial semantics preserved)

Input:
  --in_dir directory containing Phase1 backups: backup_W0_25000.json.gz ... backup_W9_25000.json.gz
  (same style as Phase2 input)

Output:
  Combined nursery archive with stored pair genomes (alice_genome, bob_genome)
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional


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
# Cosmos patterns (same library)
# ----------------------------
SILENCE = [0, 0, 0, 0, 0, 0, 0, 0]
HUM     = [1, 1, 1, 1, 0, 0, 0, 0]
DUET    = [1, 1, 0, 0, 1, 1, 0, 0]
WARBLE  = [1, 0, 1, 0, 1, 0, 1, 0]

SIGNALS = [(0, SILENCE), (1, HUM), (2, DUET),  (3, WARBLE)]  # tiny bias ok; can revert to strict list
# If you want exactly Phase2 list: SIGNALS = [(0,SILENCE),(1,HUM),(2,DUET),(3,WARBLE)]

GAP_MIN = 4
GAP_MAX = 20


class DualCosmosVariant3:
    """
    Generates a shared target stream + two observed input streams for Alice and Bob.

    Shared:
      - target label in {0,1,2,3} (per tick)
      - base bitstream from chosen pattern, with shared invert
    Per-agent:
      - independent RTS fog state (xor flips) with noise_rate
    Dropout:
      - with probability drop_prob at a tick, one agent is blinded (bit forced to 0)
      - which agent drops is random (50/50)

    This keeps the "truth" common but never perfectly present in one place.
    """
    def __init__(self, physics_cycles: int, noise_rate: float, invert_prob: float, drop_prob: float, rng: random.Random):
        self.cycles = physics_cycles
        self.noise = noise_rate
        self.inv_p = invert_prob
        self.drop = drop_prob
        self.rng = rng

        self.rts_a = 0
        self.rts_b = 0

    def generate(self) -> Tuple[List[int], List[int], List[int]]:
        bits_a: List[int] = []
        bits_b: List[int] = []
        targets: List[int] = []

        while len(targets) < self.cycles:
            target, pattern = self.rng.choice(SIGNALS)
            invert = (self.rng.random() < self.inv_p)

            n_packets = self.rng.randint(15, 25)
            burst_len = n_packets * 8

            for _ in range(burst_len):
                if len(targets) >= self.cycles:
                    break

                base = pattern[len(targets) % 8]
                if invert:
                    base = 1 - base

                # independent fog
                a = base
                b = base
                if self.rng.random() < self.noise:
                    self.rts_a ^= 1
                if self.rng.random() < self.noise:
                    self.rts_b ^= 1
                a ^= self.rts_a
                b ^= self.rts_b

                # intermittent dropout (one agent blinded)
                if self.rng.random() < self.drop:
                    if self.rng.random() < 0.5:
                        a = 0
                    else:
                        b = 0

                bits_a.append(a)
                bits_b.append(b)
                targets.append(target)

            gap = self.rng.randint(GAP_MIN, GAP_MAX)
            for _ in range(gap):
                if len(targets) >= self.cycles:
                    break

                # gap target is 0 and bit is 0 for both
                bits_a.append(0)
                bits_b.append(0)
                targets.append(0)

        return bits_a, bits_b, targets


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
# Genetics (same kernel style)
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
    # shuffle mix
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
# MAP binning (axes for Phase3 nursery)
# ----------------------------
def get_grid_index(metrics: Dict[str, float]) -> Tuple[int, int, int, int]:
    vals = [
        float(metrics["Activity"]),
        float(metrics["ChannelEntropy"]),
        float(metrics["AlgoDensity"]),
        float(metrics["RelianceIndex"]),
    ]
    idxs: List[int] = []
    for v in vals:
        v = 0.0 if v < 0.0 else 0.999 if v >= 1.0 else v
        idxs.append(int(v * GRID_SIZE))
    return tuple(idxs)  # type: ignore


# ----------------------------
# Scoring helpers
# ----------------------------
def _pair_reduce(fa: float, fb: float, mode: str) -> float:
    if mode == "min":
        return min(fa, fb)
    # default: geometric mean
    if fa <= 0.0 or fb <= 0.0:
        return 0.0
    return (fa * fb) ** 0.5


def _entropy_ratio(outputs: List[int]) -> float:
    """
    Phase3 ChannelEntropy: normalized Shannon entropy of Alice's 2-bit output stream.
    outputs are Alice's 2-bit values (0..3). We normalize by log2(4)=2 so result is in [0,1].
    This is intentionally DIFFERENT from Phase1/Phase2 entropy notions.
    """
    if not outputs:
        return 0.0

    # Count 2-bit symbols
    c0 = c1 = c2 = c3 = 0
    for o in outputs:
        s = int(o) & 3
        if s == 0:
            c0 += 1
        elif s == 1:
            c1 += 1
        elif s == 2:
            c2 += 1
        else:
            c3 += 1

    n = c0 + c1 + c2 + c3
    if n <= 0:
        return 0.0

    import math
    H = 0.0
    for c in (c0, c1, c2, c3):
        if c:
            p = c / n
            H -= p * math.log(p, 2)

    # Normalize to [0,1]
    return H / 2.0



# ----------------------------
# Evaluation: reciprocal coupling + dt + Variant3 physics + RelianceIndex
# ----------------------------
def evaluate_pair_variant3_with_reliance(
    alice_genome: List[List[int]],
    bob_genome: List[List[int]],
    bits_a: List[int],
    bits_b: List[int],
    targets: List[int],
    physics_cycles: int,
    grace_after_switch: int,
    pair_reduce_mode: str
) -> Tuple[float, Dict[str, float], bool, bool]:
    """
    Returns:
      (pair_fitness_live, metrics, alice_crashed, bob_crashed)

    Metrics includes RelianceIndex computed from (live - mute) under same cosmos streams.

    NOTE:
      - We compute f_live from reciprocal coupling.
      - We compute f_mute by rerunning same genomes with GP2 forced 0 both ways.
      - RelianceIndex = max(0, f_live - f_mute).
    """

    # ---------- helper to run one condition ----------
    def run_condition(muted: bool) -> Tuple[float, float, bool, bool, int, List[int]]:
        alice = Pic10Sim.Pic10Sim()
        alice.reset()
        alice.load(alice_genome, OPCODE_LIST)

        bob = Pic10Sim.Pic10Sim()
        bob.reset()
        bob.load(bob_genome, OPCODE_LIST)

        # per-agent scoring
        correct_a = 0
        correct_b = 0
        valid_a = 0
        valid_b = 0

        grace_a = 0
        grace_b = 0
        last_t_a: Optional[int] = None
        last_t_b: Optional[int] = None

        outputs_pair: List[int] = []  # record Alice 2-bit output only (consistent with prior)
        toggles = 0
        last_out = None

        a_crashed = False
        b_crashed = False

        last_a_gp0 = 0
        last_b_gp0 = 0

        t = 0
        while t < physics_cycles:
            in_a = bits_a[t]
            in_b = bits_b[t]

            # Step Alice
            gp2_a = 0 if muted else (last_b_gp0 & 1)
            out_a, crash_a = alice.emulate_cycle(gp2_input=gp2_a, gp3_input=in_a)
            if crash_a:
                a_crashed = True
                # crash policy: stop evaluation (non-viable)
                break
            out_a = int(out_a)
            last_a_gp0 = out_a & 1

            # Step Bob
            gp2_b = 0 if muted else (last_a_gp0 & 1)
            out_b, crash_b = bob.emulate_cycle(gp2_input=gp2_b, gp3_input=in_b)
            if crash_b:
                b_crashed = True
                break
            out_b = int(out_b)
            last_b_gp0 = out_b & 1

            # record Alice output history (as before)
            outputs_pair.append(out_a)
            if last_out is not None and out_a != last_out:
                toggles += 1
            last_out = out_a

            # dt: world does not wait
            dt_a = getattr(alice, "last_dt", 1)
            dt_b = getattr(bob, "last_dt", 1)
            try:
                dt = int(max(dt_a, dt_b))
            except Exception:
                dt = 1
            if dt < 1:
                dt = 1
            if t + dt > physics_cycles:
                dt = physics_cycles - t
                if dt < 1:
                    dt = 1

            # score for dt ticks using held outputs
            for k in range(dt):
                tgt = targets[t + k]

                # Alice
                if last_t_a is None:
                    last_t_a = tgt
                elif tgt != last_t_a:
                    grace_a = grace_after_switch
                    last_t_a = tgt

                if grace_a > 0:
                    grace_a -= 1
                else:
                    valid_a += 1
                    if out_a == tgt:
                        correct_a += 1

                # Bob
                if last_t_b is None:
                    last_t_b = tgt
                elif tgt != last_t_b:
                    grace_b = grace_after_switch
                    last_t_b = tgt

                if grace_b > 0:
                    grace_b -= 1
                else:
                    valid_b += 1
                    if out_b == tgt:
                        correct_b += 1

            t += dt

        if a_crashed or b_crashed:
            return 0.0, 0.0, a_crashed, b_crashed, toggles, outputs_pair

        fa = (correct_a / valid_a) if valid_a > 0 else 0.0
        fb = (correct_b / valid_b) if valid_b > 0 else 0.0
        return fa, fb, False, False, toggles, outputs_pair

    # ---------- live ----------
    fa_live, fb_live, a_crash, b_crash, toggles, outputs_live = run_condition(muted=False)
    if a_crash or b_crash:
        if a_crash or b_crash:
            return 0.0, {}, a_crash, b_crash

    f_live = _pair_reduce(fa_live, fb_live, pair_reduce_mode)

    # ---------- mute (same cosmos streams) ----------
    fa_mute, fb_mute, a_crash_m, b_crash_m, _, _ = run_condition(muted=True)
    # If mute crashes, treat mute fitness as 0 (max reliance). This is conservative.
    f_mute = 0.0
    if not (a_crash_m or b_crash_m):
        f_mute = _pair_reduce(fa_mute, fb_mute, pair_reduce_mode)

    reliance = f_live - f_mute
    if reliance < 0.0:
        reliance = 0.0

    entropy = _entropy_ratio(outputs_live)

    metrics = {
        "Activity": float(toggles / max(1, physics_cycles)),
        "ChannelEntropy": float(entropy),
        "AlgoDensity": float((len(alice_genome) + len(bob_genome)) / (2.0 * MAX_LINES)),
        "RelianceIndex": float(reliance),

        # Not axes, but useful diagnostics:
        "FitnessLive": float(f_live),
        "FitnessMute": float(f_mute),
        "AliceFitLive": float(fa_live),
        "BobFitLive": float(fb_live),
    }

    return float(f_live), metrics, False, False


# ----------------------------
# Archive loading (Phase1 backups)
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
# Worker functions
# ----------------------------
def _eval_task(task: Tuple[List[List[int]], List[List[int]], List[int], List[int], List[int], int, int, str]):
    alice_g, bob_g, bits_a, bits_b, targets, cycles, grace, mode = task
    return evaluate_pair_variant3_with_reliance(alice_g, bob_g, bits_a, bits_b, targets, cycles, grace, mode)


def _eval_task_list(tasks: List[Tuple[List[List[int]], List[List[int]], List[int], List[int], List[int], int, int, str]]):
    return [_eval_task(t) for t in tasks]


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    import csv
    import gzip
    import json
    import os
    import random
    import time
    import uuid
    from concurrent.futures import ProcessPoolExecutor

    # ----------------------------
    # Argparse
    # ----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, help="Phase2 archive .json.gz (e.g., backup_phase2_40000000.json.gz)")
    ap.add_argument("--n_children", type=int, required=True, help="Number of Alice/Bob child-pairs to generate/evaluate")
    ap.add_argument("--cycles", type=int, required=True, help="Physics cycles per evaluation")
    ap.add_argument("--noise", type=float, required=True, help="Telegraph noise rate")
    ap.add_argument("--invert_prob", type=float, required=True, help="Burst polarity inversion probability")
    ap.add_argument("--drop_prob", type=float, required=True, help="Variant3 dropout probability")
    ap.add_argument("--grace", type=int, required=True, help="Grace window after target switch")

    ap.add_argument("--mut_rate", type=float, required=True, help="Mutation rate")
    ap.add_argument("--pair_reduce", type=str, default="geom", choices=["geom", "min"],
                    help="Pair fitness reduction for Alice/Bob (geom or min)")

    ap.add_argument("--children_per_batch", type=int, default=200000, help="How many child-pairs per batch")
    ap.add_argument("--backup_interval_batches", type=int, default=50, help="Backup every N batches")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--workers", type=int, default=None, help="Override worker count (default: os.cpu_count())")
    ap.add_argument("--chunk_pairs", type=int, default=8192,
                    help="How many pairs to bundle per process task (reduces IPC overhead)")
    ap.add_argument("--bootstrap_pairs", type=int, default=1000000,
                    help="Number of Phase2 random pairs to evaluate to seed Phase3 archive before nursery reproduction")

    args = ap.parse_args()

    # ----------------------------
    # Parallel setup
    # ----------------------------
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    print(f"Workers: {workers}")


    # ----------------------------
    # Run dirs / paths
    # ----------------------------
    run_id = f"run_{time.strftime('%Y-%m-%dT%H-%M-%S')}_{uuid.uuid4().hex[:8]}"
    out_root = os.path.join("Phase3_Results", run_id)
    backups_dir = os.path.join(out_root, "backups")
    os.makedirs(backups_dir, exist_ok=True)

    out_archive_path = os.path.join(out_root, "phase3_archive.json.gz")
    metrics_path = os.path.join(out_root, "Metrics_Phase3.csv")

    print(f"Phase3 input archive: {args.in_file}")
    print(f"Phase3 run dir:       {out_root}")
    print(f"Phase3 output archive:{out_archive_path}")

    # ----------------------------
    # RNG
    # ----------------------------
    rng = random.Random(args.seed)

    # ----------------------------
    # Load Phase2 archive and extract genomes (tolerant schema handling)
    # ----------------------------
    phase2_data = load_archive_file(args.in_file)
    elites = extract_elites(phase2_data)
    print(f"[Phase3] Phase2 archive elites: {len(elites)}")

    def _coerce_genome(g):
        """Return a raw genome list if possible, else None."""
        if g is None:
            return None
        if isinstance(g, (list, tuple)):
            return list(g)
        if isinstance(g, dict):
            for k in ("genome", "program", "code", "bytes", "data"):
                if k in g and isinstance(g[k], (list, tuple)):
                    return list(g[k])
        return None

    genomes = []
    for e in elites:
        g = _coerce_genome(e.get("genome", None))
        if g is not None:
            genomes.append(g)

    print(f"[Phase3] elites with genome field: {len(genomes)}")

    parent_pool = [g for g in genomes if is_valid_genome(g)]
    print(f"[Phase3] valid genomes accepted: {len(parent_pool)}")

    if len(parent_pool) == 0:
        raise RuntimeError("No valid genomes found in Phase2 archive (after filtering).")

    # ----------------------------
    # Cosmos (Variant 3)
    # ----------------------------
    cosmos = DualCosmosVariant3(
        physics_cycles=args.cycles,
        noise_rate=args.noise,
        invert_prob=args.invert_prob,
        drop_prob=args.drop_prob,
        rng=rng,
    )
    cosmos_bits_a, cosmos_bits_b, cosmos_targets = cosmos.generate()

    # ----------------------------
    # MAP-Elites archive (Phase3 axes)
    # ----------------------------
    archive = {}  # key: bin idx (tuple); value: elite dict

    # ----------------------------
    # Phase3 Bootstrap: evaluate random pairs from Phase2 genome pool
    # (This is the correct way to seed Phase3 bins, since Phase2 metrics are not Phase3 metrics.)
    # ----------------------------
    seeded = 0
    best_fit = 0.0

    bootstrap_target = max(0, int(args.bootstrap_pairs))
    if bootstrap_target > 0:
        print(f"[Phase3] Bootstrapping Phase3 archive with {bootstrap_target} random Phase2 pairs...")

        boot_done = 0
        boot_batch = 0

        # Reuse your existing batching size for memory/throughput control
        boot_batch_size = min(args.children_per_batch, bootstrap_target) if args.children_per_batch > 0 else 200000

        # Small helper (local) to chunk lists
        def _chunk_list(xs, n):
            for i in range(0, len(xs), n):
                yield xs[i:i + n]

        boot_t0 = time.time()

        while boot_done < bootstrap_target:
            boot_batch += 1
            remaining = bootstrap_target - boot_done
            batch_n = min(boot_batch_size, remaining)

            # Build Phase2 parent pairs (NO crossover, NO mutation)
            pairs = []
            for _ in range(batch_n):
                ga = rng.choice(parent_pool)
                gb = rng.choice(parent_pool)
                pairs.append((ga, gb))

            # Build evaluation tasks: (alice, bob, bits_a, bits_b, targets, cycles, grace, pair_reduce)
            tasks = [
                (a, b, cosmos_bits_a, cosmos_bits_b, cosmos_targets, args.cycles, args.grace, args.pair_reduce)
                for (a, b) in pairs
            ]

            # Evaluate in parallel using bundled task lists (fast + avoids keyword drift)
            results = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for subres in ex.map(_eval_task_list, list(_chunk_list(tasks, args.chunk_pairs)), chunksize=1):
                    results.extend(subres)

            # Apply results in-order
            inserted_now = 0
            crashes_now = 0

            for i, (fit, metrics, alice_crashed, bob_crashed) in enumerate(results):
                # Phase3 crash semantics: either crash => non-viable
                if alice_crashed or bob_crashed or not metrics:
                    crashes_now += 1
                    continue

                try:
                    idx = get_grid_index(metrics)
                except Exception:
                    continue

                elite = {
                    "genome": pairs[i][0],        # Alice
                    "bob_genome": pairs[i][1],    # Bob
                    "fitness": float(fit),
                    "metrics": metrics,
                    "origin": {
                        "seed_from": "phase2_bootstrap_pair",
                        "pair_reduce": args.pair_reduce,
                    },
                }

                if (idx not in archive) or (float(fit) > float(archive[idx]["fitness"])):
                    archive[idx] = elite
                    inserted_now += 1
                    if float(fit) > best_fit:
                        best_fit = float(fit)

            boot_done += batch_n
            seeded += inserted_now

            boot_elapsed = time.time() - boot_t0
            print(
                f"[Phase3] Bootstrap {boot_done}/{bootstrap_target} | elapsed={boot_elapsed:,.1f}s | bins={len(archive)} | inserted={inserted_now} | crashes={crashes_now} | best_fit={best_fit:.6f}")

    # After bootstrap, make the nursery parent pool come from the seeded Phase3 archive (single unified pool)
    if len(archive) > 0:
        new_pool = []
        for rec in archive.values():
            ga = rec.get("genome", None)
            gb = rec.get("bob_genome", None)
            if ga is not None and is_valid_genome(ga):
                new_pool.append(ga)
            if gb is not None and is_valid_genome(gb):
                new_pool.append(gb)
        if len(new_pool) > 0:
            parent_pool = new_pool
            print(f"[Phase3] Parent pool refreshed from Phase3 archive: {len(parent_pool)} genomes")


    # Metrics CSV header
    with open(metrics_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "step",
            "batch",
            "elapsed_s",
            "pop_bins",
            "inserted",
            "valid",
            "crashes",
            "crash_frac",
            "bob_crashes",
            "bob_crash_frac_valid",
            "rate_per_s",
            "best_fit",
        ])
        w.writerow([0, 0, 0.0, len(archive), seeded, 0, 0, 0.0, 0, 0.0, 0.0, best_fit])


    # ----------------------------
    # Helpers
    # ----------------------------
    def chunk_list(xs, n):
        for i in range(0, len(xs), n):
            yield xs[i:i + n]

    # ----------------------------
    # Evolution loop over PAIRS
    # ----------------------------
    t0 = time.time()
    evaluated = 0
    step = 0
    batch = 0

    while evaluated < args.n_children:
        batch += 1
        remaining = args.n_children - evaluated
        batch_n = min(args.children_per_batch, remaining)

        # Build child pair list (alice_child, bob_child)
        pairs = []
        for _ in range(batch_n):
            pa1 = rng.choice(parent_pool)
            pa2 = rng.choice(parent_pool)
            pb1 = rng.choice(parent_pool)
            pb2 = rng.choice(parent_pool)

            # Use the project’s known crossover (no crossover_multimodal symbol exists here)
            alice_child = crossover_homologous(pa1, pa2, rng)
            bob_child   = crossover_homologous(pb1, pb2, rng)

            # mutate_genome RETURNS a new genome (not in-place)
            alice_child = mutate_genome(alice_child, args.mut_rate, rng)
            bob_child   = mutate_genome(bob_child,   args.mut_rate, rng)

            pairs.append((alice_child, bob_child))

        # ----------------------------
        # Evaluate in parallel WITHOUT using _phase3_eval_pair_task
        # (It is currently inconsistent in-file and is what’s crashing you.)
        # Instead we use _eval_task_list / _eval_task which call evaluate_pair_variant3_with_reliance positionally.
        # ----------------------------
        tasks = [
            (a, b, cosmos_bits_a, cosmos_bits_b, cosmos_targets, args.cycles, args.grace, args.pair_reduce)
            for (a, b) in pairs
        ]

        valid = 0
        crashes = 0
        bob_crashes = 0
        inserted = 0

        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # bundle tasks to reduce overhead; each future returns a LIST of results
            for subres in ex.map(_eval_task_list, list(chunk_list(tasks, args.chunk_pairs)), chunksize=1):
                results.extend(subres)

        # Apply results in order (deterministic)
        for i, (fit, metrics, alice_crashed, bob_crashed) in enumerate(results):
            step += 1
            evaluated += 1

            # Phase3 spec: if either Alice OR Bob crashes, the pair is non-viable.
            if alice_crashed or bob_crashed or not metrics:
                crashes += 1
                if bob_crashed:
                    bob_crashes += 1
                continue

            valid += 1


            try:
                idx = get_grid_index(metrics)
            except Exception:
                continue

            elite = {
                # Phase3 nursery stores the co-adapted pair (single unified pool concept)
                "genome": pairs[i][0],        # keep for backward compatibility / extract_elites()
                "bob_genome": pairs[i][1],    # new: store partner explicitly
                "fitness": float(fit),
                "metrics": metrics,
                "origin": {
                    "pair_reduce": args.pair_reduce,
                },
            }


            if (idx not in archive) or (float(fit) > float(archive[idx]["fitness"])):
                archive[idx] = elite
                inserted += 1
                if float(fit) > best_fit:
                    best_fit = float(fit)

        # Logging
        elapsed = time.time() - t0
        rate = evaluated / elapsed if elapsed > 0 else 0.0
        crash_frac = (crashes / (valid + crashes)) if (valid + crashes) > 0 else 0.0
        bob_crash_frac_valid = (bob_crashes / valid) if valid > 0 else 0.0

        with open(metrics_path, "a", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([
                step,
                batch,
                elapsed,
                len(archive),
                inserted,
                valid,
                crashes,
                crash_frac,
                bob_crashes,
                bob_crash_frac_valid,
                rate,
                best_fit,
            ])

        print(
            f"{evaluated}/{args.n_children} | elapsed={elapsed:,.1f}s | bins={len(archive)} | inserted={inserted} "
            f"| crash_frac={crash_frac:.3f} | bob_crash_frac(valid)={bob_crash_frac_valid:.3f} "
            f"| best_fit={best_fit:.6f} | rate={rate:.1f}/s"
        )

        # --- Nursery genetics: evolve a single unified parent pool from the current Phase3 archive ---
        # Include BOTH genomes from each elite pair (when available).
        new_pool = []
        for rec in archive.values():
            ga = rec.get("genome", None)
            gb = rec.get("bob_genome", None)
            if ga is not None and is_valid_genome(ga):
                new_pool.append(ga)
            if gb is not None and is_valid_genome(gb):
                new_pool.append(gb)

        # Keep a safety fallback: if the archive is still tiny or empty, retain the old pool.
        if len(new_pool) > 0:
            parent_pool = new_pool


        # Backups (stringify tuple keys so json.dump works, consistent with Phase2)
        if args.backup_interval_batches > 0 and (batch % args.backup_interval_batches == 0):
            backup_path = os.path.join(backups_dir, f"backup_phase3_{evaluated}.json.gz")
            meta = {
                "phase": 3,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "run_id": run_id,
                "in_file": args.in_file,
                "evaluated": evaluated,
                "bins": len(archive),
                "best_fit": best_fit,
                "params": vars(args),
                "opcode_hash": OPCODE_HASH,
            }
            payload = {"meta": meta, "archive": {str(k): v for k, v in archive.items()}}
            with gzip.open(backup_path, "wt", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"Wrote backup: {backup_path}")

    # Final write (stringify tuple keys)
    meta = {
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "in_file": args.in_file,
        "evaluated": evaluated,
        "bins": len(archive),
        "best_fit": best_fit,
        "params": vars(args),
        "opcode_hash": OPCODE_HASH,
    }
    payload = {"meta": meta, "archive": {str(k): v for k, v in archive.items()}}
    with gzip.open(out_archive_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    print("\nDone.")
    print(f"  evaluated: {evaluated}")
    print(f"  bins in archive: {len(archive)}")
    print(f"  best_fit: {best_fit:.6f}")
    print(f"Wrote Phase3 archive: {out_archive_path}")







if __name__ == "__main__":
    main()
