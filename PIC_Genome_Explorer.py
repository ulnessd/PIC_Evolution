
#!/usr/bin/env python3
"""
PIC_Genome_Explorer.py

Genome Explorer for evolved PIC10F200 critters across Phase1–Phase4 backups.
Designed for detailed inspection and computationally heavier per-genome assays:
  - Decode + pretty-print genomes
  - Neighbor inspection (in archive bin space)
  - Subroutine detection + reachability validation (bounded CFG)
  - Linchpin analysis via NOP-substitution ablation (crash + trace-delta)

This script intentionally avoids producing the full population plots; that remains
the job of PIC_Archive_Stats.py.

Usage examples:

  # List top elites by fitness (Phase3 unified archive)
  python3 PIC_Genome_Explorer.py list --in backup_phase3_11000000.json.gz --top 20

  # Show one elite by bin tuple
  python3 PIC_Genome_Explorer.py show --in I3_backup_phase4_000005000.json.gz --arch alice --bin 0,1,9,4

  # Subroutine analysis on one genome
  python3 PIC_Genome_Explorer.py subroutines --in ... --arch bob --bin 0,1,9,4

  # Linchpin (NOP ablation) quick assay (first 64 instruction positions)
  python3 PIC_Genome_Explorer.py linchpin --in ... --arch alice --bin 0,1,9,4 --steps 4000 --limit 64

Notes:
- Requires Pic10Sim.py from your repo. Place this script in the same directory
  as Pic10Sim.py or provide --repo_path to point at the repo folder.
- Linchpin analysis uses Pic10Sim.emulate_cycle and a simple input protocol
  (default: both inputs low). You can later swap in your cosmic-signal protocol.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------
# ISA / opcode mapping
# ---------------------------
# This list MUST match the order used by your drivers (genome op_id encoding).
OPCODE_LIST: List[str] = [
    'ADDWF_W', 'ADDWF_F', 'ANDWF_W', 'ANDWF_F', 'CLRF', 'CLRW', 'COMF_W', 'COMF_F',
    'DECF_W', 'DECF_F', 'DECFSZ_W', 'DECFSZ_F', 'INCF_W', 'INCF_F', 'INCFSZ_W', 'INCFSZ_F',
    'IORWF_W', 'IORWF_F', 'MOVF_W', 'MOVF_F', 'MOVWF', 'MOVLW', 'NOP', 'RLF_W', 'RLF_F',
    'RRF_W', 'RRF_F', 'SUBWF_W', 'SUBWF_F', 'SWAPF_W', 'SWAPF_F', 'XORWF_W', 'XORWF_F',
    'BCF', 'BSF', 'BTFSC', 'BTFSS', 'CALL', 'GOTO', 'RETLW', 'DELAY_MACRO'
]
OP_ID_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(OPCODE_LIST)}

CONTROL_FLOW = {
    "GOTO", "CALL", "RETLW",
    "BTFSC", "BTFSS",
    "INCFSZ_W", "INCFSZ_F", "DECFSZ_W", "DECFSZ_F",
}

SKIP_OPS = {"BTFSC", "BTFSS", "INCFSZ_W", "INCFSZ_F", "DECFSZ_W", "DECFSZ_F"}
JUMP_OPS = {"GOTO", "CALL", "RETLW"}

# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class EliteRef:
    arch: str  # 'unified' | 'alice' | 'bob'
    bin_tuple: Tuple[int, ...]
    fitness: float

# ---------------------------
# Backup loading
# ---------------------------

def load_backup(path: Path) -> Dict[str, Any]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_archives(blob: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict:
      arch_name -> archive_dict (bin_key->record)
    Supports:
      - Phase1/2/3 unified: blob["archive"]
      - Phase4: blob["archive"]["alice"], blob["archive"]["bob"]
    """
    arch = blob.get("archive", {})
    if isinstance(arch, dict) and "alice" in arch and "bob" in arch:
        return {"alice": arch["alice"], "bob": arch["bob"]}
    return {"unified": arch}

def parse_bin_key(k: str) -> Tuple[int, ...]:
    # supports "(0, 1, 9, 4)" or "0,1,9,4" or "[0,1,9,4]"
    s = k.strip()
    s = s.strip("()[]")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return tuple(int(p) for p in parts)

def bin_to_key(t: Tuple[int, ...]) -> str:
    return "(" + ", ".join(str(int(x)) for x in t) + ")"

def genome_from_record(rec: Dict[str, Any]) -> List[List[int]]:
    g = rec.get("genome")
    if g is None:
        raise KeyError("Record has no 'genome'.")
    return g

def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def fitness_from_record(rec: Dict[str, Any]) -> float:
    # Your backups typically store 'fit' (Phase3/4) or sometimes 'fitness'
    if "fit" in rec:
        return safe_float(rec["fit"])
    if "fitness" in rec:
        return safe_float(rec["fitness"])
    # fallback: try nested
    if "metrics" in rec and isinstance(rec["metrics"], dict) and "fitness" in rec["metrics"]:
        return safe_float(rec["metrics"]["fitness"])
    return float("nan")

# ---------------------------
# Pretty-print / decoding
# ---------------------------

def decode_genome(genome: List[List[int]]) -> List[Tuple[int, int, str]]:
    """
    Returns list of (pc, operand, mnemonic) with mnemonic looked up by op_id.
    """
    out = []
    for pc, pair in enumerate(genome):
        op_id = int(pair[0])
        operand = int(pair[1]) if len(pair) > 1 else 0
        name = OP_ID_TO_NAME.get(op_id, f"OP{op_id}")
        out.append((pc, operand, name))
    return out

def print_genome(genome: List[List[int]], limit: Optional[int] = None) -> None:
    dec = decode_genome(genome)
    if limit is not None:
        dec = dec[:limit]
    for pc, operand, name in dec:
        print(f"{pc:03d}: {name:<10s} {operand}")

def opcode_counts(genome: List[List[int]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, _, name in decode_genome(genome):
        counts[name] = counts.get(name, 0) + 1
    return counts

# ---------------------------
# Neighbor inspection (bin-space)
# ---------------------------

def neighbors_4d(t: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    # ±1 in each dimension (Manhattan neighbors)
    out = []
    dims = len(t)
    for d in range(dims):
        for delta in (-1, +1):
            u = list(t)
            u[d] = u[d] + delta
            out.append(tuple(u))
    return out

def find_record(archive: Dict[str, Any], bin_tuple: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    # keys are strings like "(0, 1, 9, 4)"
    key = bin_to_key(bin_tuple)
    if key in archive:
        return archive[key]
    # sometimes keys may be stored without spaces; try loose match
    for k in archive.keys():
        try:
            if parse_bin_key(k) == bin_tuple:
                return archive[k]
        except Exception:
            continue
    return None

# ---------------------------
# Subroutine analysis
# ---------------------------

def call_targets(genome: List[List[int]]) -> List[Tuple[int, int]]:
    """Return list of (call_pc, target_pc) from CALL instructions."""
    dec = decode_genome(genome)
    out = []
    for pc, operand, name in dec:
        if name == "CALL":
            out.append((pc, operand & 0xFF))
    return out

def bounded_cfg_reaches_retlw(genome: List[List[int]], start_pc: int, step_budget: int = 600) -> bool:
    """
    Conservative reachability: explores control-flow graph ignoring register state.
    Treat skip ops as two possible paths (skip / no-skip).
    Treat CALL as jump into target and ALSO continue fallthrough (conservative).
    """
    prog = decode_genome(genome)
    n = len(prog)
    seen = set()
    frontier = [(start_pc, 0)]
    while frontier:
        pc, steps = frontier.pop()
        if steps > step_budget:
            continue
        if pc < 0 or pc >= n:
            continue
        state = (pc, steps)
        if state in seen:
            continue
        seen.add(state)

        _, operand, op = prog[pc]
        if op == "RETLW":
            return True

        next_pc = pc + 1

        if op == "GOTO":
            frontier.append((operand & 0xFF, steps + 1))
        elif op == "CALL":
            # conservative: explore both the callee and fallthrough
            frontier.append((operand & 0xFF, steps + 1))
            frontier.append((next_pc, steps + 1))
        elif op in SKIP_OPS:
            # both possibilities: skip or not
            frontier.append((next_pc, steps + 1))
            frontier.append((pc + 2, steps + 1))
        else:
            frontier.append((next_pc, steps + 1))

    return False

def _trace_returns_to_caller(
    genome: List[List[int]],
    start_pc: int,
    caller_return_pc: int,
    max_steps: int = 2048,
    max_stack: int = 8,
) -> bool:
    """
    Lightweight control-flow trace starting at `start_pc` with an initial stack containing
    `caller_return_pc` (the PC immediately after the CALL).

    Returns True iff we encounter a RETLW that returns to caller_return_pc with an empty stack,
    i.e. a true return-to-caller.
    """
    prog = decode_genome(genome)
    n = len(prog)

    pc = int(start_pc)
    stack = [int(caller_return_pc)]
    seen = set()

    for _ in range(max_steps):
        state = (pc, tuple(stack))
        if state in seen:
            return False
        seen.add(state)

        if pc < 0 or pc >= n:
            return False

        _, operand, op = prog[pc]

        if op == "GOTO":
            if operand is None:
                return False
            pc = int(operand)
            continue

        if op == "CALL":
            if operand is None:
                return False
            if len(stack) >= max_stack:
                return False
            stack.append(pc + 1)
            pc = int(operand)
            continue

        if op == "RETLW":
            if not stack:
                return False
            ret = stack.pop()
            if ret == caller_return_pc and len(stack) == 0:
                return True
            pc = int(ret)
            continue

        # default fallthrough
        pc += 1

    return False


def _trace_returns_to_caller(genome, start_pc: int, caller_return_pc: int,
                             max_steps: int = 2048, max_stack: int = 8) -> bool:
    """
    Lightweight control-flow trace starting at `start_pc` with an initial stack containing
    `caller_return_pc` (the PC immediately after the CALL).

    Returns True iff we encounter a RETLW that pops back to caller_return_pc AND empties the stack
    (meaning we've truly returned to the CALL site).
    """
    n = len(genome)
    pc = int(start_pc)
    stack = [int(caller_return_pc)]
    seen = set()

    def _target_from_arg(arg):
        # Keep this conservative: only accept valid in-genome targets.
        try:
            t = int(arg)
        except Exception:
            return None
        return t if 0 <= t < n else None

    for _ in range(max_steps):
        state = (pc, tuple(stack))
        if state in seen:
            return False
        seen.add(state)

        if not (0 <= pc < n):
            return False

        op, arg = genome[pc]

        if op == "GOTO":
            t = _target_from_arg(arg)
            if t is None:
                return False
            pc = t
            continue

        if op == "CALL":
            t = _target_from_arg(arg)
            if t is None:
                return False
            if len(stack) >= max_stack:
                return False
            stack.append(pc + 1)
            pc = t
            continue

        if op == "RETLW":
            if not stack:
                return False
            ret = stack.pop()
            if ret == caller_return_pc and len(stack) == 0:
                return True
            pc = ret
            continue

        # Default: treat as fall-through
        pc += 1

    return False


def analyze_subroutines(
    genome: List[List[int]],
    forward_only: bool = False,
    max_jump: int = 220,
    max_steps: int = 2048,
    require_return_to_caller: bool = False,
) -> Dict[str, Any]:
    calls = call_targets(genome)
    results = []
    for call_pc, target in calls:
        ok_forward = (target > call_pc) if forward_only else True
        ok_jump = abs(target - call_pc) <= max_jump
        reaches_ret = bounded_cfg_reaches_retlw(genome, target)

        returns_to_caller = False
        if reaches_ret:
            returns_to_caller = _trace_returns_to_caller(
                genome=genome,
                start_pc=target,
                caller_return_pc=call_pc + 1,
                max_steps=max_steps,
            )

        if require_return_to_caller:
            success = bool(ok_jump and ok_forward and reaches_ret and returns_to_caller)
        else:
            success = bool(ok_jump and ok_forward and reaches_ret)

        results.append({
            "call_pc": call_pc,
            "target_pc": target,
            "reasonable_jump": bool(ok_jump),
            "forward": bool(ok_forward),
            "reaches_retlw": bool(reaches_ret),
            "returns_to_caller": bool(returns_to_caller),
            "successful_call": bool(success),
        })
    return {
        "n_calls": len(calls),
        "calls": results,
        "n_successful_calls": sum(1 for r in results if r["successful_call"]),
    }

# ---------------------------
# Linchpin analysis (NOP ablation)
# ---------------------------

def _import_pic10sim(repo_path: Path):
    # Ensure repo_path is on sys.path so Pic10Sim can be imported
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        from Pic10Sim import Pic10Sim  # type: ignore
        return Pic10Sim
    except Exception as e:
        raise RuntimeError(f"Could not import Pic10Sim from {repo_path}. Error: {e}")

def run_trace(genome: List[List[int]], steps: int, repo_path: Path,
              gp2: Optional[np.ndarray] = None, gp3: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, str]:
    """
    Execute genome for 'steps' cycles using Pic10Sim.emulate_cycle.

    Returns:
      outputs: (steps, 2) array for (gp0, gp1) outputs per cycle
      crashed: bool
      crash_reason: str
    """
    Pic10Sim = _import_pic10sim(repo_path)
    sim = Pic10Sim()
    sim.load(program=[(int(op), int(arg)) for op, arg in genome], opcode_list=OPCODE_LIST)

    if gp2 is None:
        gp2 = np.zeros(steps, dtype=np.int8)
    if gp3 is None:
        gp3 = np.zeros(steps, dtype=np.int8)

    outs = np.zeros((steps, 2), dtype=np.int8)

    for i in range(steps):
        out_bits, crashed = sim.emulate_cycle(int(gp2[i]), int(gp3[i]))
        # Pic10Sim returns out_bits with GP0 bit0 and GP1 bit1 (per your contract)
        gp0 = 1 if (out_bits & 0x01) else 0
        gp1 = 1 if (out_bits & 0x02) else 0
        outs[i, 0] = gp0
        outs[i, 1] = gp1
        if crashed:
            return outs[: i + 1], True, str(sim.crash_reason)

    return outs, False, ""

def linchpin_nop_ablation(genome: List[List[int]],
                          repo_path: Path,
                          steps: int = 4000,
                          limit: Optional[int] = None,
                          seed: int = 0) -> Dict[str, Any]:
    """
    Replace instruction i with NOP (preserving addresses) and re-run.
    Reports crash rate and a simple behavioral delta (Hamming distance of output trace)
    vs baseline over the overlapping prefix.

    This is intended as a diagnostic assay; for large genomes you can use --limit.
    """
    rng = random.Random(seed)

    # baseline trace
    base_out, base_crashed, base_reason = run_trace(genome, steps=steps, repo_path=repo_path)
    L = len(genome)
    L_test = min(L, limit) if limit is not None else L

    results = []
    n_crash = 0

    for i in range(L_test):
        g2 = [row[:] for row in genome]
        # Replace with NOP operand 0. Find opcode id for "NOP".
        nop_id = OPCODE_LIST.index("NOP")
        g2[i] = [nop_id, 0]

        out, crashed, reason = run_trace(g2, steps=steps, repo_path=repo_path)

        # Compare traces over common prefix
        m = min(len(out), len(base_out))
        if m > 0:
            delta = float(np.mean(out[:m] != base_out[:m]))
        else:
            delta = float("nan")

        n_crash += 1 if crashed else 0
        results.append({
            "pos": i,
            "orig_op": OP_ID_TO_NAME.get(int(genome[i][0]), f"OP{int(genome[i][0])}"),
            "crashed": bool(crashed),
            "crash_reason": reason,
            "delta_outputs": delta,
        })

    # opcode-level linchpin counts
    opcode_linchpin: Dict[str, int] = {}
    for r in results:
        if r["crashed"]:
            opcode_linchpin[r["orig_op"]] = opcode_linchpin.get(r["orig_op"], 0) + 1

    return {
        "baseline_crashed": bool(base_crashed),
        "baseline_crash_reason": base_reason,
        "tested_positions": L_test,
        "crash_fraction": float(n_crash) / float(L_test) if L_test else float("nan"),
        "opcode_linchpin_counts": dict(sorted(opcode_linchpin.items(), key=lambda kv: kv[1], reverse=True)),
        "position_results": results,
    }

# ---------------------------
# CLI
# ---------------------------

def cmd_list(args):
    blob = load_backup(Path(args.in_file))
    arches = detect_archives(blob)

    arch_name = args.arch
    if arch_name not in arches:
        raise SystemExit(f"--arch {arch_name} not found. Available: {list(arches.keys())}")

    archive = arches[arch_name]
    rows = []
    for k, rec in archive.items():
        try:
            bt = parse_bin_key(k)
        except Exception:
            continue
        rows.append((fitness_from_record(rec), bt, k))
    rows.sort(key=lambda t: (-(t[0] if not math.isnan(t[0]) else -1e9)))

    top = rows[: args.top]
    print(f"Archive: {arch_name} | elites={len(rows)} | showing top {len(top)} by fitness\n")
    for fit, bt, k in top:
        print(f"fit={fit: .6f}  bin={','.join(map(str, bt))}")

def cmd_show(args):
    blob = load_backup(Path(args.in_file))
    arches = detect_archives(blob)
    arch_name = args.arch
    if arch_name not in arches:
        raise SystemExit(f"--arch {arch_name} not found. Available: {list(arches.keys())}")
    archive = arches[arch_name]

    bt = tuple(int(x) for x in args.bin.split(",") if x.strip() != "")
    rec = find_record(archive, bt)
    if rec is None:
        raise SystemExit(f"Bin {bt} not found in archive.")
    fit = fitness_from_record(rec)
    print(f"Archive: {arch_name}")
    print(f"Bin: {bt}")
    print(f"Fitness: {fit}")
    if "metrics" in rec and isinstance(rec["metrics"], dict):
        print(f"Metrics keys: {sorted(rec['metrics'].keys())[:20]}{' ...' if len(rec['metrics'])>20 else ''}")
    print("\nGenome:")
    genome = genome_from_record(rec)
    print_genome(genome, limit=args.limit)

    counts = opcode_counts(genome)
    top_ops = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    print("\nTop opcode counts:")
    for name, c in top_ops:
        print(f"  {name:<12s} {c}")

def cmd_neighbors(args):
    blob = load_backup(Path(args.in_file))
    arches = detect_archives(blob)
    arch_name = args.arch
    if arch_name not in arches:
        raise SystemExit(f"--arch {arch_name} not found. Available: {list(arches.keys())}")
    archive = arches[arch_name]
    bt = tuple(int(x) for x in args.bin.split(",") if x.strip() != "")
    rec = find_record(archive, bt)
    if rec is None:
        raise SystemExit(f"Bin {bt} not found in archive.")

    neigh = neighbors_4d(bt)
    print(f"Neighbors of {bt} (arch={arch_name}):")
    for nb in neigh:
        r2 = find_record(archive, nb)
        if r2 is None:
            continue
        print(f"  {nb}  fit={fitness_from_record(r2): .6f}")

def cmd_subroutines(args):
    blob = load_backup(Path(args.in_file))
    arches = detect_archives(blob)
    arch_name = args.arch
    if arch_name not in arches:
        raise SystemExit(f"--arch {arch_name} not found. Available: {list(arches.keys())}")
    archive = arches[arch_name]

    bt = tuple(int(x) for x in args.bin.split(",") if x.strip() != "")
    rec = find_record(archive, bt)
    if rec is None:
        raise SystemExit(f"Bin {bt} not found in archive.")
    genome = genome_from_record(rec)

    report = analyze_subroutines(genome, forward_only=args.forward_only, max_jump=args.max_jump)
    print(f"Subroutine report for bin={bt} arch={arch_name}")
    print(f"n_calls={report['n_calls']}  n_successful_calls={report['n_successful_calls']}")
    for r in report["calls"]:
        print(f"  CALL@{r['call_pc']:03d} -> {r['target_pc']:03d}  "
              f"reasonable={r['reasonable_jump']} forward={r['forward']} "
              f"reaches_retlw={r['reaches_retlw']} returns_to_caller={r.get('returns_to_caller', False)}  "
              f"SUCCESS={r['successful_call']}")


def cmd_linchpin(args):
    blob = load_backup(Path(args.in_file))
    arches = detect_archives(blob)
    arch_name = args.arch
    if arch_name not in arches:
        raise SystemExit(f"--arch {arch_name} not found. Available: {list(arches.keys())}")
    archive = arches[arch_name]

    bt = tuple(int(x) for x in args.bin.split(",") if x.strip() != "")
    rec = find_record(archive, bt)
    if rec is None:
        raise SystemExit(f"Bin {bt} not found in archive.")
    genome = genome_from_record(rec)

    repo_path = Path(args.repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise SystemExit(f"--repo_path does not exist: {repo_path}")

    t0 = time.time()
    report = linchpin_nop_ablation(
        genome,
        repo_path=repo_path,
        steps=args.steps,
        limit=args.limit,
        seed=args.seed
    )
    dt = time.time() - t0

    print(f"Linchpin NOP ablation | arch={arch_name} bin={bt}")
    print(f"baseline_crashed={report['baseline_crashed']} reason={report['baseline_crash_reason']}")
    print(f"tested_positions={report['tested_positions']} crash_fraction={report['crash_fraction']:.3f} time={dt:.2f}s")
    print("\nTop linchpin opcode counts (crash-causing when NOP'd):")
    for op, c in list(report["opcode_linchpin_counts"].items())[:15]:
        print(f"  {op:<12s} {c}")

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote: {outp}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Genome Explorer for PIC evolution archives.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List top elites by fitness")
    p_list.add_argument("--in", dest="in_file", required=True)
    p_list.add_argument("--arch", default="unified", choices=["unified", "alice", "bob"])
    p_list.add_argument("--top", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show genome + summary for a single bin")
    p_show.add_argument("--in", dest="in_file", required=True)
    p_show.add_argument("--arch", default="unified", choices=["unified", "alice", "bob"])
    p_show.add_argument("--bin", required=True, help="Comma-separated bin tuple, e.g. 0,1,9,4")
    p_show.add_argument("--limit", type=int, default=None, help="Limit printed instructions")
    p_show.set_defaults(func=cmd_show)

    p_nb = sub.add_parser("neighbors", help="Show existing neighbor bins (4D Manhattan)")
    p_nb.add_argument("--in", dest="in_file", required=True)
    p_nb.add_argument("--arch", default="unified", choices=["unified", "alice", "bob"])
    p_nb.add_argument("--bin", required=True)
    p_nb.set_defaults(func=cmd_neighbors)

    p_subr = sub.add_parser("subroutines", help="Detect CALL targets and validate reachable RETLW")
    p_subr.add_argument("--in", dest="in_file", required=True)
    p_subr.add_argument("--arch", default="unified", choices=["unified", "alice", "bob"])
    p_subr.add_argument("--bin", required=True)
    p_subr.add_argument("--forward_only", action="store_true", help="Require CALL target > call PC")
    p_subr.add_argument("--max_jump", type=int, default=220, help="Max |target - call_pc| to be considered reasonable")
    p_subr.set_defaults(func=cmd_subroutines)

    p_lin = sub.add_parser("linchpin", help="NOP ablation test (crash + output delta)")
    p_lin.add_argument("--in", dest="in_file", required=True)
    p_lin.add_argument("--arch", default="unified", choices=["unified", "alice", "bob"])
    p_lin.add_argument("--bin", required=True)
    p_lin.add_argument("--repo_path", required=True, help="Path to folder containing Pic10Sim.py")
    p_lin.add_argument("--steps", type=int, default=4000)
    p_lin.add_argument("--limit", type=int, default=None, help="Only test first N instruction positions")
    p_lin.add_argument("--seed", type=int, default=0)
    p_lin.add_argument("--out_json", default=None, help="Write full report JSON to path")
    p_lin.set_defaults(func=cmd_linchpin)

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
