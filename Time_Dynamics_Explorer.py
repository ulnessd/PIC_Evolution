#!/usr/bin/env python3
"""
time_dynamics_explorer.py

Temporal performance & dynamics explorer for PIC evolution archives.

Must-haves + nice-to-haves implemented:
- Iterate over a sequence of backup files (Phase1–Phase4).
- For each backup, compute per-archive time-series:
    * elites count
    * max fitness, mean fitness
    * 4D center of mass (COM) of occupied bins
    * 4D fitness-weighted COM
    * per-axis occupancy marginals (counts over bin index)
- Produce plots:
    * elites vs progress
    * max/mean fitness vs progress
    * COM vs progress (unweighted + weighted)
    * per-axis occupancy heatmaps: time-index x bin-index (counts)
- Oscilloscope plot:
    * GP2/GP3 inputs + GP1/GP0 outputs vs cycle for a selected critter
    * default inputs are zeros; can provide CSV inputs

This script treats PIC_Genome_Explorer.py as an immutable function source ("engine").
Place this script in the same folder as PIC_Genome_Explorer.py, or pass --engine_path.

USAGE (time series):
  python3 time_dynamics_explorer.py series \
    --glob "Phase3_Results/run1/backups/backup_phase3_*.json.gz" \
    --out Time_Phase3

USAGE (oscilloscope):
  python3 time_dynamics_explorer.py scope \
    --in Phase3Data/backup_phase3_11000000.json.gz \
    --arch unified --bin 1,15,9,0 \
    --repo_path /home/darin/PycharmProjects/PICEvolutionMSI/Phase1_Driver \
    --steps 4000 \
    --out Scope_bin_1_15_9_0.png
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Engine import ----------
def import_engine(engine_path: Optional[str]):
    if engine_path:
        ep = Path(engine_path).expanduser().resolve()
        if ep.is_dir():
            if str(ep) not in sys.path:
                sys.path.insert(0, str(ep))
        else:
            if str(ep.parent) not in sys.path:
                sys.path.insert(0, str(ep.parent))
    try:
        import PIC_Genome_Explorer as eng  # type: ignore
        return eng
    except Exception as e:
        raise RuntimeError(
            "Could not import PIC_Genome_Explorer.py as a module.\n"
            "Place time_dynamics_explorer.py in the same folder as PIC_Genome_Explorer.py,\n"
            "or pass --engine_path /path/to/folder.\n"
            f"Import error: {e}"
        )


# ---------- Pic10Sim import for scope ----------
def import_pic10sim(repo_path: Path):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        from Pic10Sim import Pic10Sim  # type: ignore
        return Pic10Sim
    except Exception as e:
        raise RuntimeError(f"Could not import Pic10Sim from {repo_path}. Error: {e}")


# ---------- Helpers ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def glob_paths(pattern: str) -> List[Path]:
    import glob
    pat = str(Path(pattern).expanduser())
    return [Path(x) for x in glob.glob(pat)]

def extract_number_for_sort(path: Path) -> int:
    s = path.name
    nums = re.findall(r"\d+", s)
    if not nums:
        return 0
    return int(nums[-1])

def best_progress(blob: Dict[str, Any], path: Path) -> int:
    for k in ("evaluated", "n_evaluated", "n_eval", "eval_count"):
        if k in blob:
            try:
                return int(blob[k])
            except Exception:
                pass
    return extract_number_for_sort(path)

def compute_com(bin_tuples: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if bin_tuples.size == 0:
        return np.full(4, np.nan, dtype=float)
    x = bin_tuples.astype(float)
    if weights is None:
        return np.mean(x, axis=0)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    s = float(np.sum(w))
    if s <= 0:
        return np.full(4, np.nan, dtype=float)
    return (x * w[:, None]).sum(axis=0) / s

def compute_axis_sizes(bin_tuples: np.ndarray, default_dim: int = 16) -> Tuple[int,int,int,int]:
    if bin_tuples.size == 0:
        return (default_dim, default_dim, default_dim, default_dim)
    maxs = bin_tuples.max(axis=0)
    return tuple(int(max(default_dim, int(m) + 1)) for m in maxs)  # type: ignore

def per_axis_marginals(bin_tuples: np.ndarray, axis_sizes: Tuple[int,int,int,int]) -> List[np.ndarray]:
    out = []
    for ax in range(4):
        counts = np.zeros(axis_sizes[ax], dtype=int)
        if bin_tuples.size:
            vals = bin_tuples[:, ax]
            vals = vals[(vals >= 0) & (vals < axis_sizes[ax])]
            if len(vals) > 0:
                cc = np.bincount(vals, minlength=axis_sizes[ax])
                counts[:len(cc)] = cc[:axis_sizes[ax]]
        out.append(counts)
    return out

def save_line_plot(x, ys: Dict[str, np.ndarray], out_path: Path, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, y in ys.items():
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_heatmap_time_axis(mat: np.ndarray, out_path: Path, xlabel: str, ylabel: str, title: str, cbar_label: str) -> None:
    plt.figure(figsize=(9, 5))
    im = plt.imshow(mat, origin="lower", aspect="auto")
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------- Series mode ----------
def run_series(args):
    eng = import_engine(args.engine_path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    files: List[Path] = []
    if args.glob:
        files.extend(glob_paths(args.glob))
    if args.inputs:
        files.extend([Path(x) for x in args.inputs])
    if not files:
        raise SystemExit("No inputs found. Use --glob or --inputs.")

    files = sorted(list({str(p): p for p in files}.values()), key=extract_number_for_sort)

    records = []
    axis_sizes = (args.default_dim, args.default_dim, args.default_dim, args.default_dim)
    occ_time = [[] for _ in range(4)]

    t0 = time.time()

    for fp in files:
        blob = eng.load_backup(fp)
        arches = eng.detect_archives(blob)
        if args.arch not in arches:
            continue

        archive: Dict[str, Any] = arches[args.arch]

        bins = []
        fits = []
        for k, rec in archive.items():
            try:
                bt = eng.parse_bin_key(k)
            except Exception:
                continue
            if len(bt) != 4:
                continue
            bins.append(bt)
            fits.append(eng.fitness_from_record(rec))

        bt_arr = np.array(bins, dtype=int) if bins else np.zeros((0, 4), dtype=int)
        fit_arr = np.array(fits, dtype=float) if fits else np.zeros((0,), dtype=float)

        axis_sizes = tuple(
            max(axis_sizes[i], int(bt_arr[:, i].max()) + 1) if bt_arr.size else axis_sizes[i]
            for i in range(4)
        )  # type: ignore

        elites = int(len(bins))
        max_fit = float(np.nanmax(fit_arr)) if elites else float("nan")
        mean_fit = float(np.nanmean(fit_arr)) if elites else float("nan")
        com = compute_com(bt_arr, None)
        wcom = compute_com(bt_arr, fit_arr)
        progress = best_progress(blob, fp)

        records.append({
            "file": fp.name,
            "path": str(fp),
            "progress": progress,
            "elites": elites,
            "max_fitness": max_fit,
            "mean_fitness": mean_fit,
            "com0": com[0], "com1": com[1], "com2": com[2], "com3": com[3],
            "wcom0": wcom[0], "wcom1": wcom[1], "wcom2": wcom[2], "wcom3": wcom[3],
        })

        marg = per_axis_marginals(bt_arr, axis_sizes=axis_sizes)
        for ax in range(4):
            occ_time[ax].append(marg[ax])

    if not records:
        raise SystemExit(f"No backups contained the requested archive '{args.arch}'.")

    df = pd.DataFrame(records).sort_values("progress").reset_index(drop=True)
    df.to_csv(out_dir / f"time_series_{args.arch}.csv", index=False)

    # Occupancy matrices (T x K) for each axis
    for ax in range(4):
        K = axis_sizes[ax]
        T = len(occ_time[ax])
        mat = np.zeros((T, K), dtype=int)
        for ti in range(T):
            v = occ_time[ax][ti]
            mat[ti, :min(len(v), K)] = v[:K]
        np.save(out_dir / f"axis_{ax}_occupancy_time.npy", mat)

    x = df["progress"].to_numpy()

    save_line_plot(
        x,
        {"elites": df["elites"].to_numpy()},
        out_dir / f"elites_vs_progress_{args.arch}.png",
        xlabel="progress (evaluated or filename counter)",
        ylabel="elites (occupied bins)",
        title=f"Elites vs progress | arch={args.arch}"
    )

    save_line_plot(
        x,
        {"max_fitness": df["max_fitness"].to_numpy(), "mean_fitness": df["mean_fitness"].to_numpy()},
        out_dir / f"fitness_vs_progress_{args.arch}.png",
        xlabel="progress",
        ylabel="fitness",
        title=f"Fitness vs progress | arch={args.arch}"
    )

    save_line_plot(
        x,
        {"com0": df["com0"].to_numpy(), "com1": df["com1"].to_numpy(), "com2": df["com2"].to_numpy(), "com3": df["com3"].to_numpy()},
        out_dir / f"com_vs_progress_{args.arch}.png",
        xlabel="progress",
        ylabel="COM (bin index)",
        title=f"4D center of mass vs progress | arch={args.arch}"
    )

    save_line_plot(
        x,
        {"wcom0": df["wcom0"].to_numpy(), "wcom1": df["wcom1"].to_numpy(), "wcom2": df["wcom2"].to_numpy(), "wcom3": df["wcom3"].to_numpy()},
        out_dir / f"wcom_vs_progress_{args.arch}.png",
        xlabel="progress",
        ylabel="fitness-weighted COM (bin index)",
        title=f"4D fitness-weighted COM vs progress | arch={args.arch}"
    )

    for ax in range(4):
        mat = np.load(out_dir / f"axis_{ax}_occupancy_time.npy")
        save_heatmap_time_axis(
            mat,
            out_dir / f"axis_{ax}_occupancy_heat_{args.arch}.png",
            xlabel=f"bin index along axis_{ax}",
            ylabel="backup index (sorted)",
            title=f"Axis {ax} occupancy over time | arch={args.arch}",
            cbar_label="occupied bins count"
        )

    dt = time.time() - t0
    print(f"Done. Outputs in: {out_dir}")
    print(f"Backups processed (matching arch): {len(df)}")
    print(f"Total runtime: {dt:.2f} s")
    print(f"CSV: {out_dir / f'time_series_{args.arch}.csv'}")


# ---------- Scope mode ----------
def run_scope(args):
    eng = import_engine(args.engine_path)

    fp = Path(args.in_file)
    blob = eng.load_backup(fp)
    arches = eng.detect_archives(blob)
    if args.arch not in arches:
        raise SystemExit(f"--arch {args.arch} not found. Available: {list(arches.keys())}")
    archive: Dict[str, Any] = arches[args.arch]

    bt = tuple(int(x) for x in args.bin.split(",") if x.strip() != "")
    rec = eng.find_record(archive, bt)
    if rec is None:
        raise SystemExit(f"Bin {bt} not found.")

    genome = eng.genome_from_record(rec)

    repo_path = Path(args.repo_path).expanduser().resolve()
    Pic10Sim = import_pic10sim(repo_path)
    sim = Pic10Sim()
    sim.load(program=[(int(op), int(arg)) for op, arg in genome], opcode_list=eng.OPCODE_LIST)

    steps = int(args.steps)

    if args.inputs_csv:
        df_in = pd.read_csv(args.inputs_csv)
        if "gp2" not in df_in.columns or "gp3" not in df_in.columns:
            raise SystemExit("inputs_csv must have columns gp2,gp3")
        gp2 = df_in["gp2"].to_numpy(dtype=int)
        gp3 = df_in["gp3"].to_numpy(dtype=int)
        if len(gp2) < steps or len(gp3) < steps:
            raise SystemExit("inputs_csv must have at least 'steps' rows")
    else:
        gp2 = np.zeros(steps, dtype=int)
        gp3 = np.zeros(steps, dtype=int)

    out_gp0 = np.zeros(steps, dtype=int)
    out_gp1 = np.zeros(steps, dtype=int)
    crashed_at = None
    crash_reason = ""

    for i in range(steps):
        out_bits, crashed = sim.emulate_cycle(int(gp2[i]), int(gp3[i]))
        out_gp0[i] = 1 if (out_bits & 0x01) else 0
        out_gp1[i] = 1 if (out_bits & 0x02) else 0
        if crashed:
            crashed_at = i
            crash_reason = str(sim.crash_reason)
            break

    T = crashed_at + 1 if crashed_at is not None else steps
    t = np.arange(T)

    plt.figure(figsize=(10, 6))
    plt.plot(t, gp2[:T], label="GP2_in")
    plt.plot(t, gp3[:T], label="GP3_in")
    plt.plot(t, out_gp1[:T], label="GP1_out")
    plt.plot(t, out_gp0[:T], label="GP0_out")
    plt.ylim(-0.2, 1.2)
    plt.xlabel("cycle")
    plt.ylabel("digital level")
    title = f"Oscilloscope | arch={args.arch} bin={bt}"
    if crashed_at is not None:
        title += f" | CRASH at {crashed_at} ({crash_reason})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    outp = Path(args.out).expanduser()
    ensure_dir(outp.parent)
    plt.savefig(outp, dpi=200)
    plt.close()

    print(f"Wrote: {outp}")
    if crashed_at is not None:
        print(f"Crashed at cycle {crashed_at}: {crash_reason}")


def build_parser():
    ap = argparse.ArgumentParser(description="Temporal dynamics explorer + oscilloscope.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_series = sub.add_parser("series", help="Compute time-series metrics across a sequence of backups.")
    p_series.add_argument("--glob", default=None, help="Glob pattern for backups, quoted.")
    p_series.add_argument("--inputs", nargs="*", default=None, help="Explicit list of backup paths.")
    p_series.add_argument("--out", dest="out_dir", required=True, help="Output directory.")
    p_series.add_argument("--arch", default="unified", choices=["unified", "alice", "bob"], help="Archive to analyze.")
    p_series.add_argument("--engine_path", default=None, help="Folder containing PIC_Genome_Explorer.py")
    p_series.add_argument("--default_dim", type=int, default=16, help="Minimum axis size (default 16).")
    p_series.set_defaults(func=run_series)

    p_scope = sub.add_parser("scope", help="Oscilloscope plot for one critter using Pic10Sim.")
    p_scope.add_argument("--in", dest="in_file", required=True)
    p_scope.add_argument("--arch", default="unified", choices=["unified", "alice", "bob"])
    p_scope.add_argument("--bin", required=True)
    p_scope.add_argument("--repo_path", required=True, help="Path to folder containing Pic10Sim.py")
    p_scope.add_argument("--steps", type=int, default=4000)
    p_scope.add_argument("--inputs_csv", default=None, help="CSV with columns gp2,gp3")
    p_scope.add_argument("--engine_path", default=None, help="Folder containing PIC_Genome_Explorer.py")
    p_scope.add_argument("--out", required=True, help="Output PNG path")
    p_scope.set_defaults(func=run_scope)

    return ap

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
