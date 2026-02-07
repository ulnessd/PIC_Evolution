#!/usr/bin/env python3
"""
plot_phase_backup_grid.py

Create a 3x3 figure:
  - first 8 panels: paired scatter plots among axes (and optionally fitness)
  - last panel: archive population vs batch index + archive-best fitness vs batch index,
                computed across multiple backups if provided.

Usage:
  python plot_phase_backup_grid.py --in /path/to/I4_backup_phase4_000100000.json.gz
  python plot_phase_backup_grid.py --in /path/to/I4/backups/ --out out.png
  python plot_phase_backup_grid.py --in /path/to/I4/backups/ --glob "*backup_phase4_*.json.gz"

Notes:
  - If you pass a directory, the script will read all matching backup files,
    sort them by "evaluated" (or by filename as fallback), and build the time series.
  - The scatter plots use the LAST backup in the sequence (highest evaluated).
"""

import argparse
import gzip
import json
import os
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


def _load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _get_archives(d: dict):
    # Support both {"archive": {"alice":..., "bob":...}} and {"archives": {"alice":..., ...}}
    if "archive" in d and isinstance(d["archive"], dict):
        a = d["archive"].get("alice", {})
        b = d["archive"].get("bob", {})
        return a, b
    if "archives" in d and isinstance(d["archives"], dict):
        a = d["archives"].get("alice", {})
        b = d["archives"].get("bob", {})
        return a, b
    # Sometimes stored under top-level "alice"/"bob"
    a = d.get("alice", {})
    b = d.get("bob", {})
    return a, b


def _get_meta(d: dict) -> dict:
    return d.get("meta", d.get("metadata", d.get("Meta", {}))) or {}


def _infer_evaluated(meta: dict, path: Path) -> int:
    # Prefer explicit meta
    for k in ("evaluated", "n_evaluated", "evals", "pairs_evaluated"):
        if k in meta:
            try:
                return int(meta[k])
            except Exception:
                pass
    # Fallback: parse number from filename like ..._000100000...
    m = re.search(r"(\d{6,})", path.name)
    if m:
        return int(m.group(1))
    return 0


def _best_fit_from_archive(archive: dict) -> float:
    best = None
    for rec in archive.values():
        try:
            f = float(rec.get("fit", rec.get("fitness", np.nan)))
        except Exception:
            continue
        if np.isnan(f):
            continue
        best = f if best is None else max(best, f)
    return float(best) if best is not None else float("nan")


def _axes_table_from_archive(archive: dict):
    """
    Build a list of dict rows from archive records:
      - includes all numeric metric keys found in rec["metrics"]
      - includes 'fit' as well
    """
    rows = []
    for rec in archive.values():
        metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics", {}), dict) else {}
        row = {}
        # metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                row[k] = float(v)
        # fitness
        try:
            row["fit"] = float(rec.get("fit", np.nan))
        except Exception:
            row["fit"] = np.nan

        if len(row) > 0:
            rows.append(row)

    if not rows:
        return [], []

    # Determine candidate columns: numeric keys present in most rows
    # Keep stable, readable order: common “axis-like” keys first if present
    preferred = [
        "PartnerFloor", "PartnerRobustness", "ResponseVariance",
        "ChannelEntropy", "OpcodeEntropy", "RelianceIndex",
        "BobLatencyNorm", "ProtocolStability", "FitnessLive", "FitnessMute",
        "AliceFitLive", "BobFitLive", "fit"
    ]

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # Start with preferred ordering, then remaining alphabetical
    cols = [k for k in preferred if k in all_keys]
    cols += sorted([k for k in all_keys if k not in cols])

    # Convert to aligned array for plotting convenience
    # (Rows may be missing keys; fill with nan)
    table = []
    for r in rows:
        table.append([r.get(c, np.nan) for c in cols])

    return cols, np.asarray(table, dtype=float)


def _choose_8_pairs(cols):
    """
    Choose 8 axis-axis pairs for scatter plots.
    Strategy:
      - treat 'fit' as a valid axis if present
      - if >=5 variables => many pairs; pick 8 in a “useful” order
      - else use all pairs and repeat some (rare) (but usually you'll have >=5 with fit).
    """
    # Prefer plotting among “axes” first, then include fit pairings
    priority = [
        "PartnerFloor", "PartnerRobustness", "ResponseVariance",
        "ChannelEntropy", "OpcodeEntropy", "RelianceIndex",
        "BobLatencyNorm", "ProtocolStability", "fit"
    ]
    vars_ = [v for v in priority if v in cols]

    # If too few, add remaining cols
    for c in cols:
        if c not in vars_ and c != "fit":
            vars_.append(c)

    pairs = list(combinations(vars_, 2))

    # If we still have <8 (unlikely), pad by repeating from start
    if len(pairs) >= 8:
        return pairs[:8]
    while len(pairs) < 8 and pairs:
        pairs.append(pairs[len(pairs) % max(1, len(pairs))])
    return pairs


def _collect_backup_files(inp: Path, glob_pat: str):
    if inp.is_dir():
        files = sorted(inp.glob(glob_pat))
        return [p for p in files if p.is_file()]
    if inp.is_file():
        return [inp]
    raise FileNotFoundError(f"Input not found: {inp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="backup .json.gz file OR directory containing backups")
    ap.add_argument("--glob", default="*.json.gz", help="glob pattern if --in is a directory")
    ap.add_argument("--out", default=None, help="output image filename (png/pdf). If omitted, shows window.")
    ap.add_argument("--side", default="alice", choices=["alice", "bob"], help="which archive side to use for scatter plots")
    ap.add_argument("--title", default=None, help="figure title override")
    args = ap.parse_args()

    inp = Path(args.inp)

    backups = _collect_backup_files(inp, args.glob)
    if not backups:
        raise RuntimeError("No backups found to plot.")

    # Load all backups to build time series
    series = []
    loaded = []
    for p in backups:
        try:
            d = _load_json_gz(p)
        except Exception as e:
            print(f"Skipping {p} (load error: {e})")
            continue

        meta = _get_meta(d)
        a, b = _get_archives(d)
        evaluated = _infer_evaluated(meta, p)

        # bins counts
        bins_a = meta.get("bins_a", meta.get("binsA", None))
        bins_b = meta.get("bins_b", meta.get("binsB", None))
        if bins_a is None:
            bins_a = len(a)
        if bins_b is None:
            bins_b = len(b)

        # archive-best fitness from requested side
        arc = a if args.side == "alice" else b
        best_fit = _best_fit_from_archive(arc)

        series.append((evaluated, bins_a, bins_b, best_fit, p.name))
        loaded.append((p, d))

    if not series:
        raise RuntimeError("Could not load any backups successfully.")

    # Sort by evaluated (fallback to filename)
    series.sort(key=lambda t: (t[0], t[4]))
    loaded.sort(key=lambda t: ( _infer_evaluated(_get_meta(t[1]), t[0]), t[0].name ))

    # Use LAST backup for the scatter plots
    last_path, last_data = loaded[-1]
    last_a, last_b = _get_archives(last_data)
    last_arc = last_a if args.side == "alice" else last_b

    cols, X = _axes_table_from_archive(last_arc)
    if X is None or len(X) == 0:
        raise RuntimeError("No numeric metrics found in archive records to scatter plot.")

    pairs8 = _choose_8_pairs(cols)

    # Build the 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    axes = axes.flatten()

    # Scatter panels
    for i, (xk, yk) in enumerate(pairs8):
        ax = axes[i]
        xi = cols.index(xk)
        yi = cols.index(yk)

        x = X[:, xi]
        y = X[:, yi]

        # Drop NaNs
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        ax.scatter(x, y, s=10, alpha=0.35)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title(f"{yk} vs {xk} (n={len(x)})")

    # Panel 9: population + best fit vs batch index
    ax = axes[8]
    evals = np.array([t[0] for t in series], dtype=float)
    bins = np.array([t[1] if args.side == "alice" else t[2] for t in series], dtype=float)
    best = np.array([t[3] for t in series], dtype=float)

    batch_idx = np.arange(1, len(series) + 1, dtype=float)

    ax.plot(batch_idx, bins, label="archive bins")
    ax.set_xlabel("Backup index (increasing time)")
    ax.set_ylabel("Archive bins")

    ax2 = ax.twinx()
    ax2.plot(batch_idx, best, label="archive best fit")
    ax2.set_ylabel("Best fit")

    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.set_title("Archive population and best-fit vs time (from backups)")

    # Title
    if args.title:
        fig.suptitle(args.title, fontsize=14)
    else:
        fig.suptitle(
            f"Backup summary grid ({args.side}) — last={last_path.name}",
            fontsize=14
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.out:
        outp = Path(args.out)
        fig.savefig(outp, dpi=200)
        print(f"Saved: {outp}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
