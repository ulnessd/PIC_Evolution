#!/usr/bin/env python3
"""
PIC Archive Statistics + Visualization
--------------------------------------
Loads MAP-Elites backups from Phase 1–4 (gzip'd JSON) for the PIC10F200 evolution project and
produces:
  - opcode histograms
  - summary stats (length, branching %, delay %, etc.)
  - scatter plots for 4D archive axes (6 pair plots)
  - optional Smith–Waterman neighbor homology field (computationally expensive)

Design goals:
  - Robust to Phase1/2/3 "flat archive" and Phase4 "nested alice/bob archives"
  - Robust to fitness field name differences: fitness vs fit
  - Avoid assumptions about metric keys; flatten metrics and keep bin tuple as the canonical axis coordinates

Usage examples:
  python PIC_Archive_Stats.py --in backup_phase3_10000.json.gz --out out_phase3
  python PIC_Archive_Stats.py --in I3_backup_phase4_000005000.json.gz --out out_I3 --smith_waterman --sw_max_bins 1200

Notes:
  - Genome format is expected to be list[[op_id, operand], ...]
  - Opcode IDs are mapped using OPCODE_LIST from the codebase (0 -> NOP, 1..N -> names)
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Opcode mapping
# -------------------------
OPCODE_LIST: List[str] = ['ADDWF_W', 'ADDWF_F', 'ANDWF_W', 'ANDWF_F', 'CLRF', 'CLRW', 'COMF_W', 'COMF_F', 'DECF_W', 'DECF_F', 'DECFSZ_W', 'DECFSZ_F', 'INCF_W', 'INCF_F', 'INCFSZ_W', 'INCFSZ_F', 'IORWF_W', 'IORWF_F', 'MOVF_W', 'MOVF_F', 'MOVWF', 'MOVLW', 'NOP', 'RLF_W', 'RLF_F', 'RRF_W', 'RRF_F', 'SUBWF_W', 'SUBWF_F', 'SWAPF_W', 'SWAPF_F', 'XORWF_W', 'XORWF_F', 'BCF', 'BSF', 'BTFSC', 'BTFSS', 'CALL', 'GOTO', 'RETLW', 'DELAY_MACRO']
OP_ID_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(OPCODE_LIST)}

CONTROL_FLOW = {
    "GOTO", "CALL", "RETLW",
    "BTFSC", "BTFSS",
    "INCFSZ_W", "INCFSZ_F", "DECFSZ_W", "DECFSZ_F",
}
DELAY_OP = "DELAY_MACRO"


# -------------------------
# Helpers
# -------------------------
def load_backup(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def parse_bin_key(k: str) -> Tuple[int, ...]:
    # keys are stored as strings like "(3, 7, 1, 15)"
    k = k.strip()
    if k.startswith("(") and k.endswith(")"):
        inner = k[1:-1].strip()
        if not inner:
            return tuple()
        parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
        return tuple(int(p) for p in parts)
    # fall back: try JSON list style
    if k.startswith("[") and k.endswith("]"):
        inner = k[1:-1].strip()
        if not inner:
            return tuple()
        parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
        return tuple(int(p) for p in parts)
    raise ValueError(f"Unrecognized bin key format: {k}")


def genome_to_opcode_seq(genome: List[List[int]], compress: bool = False) -> List[int]:
    # opcode IDs only
    ops = [int(pair[0]) for pair in genome]
    if not compress:
        return ops
    # run-length compress consecutive identical opcodes (helps SW speed a lot)
    out: List[int] = []
    last: Optional[int] = None
    for op in ops:
        if last is None or op != last:
            out.append(op)
            last = op
    return out


def genome_stats(genome: List[List[int]]) -> Dict[str, float]:
    n = len(genome)
    if n == 0:
        return {"code_len": 0, "branch_frac": 0.0, "delay_frac": 0.0, "unique_opcode_frac": 0.0}

    op_ids = [int(pair[0]) for pair in genome]
    op_names = [OP_ID_TO_NAME.get(op_id, f"OP{op_id}") for op_id in op_ids]

    # Existing
    n_branch = sum(1 for nm in op_names if nm in CONTROL_FLOW)
    n_delay = sum(1 for nm in op_names if nm == DELAY_OP)
    n_unique = len(set(op_names))

    # Opcode entropy (Shannon, base-2). Max is log2(#distinct opcodes used).
    counts = {}
    for op_id in op_ids:
        counts[op_id] = counts.get(op_id, 0) + 1
    H = 0.0
    for c in counts.values():
        p = c / float(n)
        H -= p * math.log(p, 2)

    # Coarse opcode-category fractions (best-effort, used in geno↔pheno plots)
    SKIP_OPS = {"BTFSC", "BTFSS", "INCFSZ_W", "INCFSZ_F", "DECFSZ_W", "DECFSZ_F"}
    JUMP_OPS = {"GOTO", "CALL", "RETLW"}

    n_skip = sum(1 for nm in op_names if nm in SKIP_OPS)
    n_jump = sum(1 for nm in op_names if nm in JUMP_OPS)

    return {
        "code_len": float(n),
        "branch_frac": float(n_branch) / float(n),
        "delay_frac": float(n_delay) / float(n),
        "unique_opcode_frac": float(n_unique) / float(n),
        "opcode_entropy": float(H),
        "skip_frac": float(n_skip) / float(n),
        "jump_frac": float(n_jump) / float(n),
    }


def flatten_metrics(metrics: Any, prefix: str = "m_") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            key = f"{prefix}{k}"
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[key] = v
            else:
                # keep repr for nested structures so we don't lose info
                out[key] = repr(v)
    return out


# -------------------------
# Record extraction
# -------------------------
@dataclass
class ArchiveFrame:
    name: str                  # 'unified' or 'alice'/'bob'
    meta: Dict[str, Any]
    df: pd.DataFrame           # one row per elite


def extract_archive_frames(backup: Dict[str, Any]) -> List[ArchiveFrame]:
    meta = backup.get("meta", {})
    arch = backup.get("archive", {})

    frames: List[ArchiveFrame] = []

    def build_df(arch_name: str, arch_dict: Dict[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for k, rec in arch_dict.items():
            bin_tuple = parse_bin_key(k)
            # fitness field varies: 'fitness' (P1-3) vs 'fit' (P4)
            fit = rec.get("fitness", rec.get("fit", None))
            row: Dict[str, Any] = {
                "archive": arch_name,
                "bin_key": k,
                "bin_tuple": bin_tuple,
                "fitness": fit,
                "origin": rec.get("origin", None),
                "id": rec.get("id", None),
                "has_bob_genome": "bob_genome" in rec,
            }

            genome = rec.get("genome", None)
            if isinstance(genome, list):
                row["genome"] = genome
                gs = genome_stats(genome)
                row.update(gs)
            else:
                row["genome"] = None

            bob_genome = rec.get("bob_genome", None)
            if isinstance(bob_genome, list):
                row["bob_genome"] = bob_genome
                row["bob_code_len"] = float(len(bob_genome))
                row["bob_branch_frac"] = genome_stats(bob_genome)["branch_frac"]
                row["bob_delay_frac"] = genome_stats(bob_genome)["delay_frac"]
            else:
                row["bob_genome"] = None

            metrics = rec.get("metrics", {})
            row.update(flatten_metrics(metrics))

            rows.append(row)

        df = pd.DataFrame(rows)
        # expand bin_tuple to columns
        if not df.empty:
            max_dim = max(len(t) for t in df["bin_tuple"])
            for d in range(max_dim):
                df[f"axis_{d}"] = df["bin_tuple"].apply(lambda t: t[d] if len(t) > d else np.nan)
        return df

    # Phase4 nested archives
    if isinstance(arch, dict) and ("alice" in arch or "bob" in arch) and all(isinstance(arch.get(k), dict) for k in arch.keys()):
        for name in ["alice", "bob"]:
            if name in arch and isinstance(arch[name], dict):
                frames.append(ArchiveFrame(name=name, meta=meta, df=build_df(name, arch[name])))
    else:
        if isinstance(arch, dict):
            frames.append(ArchiveFrame(name="unified", meta=meta, df=build_df("unified", arch)))

    return frames


# -------------------------
# Plotting & outputs
# -------------------------
def save_opcode_histogram(df: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    genomes = [g for g in df["genome"].tolist() if isinstance(g, list)]
    if not genomes:
        return

    counts = np.zeros(len(OPCODE_LIST), dtype=np.int64)  # 0..N
    for g in genomes:
        for op_id, _ in g:
            op_id = int(op_id)
            if 0 <= op_id < len(counts):
                counts[op_id] += 1

    labels = [OP_ID_TO_NAME[i] for i in range(len(counts))]
    x = np.arange(len(counts))

    plt.figure()
    plt.bar(x, counts)
    plt.xticks(x, labels, rotation=90, fontsize=7)
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Opcode usage (global)")
    plt.tight_layout()
    plt.savefig(out_dir / "opcode_hist_global.png", dpi=200)
    plt.close()

def save_opcode_histogram_overlay(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    def count_ops(df: pd.DataFrame) -> np.ndarray:
        genomes = [g for g in df["genome"].tolist() if isinstance(g, list)]
        counts = np.zeros(len(OPCODE_LIST), dtype=np.int64)
        for g in genomes:
            for op_id, _ in g:
                op_id = int(op_id)
                if 0 <= op_id < len(counts):
                    counts[op_id] += 1
        return counts

    ca = count_ops(df_a)
    cb = count_ops(df_b)

    labels = [OP_ID_TO_NAME[i] for i in range(len(OPCODE_LIST))]
    x = np.arange(len(OPCODE_LIST))

    plt.figure()
    plt.bar(x, ca, alpha=0.55, label="Alice")
    plt.bar(x, cb, alpha=0.55, label="Bob")
    plt.xticks(x, labels, rotation=90, fontsize=7)
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Opcode usage (Alice vs Bob)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "opcode_hist_overlay.png", dpi=200)
    plt.close()


def save_axis_pair_scatters_overlay(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    axis_cols = [c for c in df_a.columns if c.startswith("axis_") and df_a[c].notna().any()]
    if len(axis_cols) < 2:
        return
    axis_cols = axis_cols[:4]

    pairs = []
    for i in range(len(axis_cols)):
        for j in range(i + 1, len(axis_cols)):
            pairs.append((axis_cols[i], axis_cols[j]))

    for a, b in pairs:
        plt.figure()
        plt.scatter(df_a[a], df_a[b], s=8, alpha=0.6, label="Alice")
        plt.scatter(df_b[a], df_b[b], s=8, alpha=0.6, label="Bob")
        plt.xlabel(a)
        plt.ylabel(b)
        plt.title(f"{title_prefix} {a} vs {b} (bins) Alice vs Bob")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_overlay_{a}_vs_{b}.png", dpi=200)
        plt.close()


def save_metric_pair_scatters_overlay(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    cols = choose_axis_metric_columns(df_a)
    if len(cols) < 2:
        return
    cols = cols[:4]

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j]))

    for a, b in pairs:
        xa = pd.to_numeric(df_a[a], errors="coerce")
        xb = pd.to_numeric(df_a[b], errors="coerce")
        ma = xa.notna() & xb.notna()

        ya = pd.to_numeric(df_b[a], errors="coerce")
        yb = pd.to_numeric(df_b[b], errors="coerce")
        mb = ya.notna() & yb.notna()

        plt.figure()
        plt.scatter(xa[ma], xb[ma], s=8, alpha=0.6, label="Alice")
        plt.scatter(ya[mb], yb[mb], s=8, alpha=0.6, label="Bob")
        plt.xlabel(a.replace("m_", ""))
        plt.ylabel(b.replace("m_", ""))
        plt.title(f"{title_prefix} {a.replace('m_','')} vs {b.replace('m_','')} (metrics) Alice vs Bob")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"metric_scatter_overlay_{a}_vs_{b}.png", dpi=200)
        plt.close()

def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _scatter_overlay(df_a: pd.DataFrame, df_b: pd.DataFrame, xcol: str, ycol: str,
                    out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    xa = pd.to_numeric(df_a.get(xcol, np.nan), errors="coerce")
    ya = pd.to_numeric(df_a.get(ycol, np.nan), errors="coerce")
    ma = xa.notna() & ya.notna()

    xb = pd.to_numeric(df_b.get(xcol, np.nan), errors="coerce")
    yb = pd.to_numeric(df_b.get(ycol, np.nan), errors="coerce")
    mb = xb.notna() & yb.notna()

    if ma.sum() == 0 and mb.sum() == 0:
        return

    plt.figure()
    if ma.sum() > 0:
        plt.scatter(xa[ma], ya[ma], s=10, alpha=0.55, label="Alice")
    if mb.sum() > 0:
        plt.scatter(xb[mb], yb[mb], s=10, alpha=0.55, label="Bob")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_phase4_genotype_phenotype_overlays(df_a: pd.DataFrame, df_b: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    """
    Phase4 overlay folder: genotype↔phenotype only (Alice vs Bob superposed).
    Skips plots whose phenotype metric columns don't exist.
    """

    # ---- Phenotype metric column aliases (best-effort, because worlds vary) ----
    m_RI   = _first_existing(df_a, ["m_RelianceIndex", "m_RI", "m_Reliance", "m_reliance_index"])
    m_act  = _first_existing(df_a, ["m_Activity", "m_activity"])
    m_ce   = _first_existing(df_a, ["m_ChannelEntropy", "m_channel_entropy", "m_Channel_Entropy"])
    m_resp = _first_existing(df_a, ["m_Responsiveness", "m_responsiveness"])
    m_ps   = _first_existing(df_a, ["m_ProtocolStability", "m_protocol_stability", "m_Protocol_Stability"])
    m_sigs = _first_existing(df_a, ["m_SignalSparsity", "m_signal_sparsity", "m_CommSparsity", "m_CommunicationSparsity"])
    m_prv  = _first_existing(df_a, ["m_PartnerRobustnessVar", "m_partner_robustness_var", "m_PartnerRobustnessVariance", "m_partner_robustness_variance"])
    m_rv   = _first_existing(df_a, ["m_ResponseVariance", "m_response_variance"])
    m_sync = _first_existing(df_a, ["m_SynchronizationIndex", "m_synchronization_index", "m_SyncIndex", "m_sync_index"])
    m_mem  = _first_existing(df_a, ["m_MemoryUtilization", "m_memory_utilization", "m_MemoryUtilFraction", "m_memory_util_fraction"])
    m_rec  = _first_existing(df_a, ["m_RecoveryTime", "m_recovery_time"])

    # ---- Genotype columns (we compute these in genome_stats) ----
    g_len   = "code_len"
    g_br    = "branch_frac"
    g_del   = "delay_frac"
    g_uniq  = "unique_opcode_frac"
    g_H     = "opcode_entropy"
    g_skip  = "skip_frac"
    g_jump  = "jump_frac"

    # Helper for filenames
    def fname(x: str, y: str) -> str:
        return f"geno_pheno__{x}__vs__{y}.png".replace("/", "_")

    # ---- 19 plots (skip automatically if missing phenotype columns) ----

    # A) Algorithmic richness vs reliance
    if m_RI:
        _scatter_overlay(df_a, df_b, g_H,   m_RI, out_dir / fname(g_H,   m_RI), f"{title_prefix} OpcodeEntropy vs RelianceIndex", "Opcode Entropy (genotype)", "Reliance Index (phenotype)")
        _scatter_overlay(df_a, df_b, g_uniq,m_RI, out_dir / fname(g_uniq,m_RI), f"{title_prefix} UniqueOpcodeFrac vs RelianceIndex", "Unique opcode fraction (genotype)", "Reliance Index (phenotype)")
        _scatter_overlay(df_a, df_b, g_br,  m_RI, out_dir / fname(g_br,  m_RI), f"{title_prefix} BranchFrac vs RelianceIndex", "Branch fraction (genotype)", "Reliance Index (phenotype)")
        _scatter_overlay(df_a, df_b, g_len, m_RI, out_dir / fname(g_len, m_RI), f"{title_prefix} CodeLen vs RelianceIndex", "Code length (genotype)", "Reliance Index (phenotype)")

        # B) Timing structure vs reliance
        _scatter_overlay(df_a, df_b, g_del, m_RI, out_dir / fname(g_del, m_RI), f"{title_prefix} DelayFrac vs RelianceIndex", "Delay fraction (genotype)", "Reliance Index (phenotype)")

    if m_act:
        # (genotype↔phenotype: activity is phenotype, delay is genotype)
        _scatter_overlay(df_a, df_b, g_del, m_act, out_dir / fname(g_del, m_act), f"{title_prefix} DelayFrac vs Activity", "Delay fraction (genotype)", "Activity (phenotype)")

    if m_act and m_RI:
        _scatter_overlay(df_a, df_b, m_act, m_RI, out_dir / fname(m_act, m_RI), f"{title_prefix} Activity vs RelianceIndex", "Activity (phenotype)", "Reliance Index (phenotype)")

    # C) Information structure vs internal complexity
    if m_ce:
        _scatter_overlay(df_a, df_b, m_ce, g_H,   out_dir / fname(m_ce, g_H),   f"{title_prefix} ChannelEntropy vs OpcodeEntropy", "Channel entropy (phenotype)", "Opcode entropy (genotype)")
        _scatter_overlay(df_a, df_b, m_ce, g_br,  out_dir / fname(m_ce, g_br),  f"{title_prefix} ChannelEntropy vs BranchFrac", "Channel entropy (phenotype)", "Branch fraction (genotype)")
        _scatter_overlay(df_a, df_b, m_ce, g_len, out_dir / fname(m_ce, g_len), f"{title_prefix} ChannelEntropy vs CodeLen", "Channel entropy (phenotype)", "Code length (genotype)")

    # D) Robustness vs internal complexity
    if m_prv:
        _scatter_overlay(df_a, df_b, m_prv, g_H, out_dir / fname(m_prv, g_H), f"{title_prefix} PartnerRobustnessVar vs OpcodeEntropy", "Partner robustness variance (phenotype)", "Opcode entropy (genotype)")
    if m_rv:
        _scatter_overlay(df_a, df_b, m_rv, g_len, out_dir / fname(m_rv, g_len), f"{title_prefix} ResponseVariance vs CodeLen", "Response variance (phenotype)", "Code length (genotype)")
    if m_rec:
        _scatter_overlay(df_a, df_b, m_rec, g_br, out_dir / fname(m_rec, g_br), f"{title_prefix} RecoveryTime vs BranchFrac", "Recovery time (phenotype)", "Branch fraction (genotype)")

    # E) Homology vs phenotype (only if homology_field exists)
    # If you've computed SW, you can merge mean_sw into df earlier; simplest is to plot if column exists.
    if "mean_sw" in df_a.columns and m_RI:
        _scatter_overlay(df_a, df_b, "mean_sw", m_RI, out_dir / fname("mean_sw", m_RI), f"{title_prefix} MeanNeighborHomology vs RelianceIndex", "Mean neighbor SW score (genotype)", "Reliance Index (phenotype)")
    if "mean_sw" in df_a.columns and "fitness" in df_a.columns:
        _scatter_overlay(df_a, df_b, "mean_sw", "fitness", out_dir / fname("mean_sw", "fitness"), f"{title_prefix} MeanNeighborHomology vs Fitness", "Mean neighbor SW score (genotype)", "Fitness (phenotype/score)")
    if "mean_sw" in df_a.columns and m_act:
        _scatter_overlay(df_a, df_b, "mean_sw", m_act, out_dir / fname("mean_sw", m_act), f"{title_prefix} MeanNeighborHomology vs Activity", "Mean neighbor SW score (genotype)", "Activity (phenotype)")

    # F) Phase4-specific ecology questions (best-effort)
    if m_sync:
        _scatter_overlay(df_a, df_b, m_sync, g_H, out_dir / fname(m_sync, g_H), f"{title_prefix} SyncIndex vs OpcodeEntropy", "Synchronization index (phenotype)", "Opcode entropy (genotype)")
    if m_mem:
        _scatter_overlay(df_a, df_b, m_mem, g_br, out_dir / fname(m_mem, g_br), f"{title_prefix} MemoryUtilization vs BranchFrac", "Memory utilization (phenotype)", "Branch fraction (genotype)")
    if m_sigs:
        # We don't currently compute a true "control-flow depth"; jump/skip fractions are a decent proxy.
        _scatter_overlay(df_a, df_b, m_sigs, g_jump, out_dir / fname(m_sigs, g_jump), f"{title_prefix} SignalSparsity vs JumpFrac", "Signal sparsity (phenotype)", "Jump fraction (genotype)")
        _scatter_overlay(df_a, df_b, m_sigs, g_skip, out_dir / fname(m_sigs, g_skip), f"{title_prefix} SignalSparsity vs SkipFrac", "Signal sparsity (phenotype)", "Skip fraction (genotype)")


def save_summary_stats(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["code_len", "branch_frac", "delay_frac", "unique_opcode_frac"]
    have = [c for c in cols if c in df.columns]
    stats = {}
    for c in have:
        vals = pd.to_numeric(df[c], errors="coerce")
        stats[c] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    (out_dir / "summary_stats.json").write_text(json.dumps(stats, indent=2))


def save_axis_pair_scatters(df: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    # Identify axis columns
    axis_cols = [c for c in df.columns if c.startswith("axis_") and df[c].notna().any()]
    if len(axis_cols) < 2:
        return

    # Use first 4 axes if more exist (Map-Elites typically 4D in your project)
    axis_cols = axis_cols[:4]
    pairs = []
    for i in range(len(axis_cols)):
        for j in range(i + 1, len(axis_cols)):
            pairs.append((axis_cols[i], axis_cols[j]))

    for a, b in pairs:
        plt.figure()
        plt.scatter(df[a], df[b], s=8, alpha=0.6)
        plt.xlabel(a)
        plt.ylabel(b)
        plt.title(f"{title_prefix} {a} vs {b} (bins)")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{a}_vs_{b}.png", dpi=200)
        plt.close()

def choose_axis_metric_columns(df: pd.DataFrame) -> List[str]:
    """
    Pick 4 metric columns (m_*) that correspond to the archive axes, as best-effort.
    Phase1/2 usually: Activity, Responsiveness, ChannelEntropy, AlgoDensity
    Phase3 usually:   Activity, ChannelEntropy, AlgoDensity, RelianceIndex
    Phase4 varies; we still prefer those if present.
    """
    # Prefer these in order
    preferred_sets = [
        ["m_Activity", "m_Responsiveness", "m_ChannelEntropy", "m_AlgoDensity"],
        ["m_Activity", "m_ChannelEntropy", "m_AlgoDensity", "m_RelianceIndex"],
        ["m_Activity", "m_ChannelEntropy", "m_AlgoDensity", "m_OpcodeEntropy"],
    ]

    for cols in preferred_sets:
        if all(c in df.columns for c in cols):
            return cols

    # Fallback: take first 4 numeric m_* columns
    m_cols = [c for c in df.columns if c.startswith("m_")]
    numeric = []
    for c in m_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric.append(c)
    return numeric[:4]

def save_metric_pair_scatters(df: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    cols = choose_axis_metric_columns(df)
    if len(cols) < 2:
        return

    # Use first 4
    cols = cols[:4]

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j]))

    for a, b in pairs:
        xa = pd.to_numeric(df[a], errors="coerce")
        xb = pd.to_numeric(df[b], errors="coerce")
        mask = xa.notna() & xb.notna()

        plt.figure()
        plt.scatter(xa[mask], xb[mask], s=8, alpha=0.6)
        plt.xlabel(a.replace("m_", ""))
        plt.ylabel(b.replace("m_", ""))
        plt.title(f"{title_prefix} {a.replace('m_','')} vs {b.replace('m_','')} (metric values)")
        plt.tight_layout()
        plt.savefig(out_dir / f"metric_scatter_{a}_vs_{b}.png", dpi=200)
        plt.close()


# -------------------------
# Smith–Waterman (local alignment)
# -------------------------
def smith_waterman_score(a: List[int], b: List[int], match: int = 2, mismatch: int = -1, gap: int = -2) -> int:
    """
    Classic Smith–Waterman local alignment score.
    O(len(a)*len(b)) time, O(len(b)) memory.

    For speed, pass compressed opcode sequences (run-length compression) and/or limit bins.
    """
    if not a or not b:
        return 0
    m = len(b)
    prev = np.zeros(m + 1, dtype=np.int32)
    best = 0

    # sequential recurrence (cannot be fully vectorized because of left dependency)
    for ai in a:
        curr = np.zeros(m + 1, dtype=np.int32)
        left = 0
        for j in range(1, m + 1):
            diag = prev[j - 1] + (match if ai == b[j - 1] else mismatch)
            up = prev[j] + gap
            left = left + gap
            v = diag
            if up > v:
                v = up
            if left > v:
                v = left
            if v < 0:
                v = 0
            curr[j] = v
            left = v
            if v > best:
                best = v
        prev = curr
    return int(best)


def compute_neighbor_homology_field(
    df: pd.DataFrame,
    out_dir: Path,
    compress: bool = True,
    max_bins: int = 1200,
    sample_seed: int = 0,
) -> None:
    """
    Compute Smith–Waterman scores between each occupied bin and its direct neighbors in the bin lattice.
    Produces:
      - homology_field.csv: bin_tuple, mean_sw, max_sw, n_neighbors_used
      - projection heatmaps for each axis-pair (mean_sw averaged in projection)
    """
    if df.empty:
        return

    # We need genome and bin_tuple
    work = df.dropna(subset=["genome"]).copy()
    work["bin_tuple"] = work["bin_tuple"].apply(lambda t: tuple(int(x) for x in t))
    # Limit bins for tractability
    if len(work) > max_bins:
        work = work.sample(n=max_bins, random_state=sample_seed).reset_index(drop=True)

    # index by bin tuple
    bin_to_genome = {}
    for _, row in work.iterrows():
        bin_to_genome[row["bin_tuple"]] = row["genome"]

    dims = max(len(t) for t in bin_to_genome.keys())
    # infer max coordinate per dim from observed (assumes 0..15)
    max_coord = [max(t[d] for t in bin_to_genome.keys() if len(t) > d) for d in range(dims)]

    def neighbors(t: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        neigh = []
        for d in range(dims):
            for delta in (-1, +1):
                tt = list(t)
                tt[d] += delta
                if tt[d] < 0 or tt[d] > max_coord[d]:
                    continue
                neigh.append(tuple(tt))
        return neigh

    rows = []
    for t, genome in bin_to_genome.items():
        a = genome_to_opcode_seq(genome, compress=compress)
        scores = []
        for nb in neighbors(t):
            if nb not in bin_to_genome:
                continue
            bgen = bin_to_genome[nb]
            b = genome_to_opcode_seq(bgen, compress=compress)
            scores.append(smith_waterman_score(a, b))
        if scores:
            rows.append({
                "bin_tuple": t,
                "mean_sw": float(np.mean(scores)),
                "max_sw": float(np.max(scores)),
                "n_neighbors_used": int(len(scores)),
            })
        else:
            rows.append({
                "bin_tuple": t,
                "mean_sw": float("nan"),
                "max_sw": float("nan"),
                "n_neighbors_used": 0,
            })

    out_df = pd.DataFrame(rows)
    # Make a stable string join key
    out_df["bin_tuple_key"] = out_df["bin_tuple"].apply(lambda t: ",".join(str(int(x)) for x in t))
    out_df.to_csv(out_dir / "homology_field.csv", index=False)

    # --- Optional: correlate homology with fitness (and write merged copy) ---
    # df already has bin_tuple_key computed in main() before calling SW. :contentReference[oaicite:2]{index=2}
    if "bin_tuple_key" in df.columns:
        sw_cols = [c for c in ["bin_tuple_key", "mean_sw", "max_sw", "n_neighbors_used"] if c in out_df.columns]
        merged = df.merge(out_df[sw_cols], on="bin_tuple_key", how="left")

        # Scatter: fitness vs mean_sw (only if fitness exists)
        if "fitness" in merged.columns:
            x = pd.to_numeric(merged["mean_sw"], errors="coerce")
            y = pd.to_numeric(merged["fitness"], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.any():
                plt.figure()
                plt.scatter(x[mask], y[mask], s=8, alpha=0.6)
                plt.xlabel("Mean neighbor SW score")
                plt.ylabel("Fitness")
                plt.title("Fitness vs mean neighbor homology (Smith–Waterman)")
                plt.tight_layout()
                plt.savefig(out_dir / "sw_scatter_fitness_vs_mean_sw.png", dpi=200)
                plt.close()


    # Make 2D projection heatmaps for mean_sw
    # Expand tuple into axis columns
    for d in range(dims):
        out_df[f"axis_{d}"] = out_df["bin_tuple"].apply(lambda x: x[d] if len(x) > d else np.nan)

    axis_cols = [f"axis_{d}" for d in range(dims)]
    axis_cols = axis_cols[:4]  # first 4
    # Try to recover human-readable axis names from metrics
    try:
        metric_cols = choose_axis_metric_columns(df)
        axis_label_map = {}
        for i, col in enumerate(axis_cols):
            if i < len(metric_cols):
                axis_label_map[col] = metric_cols[i].replace("m_", "")
            else:
                axis_label_map[col] = col
    except Exception:
        axis_label_map = {c: c for c in axis_cols}

    pairs = []
    for i in range(len(axis_cols)):
        for j in range(i + 1, len(axis_cols)):
            pairs.append((axis_cols[i], axis_cols[j]))

    for a_col, b_col in pairs:
        # pivot mean_sw by averaging over other dims
        piv = out_df.pivot_table(index=a_col, columns=b_col, values="mean_sw", aggfunc="mean")
        plt.figure()
        plt.imshow(piv.values, origin="lower", aspect="auto")
        plt.colorbar(label="mean SW score")
        xlabel = axis_label_map.get(b_col, b_col)
        ylabel = axis_label_map.get(a_col, a_col)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Mean neighbor homology projection: {ylabel} vs {xlabel}")
        plt.tight_layout()
        plt.savefig(out_dir / f"homology_heat_{a_col}_{b_col}.png", dpi=200)
        plt.close()


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_files", nargs="+", required=True, help="One or more *.json.gz backup files.")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory.")
    ap.add_argument("--smith_waterman", action="store_true", help="Compute Smith–Waterman neighbor homology field (slow).")
    ap.add_argument("--sw_compress", action="store_true", help="Run-length compress opcode sequences before SW.")
    ap.add_argument("--sw_max_bins", type=int, default=1200, help="Max bins to include in SW calculation (subsample if larger).")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    t_all0 = time.perf_counter()

    for in_path in args.in_files:
        t_file0 = time.perf_counter()
        backup = load_backup(in_path)
        frames = extract_archive_frames(backup)

        base = Path(in_path).name.replace(".json.gz", "")
        alice_df = None
        bob_df = None
        for fr in frames:
            t_file1 = time.perf_counter()
            print(f"[{base}] total file time: {t_file1 - t_file0:.2f} s")

            out_dir = out_root / f"{base}__{fr.name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            t_arch0 = time.perf_counter()

            # Save normalized dataframe
            df = fr.df.copy()
            # Stable join key for SW merge (matches homology_field.csv bin_tuple_key)
            df["bin_tuple_key"] = df["bin_tuple"].apply(lambda t: ",".join(str(int(x)) for x in t))

            if fr.name == "alice":
                alice_df = df
            elif fr.name == "bob":
                bob_df = df
            df.to_pickle(out_dir / "elites.pkl")
            # A light CSV with basic columns (genomes omitted)
            light_cols = [c for c in df.columns if c not in ("genome", "bob_genome")]
            df[light_cols].to_csv(out_dir / "elites_light.csv", index=False)

            # Save meta
            (out_dir / "meta.json").write_text(json.dumps(fr.meta, indent=2))

            # Outputs
            title = f"{base} / {fr.name}"
            save_opcode_histogram(df, out_dir, title_prefix=title)
            save_summary_stats(df, out_dir)
            save_axis_pair_scatters(df, out_dir, title_prefix=title)
            save_metric_pair_scatters(df, out_dir, title_prefix=title)

            if args.smith_waterman:
                compute_neighbor_homology_field(
                    df,
                    out_dir,
                    compress=args.sw_compress or True,   # default True in practice
                    max_bins=int(args.sw_max_bins),
                )

                # Merge SW outputs back into df (enables geno↔pheno plots using mean_sw)
                sw_path = out_dir / "homology_field.csv"
                if sw_path.exists():
                    sw = pd.read_csv(sw_path)
                    if "bin_tuple_key" in sw.columns:
                        sw_cols = [c for c in ["bin_tuple_key", "mean_sw", "max_sw", "n_neighbors_used"] if c in sw.columns]
                        df = df.merge(sw[sw_cols], on="bin_tuple_key", how="left")
                        # Re-save enriched elites outputs
                        df.to_pickle(out_dir / "elites.pkl")
                        light_cols = [c for c in df.columns if c not in ("genome", "bob_genome")]
                        df[light_cols].to_csv(out_dir / "elites_light.csv", index=False)
                        # IMPORTANT: update saved Phase4 frames so overlay sees mean_sw/max_sw
                        if fr.name == "alice":
                            alice_df = df
                        elif fr.name == "bob":
                            bob_df = df


            t_arch1 = time.perf_counter()
            print(f"[{base} / {fr.name}] done in {t_arch1 - t_arch0:.2f} s | elites={len(df)}")
        # Phase4 overlay folder if both Alice and Bob exist
        # Phase4 overlay folder if both Alice and Bob exist
        if alice_df is not None and bob_df is not None:
            overlay_dir = out_root / f"{base}__overlay"
            overlay_dir.mkdir(parents=True, exist_ok=True)

            title_overlay = f"{base} / overlay"
            # Overlay folder: genotype↔phenotype only
            save_phase4_genotype_phenotype_overlays(alice_df, bob_df, overlay_dir, title_prefix=title_overlay)



    t_all1 = time.perf_counter()
    print(f"Done. Outputs in: {out_root}")
    print(f"Total runtime: {t_all1 - t_all0:.2f} s")


if __name__ == "__main__":
    main()
