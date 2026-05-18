"""make_report_figures.py — Generate publication-quality PNGs for the report.

Outputs:
  - training_trajectory.png        (val/auc per epoch for each fold)
  - per_class_auc_distribution.png (sorted bar chart of per-class AUC)
  - confusion_matrix_top20.png     (top-20 most-confused class pairs)
  - rare_class_analysis.png        (per-class AUC vs training-sample count)
  - ensemble_vs_single.png         (single ckpt vs ensemble AUC bars)
  - "report_figures/" directory

Usage:
    python scripts/make_report_figures.py \
        --logs /workspace/eca_stage1_fold1.log /workspace/eca_stage1_fold2.log \
        --per_class_csv /workspace/ensemble_per_class_auc.csv \
        --train_csv /workspace/birdclef-2025/train.csv \
        --output_dir /workspace/report_figures
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Style
plt.rcParams.update({
    "figure.dpi":         110,
    "savefig.dpi":        160,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
})


def parse_training_log(log_path: Path) -> pd.DataFrame:
    """Extract Epoch X: val/auc = Y rows from a training log."""
    rows = []
    pattern = re.compile(r"Epoch (\d+): val/auc = ([\d\.]+)")
    if not log_path.exists():
        print(f"[WARN] log not found: {log_path}")
        return pd.DataFrame(columns=["epoch", "val_auc"])
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({"epoch": int(m.group(1)), "val_auc": float(m.group(2))})
    df = pd.DataFrame(rows)
    if not df.empty:
        # Drop epoch-0 sanity row (first occurrence per epoch is sanity, second is post-train)
        df = df.groupby("epoch", as_index=False).tail(1).reset_index(drop=True)
    return df


def fig_training_trajectory(log_paths: list, output: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for log_path in log_paths:
        df = parse_training_log(Path(log_path))
        if df.empty:
            continue
        ax.plot(df["epoch"], df["val_auc"],
                marker="o", markersize=4, lw=1.5,
                label=Path(log_path).stem)
    ax.axhline(y=0.82, color="grey", ls=":", lw=1, label="target AUC = 0.82")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation ROC AUC (macro)")
    ax.set_title("Training trajectory across folds")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"  wrote {output.name}")


def fig_per_class_auc(per_class_csv: Path, output: Path):
    if not per_class_csv.exists():
        print(f"[SKIP] {per_class_csv.name} not found")
        return
    df = pd.read_csv(per_class_csv).sort_values("auc", ascending=False)
    df = df.dropna(subset=["auc"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2ca02c" if a > 0.9 else "#1f77b4" if a > 0.75 else "#ff7f0e" if a > 0.6 else "#d62728"
              for a in df["auc"]]
    ax.bar(range(len(df)), df["auc"], color=colors, width=1.0, edgecolor="none")
    ax.axhline(y=df["auc"].mean(), color="black", ls="--", lw=1,
               label=f"macro mean = {df['auc'].mean():.3f}")
    ax.set_xlim(-1, len(df))
    ax.set_xlabel("Class rank (sorted by AUC)")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"Per-class AUC distribution ({len(df)} classes)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"  wrote {output.name}")


def fig_rare_class_analysis(per_class_csv: Path, output: Path):
    if not per_class_csv.exists():
        return
    df = pd.read_csv(per_class_csv).dropna(subset=["auc"])
    if "n_pos" not in df.columns:
        print(f"[SKIP] {per_class_csv.name} has no n_pos column")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["n_pos"], df["auc"], alpha=0.55, s=18, color="#1f77b4")
    ax.axhline(y=df["auc"].mean(), color="black", ls="--", lw=1,
               label=f"macro mean = {df['auc'].mean():.3f}")
    ax.axvline(x=50, color="grey", ls=":", lw=1, label="rare-class filter cutoff")
    ax.set_xlabel("Training samples (val set)")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Per-class AUC vs training-sample count")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"  wrote {output.name}")


def fig_ensemble_vs_single(single_aucs: dict, ensemble_auc: float, output: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(single_aucs.keys()) + ["Ensemble + TTA"]
    aucs  = list(single_aucs.values()) + [ensemble_auc]
    colors = ["#1f77b4"] * len(single_aucs) + ["#2ca02c"]
    bars = ax.bar(range(len(names)), aucs, color=colors)
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005,
                f"{a:.4f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.axhline(y=0.82, color="grey", ls=":", lw=1, label="target = 0.82")
    ax.set_ylabel("Validation ROC AUC")
    ax.set_title("Single-fold vs ensemble performance")
    ax.set_ylim(0.75, max(0.85, ensemble_auc + 0.02))
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"  wrote {output.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="*", default=[],
                   help="Training log file paths (multi-fold).")
    p.add_argument("--per_class_csv", default=None,
                   help="CSV from evaluate_ensemble.py --output_csv.")
    p.add_argument("--train_csv", default="/workspace/birdclef-2025/train.csv")
    p.add_argument("--ensemble_auc", type=float, default=None,
                   help="Final ensemble val/auc (for ensemble_vs_single.png).")
    p.add_argument("--single_aucs", nargs="*", default=[],
                   help="Per-fold single-ckpt AUCs, e.g. fold0=0.7756 fold1=0.77.")
    p.add_argument("--output_dir", default="/workspace/report_figures")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    print(f"output directory: {out}")
    print()

    # 1. Training trajectory
    if args.logs:
        fig_training_trajectory(args.logs, out / "training_trajectory.png")

    # 2. Per-class AUC distribution
    if args.per_class_csv:
        fig_per_class_auc(Path(args.per_class_csv), out / "per_class_auc_distribution.png")
        fig_rare_class_analysis(Path(args.per_class_csv), out / "rare_class_analysis.png")

    # 3. Ensemble comparison
    if args.ensemble_auc and args.single_aucs:
        single = {}
        for kv in args.single_aucs:
            if "=" in kv:
                k, v = kv.split("=", 1)
                single[k] = float(v)
        fig_ensemble_vs_single(single, args.ensemble_auc, out / "ensemble_vs_single.png")

    print()
    print(f"[OK] figures saved to {out}/")


if __name__ == "__main__":
    sys.exit(main())
