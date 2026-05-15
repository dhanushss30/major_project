"""diagnose_thresholds.py — Show new_max_confidence distribution after
masking rare classes, at multiple thresholds.

Tells us whether common-bird pseudo-label signal exists at any threshold,
or whether the model is genuinely blind to common birds in soundscape audio.

Usage:
    python scripts/diagnose_thresholds.py \
        --train_csv /workspace/birdclef-2025/train.csv \
        --parquet   /workspace/pseudo_labels/pseudo_labels_iter1_full.parquet \
        --min_samples 50
"""

import argparse
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--min_samples", type=int, default=50)
    args = p.parse_args()

    df_train = pd.read_csv(args.train_csv)
    counts = df_train["primary_label"].value_counts()
    labels = sorted(df_train["primary_label"].unique())
    rare_idx = [i for i, l in enumerate(labels) if counts[l] < args.min_samples]
    print(f"rare classes (train < {args.min_samples}): {len(rare_idx)} of {len(labels)}")

    df = pd.read_parquet(args.parquet)
    print(f"input rows: {len(df)}")

    P = np.stack(df["soft_labels"].values).astype(np.float32)
    print(f"original new_max stats (no masking):")
    om = P.max(1)
    print(f"  mean: {om.mean():.4f}  median: {float(np.median(om)):.4f}  "
          f"min: {om.min():.4f}  max: {om.max():.4f}")

    P[:, rare_idx] = 0.0
    m = P.max(1)
    print()
    print("after masking rare classes:")
    print(f"  mean: {m.mean():.4f}  median: {float(np.median(m)):.4f}  "
          f"min: {m.min():.4f}  max: {m.max():.4f}")

    print()
    print("threshold | rows kept | % of input")
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        n = int((m >= t).sum())
        pct = 100.0 * n / len(df)
        print(f"  >= {t:.2f}  |  {n:>6}   | {pct:5.1f}%")

    print()
    print("how many non-zero common-bird classes per row (at >= 0.10 they are stored):")
    nz = (P >= 0.10).sum(1)
    print(f"  mean: {nz.mean():.2f}  median: {int(np.median(nz))}  "
          f"min: {int(nz.min())}  max: {int(nz.max())}")
    print(f"  rows with at least 1 non-zero common-bird value: "
          f"{int((nz >= 1).sum())} of {len(df)}")
    print(f"  rows with at least 3 non-zero common-bird values: "
          f"{int((nz >= 3).sum())} of {len(df)}")


if __name__ == "__main__":
    sys.exit(main())
