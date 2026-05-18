"""mask_and_filter_pseudo.py — Post-process raw pseudo-labels.

Takes the parquet produced by run_pseudo_gen_v3_raw.sh (which used
max_prob_threshold=0.0 so all 116k chunks are stored with near-raw probs),
masks rare-class columns to zero, recomputes max_confidence, and filters
rows by a real max_prob threshold. Writes a new parquet.

Usage:
    python scripts/mask_and_filter_pseudo.py \
        --train_csv /workspace/birdclef-2025/train.csv \
        --input_parquet /workspace/pseudo_labels/pseudo_labels_iter1_full.parquet \
        --output_parquet /workspace/pseudo_labels/pseudo_labels_iter1_v3masked.parquet \
        --min_class_samples 50 \
        --max_prob_threshold 0.20
"""

import argparse
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",          required=True)
    p.add_argument("--input_parquet",      required=True)
    p.add_argument("--output_parquet",     required=True)
    p.add_argument("--min_class_samples",  type=int,   default=50)
    p.add_argument("--max_prob_threshold", type=float, default=0.20)
    args = p.parse_args()

    df_train = pd.read_csv(args.train_csv)
    counts   = df_train["primary_label"].value_counts()
    labels   = sorted(df_train["primary_label"].unique())
    rare_idx = [i for i, l in enumerate(labels) if counts[l] < args.min_class_samples]
    print(f"rare classes (train < {args.min_class_samples}): "
          f"{len(rare_idx)} of {len(labels)}")

    df = pd.read_parquet(args.input_parquet)
    print(f"input rows: {len(df):,}")

    P = np.stack(df["soft_labels"].values).astype(np.float32)
    print(f"prob matrix shape: {P.shape}")

    # Mask rare classes
    P[:, rare_idx] = 0.0
    new_max = P.max(axis=1)
    print(f"after masking rare classes — new_max stats:")
    print(f"  mean: {new_max.mean():.4f}  median: {float(np.median(new_max)):.4f}  "
          f"max: {new_max.max():.4f}")

    # Filter rows by new threshold
    keep = new_max >= args.max_prob_threshold
    print(f"rows passing max_prob >= {args.max_prob_threshold}: "
          f"{int(keep.sum()):,} of {len(df):,}")

    if keep.sum() == 0:
        print("[ERROR] no rows pass threshold — nothing to save.")
        sys.exit(1)

    P_kept = P[keep]
    df_out = df[keep].copy().reset_index(drop=True)
    df_out["soft_labels"]    = [P_kept[i].tolist() for i in range(len(P_kept))]
    df_out["max_confidence"] = new_max[keep].astype("float64")
    df_out.to_parquet(args.output_parquet, index=False)
    print(f"wrote: {args.output_parquet}  ({len(df_out):,} rows)")

    # Verification
    print()
    print("=== top-12 species in filtered set ===")
    top1 = P_kept.argmax(1)
    u, c = np.unique(top1, return_counts=True)
    print(f"distinct top-1 classes: {len(u)} of {len(labels)}")
    for i in np.argsort(-c)[:12]:
        cls = int(u[i])
        lbl = labels[cls]
        n_train = int(counts[lbl])
        pct = 100 * c[i] / len(top1)
        print(f"  cls {cls:>3}: {lbl:<14} {int(c[i]):>5} chunks "
              f"({pct:5.1f}%, {n_train:>4} train samples)")


if __name__ == "__main__":
    sys.exit(main())
