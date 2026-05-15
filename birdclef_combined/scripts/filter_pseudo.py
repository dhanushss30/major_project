"""filter_pseudo.py — Drop rare-class predictions from a pseudo-label parquet.

The generate_pseudo_labels.py output assigns confident predictions to *any*
class, including iNaturalist amphibians/insects with only 2-3 training
samples. These dominate the high-confidence selections because the model
overfits to their few examples. Training a student on that signal would
make it worse.

This script:
  1. Identifies "rare" classes (train sample count < min_samples).
  2. Zeros their probabilities in the soft_labels matrix.
  3. Recomputes max_confidence per row.
  4. Drops rows whose new max_confidence falls below max_prob_threshold.
  5. Writes a new parquet alongside the original.

Usage:
    python scripts/filter_pseudo.py \
        --train_csv /workspace/birdclef-2025/train.csv \
        --input_parquet  /workspace/pseudo_labels/pseudo_labels_iter1_full.parquet \
        --output_parquet /workspace/pseudo_labels/pseudo_labels_iter1_filtered.parquet \
        --min_samples 50 \
        --max_prob_threshold 0.5
"""

import argparse
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--output_parquet", required=True)
    p.add_argument("--min_samples", type=int, default=50)
    p.add_argument("--max_prob_threshold", type=float, default=0.5)
    args = p.parse_args()

    df_train = pd.read_csv(args.train_csv)
    counts = df_train["primary_label"].value_counts()
    labels = sorted(df_train["primary_label"].unique())
    rare_idx = [i for i, l in enumerate(labels) if counts[l] < args.min_samples]
    print(f"rare classes dropped (train < {args.min_samples}): "
          f"{len(rare_idx)} of {len(labels)}")

    df = pd.read_parquet(args.input_parquet)
    print(f"input rows: {len(df)}")

    P = np.stack(df["soft_labels"].values).astype(np.float32)
    P[:, rare_idx] = 0.0
    new_max = P.max(axis=1)
    keep = new_max >= args.max_prob_threshold
    print(f"rows passing new threshold (>={args.max_prob_threshold}): "
          f"{int(keep.sum())} of {len(df)}")

    P_kept = P[keep]
    df_out = df[keep].copy().reset_index(drop=True)
    df_out["soft_labels"] = [P_kept[i].tolist() for i in range(len(P_kept))]
    df_out["max_confidence"] = new_max[keep].astype("float64")
    df_out.to_parquet(args.output_parquet, index=False)
    print(f"wrote: {args.output_parquet}  ({len(df_out)} rows)")

    print()
    print("=== verification ===")
    top1 = P_kept.argmax(1)
    u, c = np.unique(top1, return_counts=True)
    print(f"distinct top-1 classes: {len(u)} of {len(labels)}")
    print("top-12 species in filtered set:")
    for i in np.argsort(-c)[:12]:
        cls = int(u[i])
        lbl = labels[cls]
        n_train = int(counts[lbl])
        pct = 100 * c[i] / len(top1)
        print(f"  cls {cls:>3}: {lbl:<14} {int(c[i]):>5} chunks "
              f"({pct:5.1f}%, {n_train:>4} train samples)")


if __name__ == "__main__":
    sys.exit(main())
