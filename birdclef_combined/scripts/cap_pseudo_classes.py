"""cap_pseudo_classes.py — Per-class cap + drop suspected non-bird classes.

Reads the masked parquet from mask_and_filter_pseudo.py and:
  1. Drops rows whose top-1 class is in --drop_classes (numeric iNat taxa
     that passed the rare-class filter but aren't target birds).
  2. Caps each remaining top-1 class at --cap_per_class chunks (seeded
     random sample without replacement).

Prevents the student from collapsing into a single-class detector when
one species (e.g. trokin / Tropical Kingbird) dominates the soundscape
pseudo-labels.

Usage:
    python scripts/cap_pseudo_classes.py \
        --train_csv /workspace/birdclef-2025/train.csv \
        --input_parquet /workspace/pseudo_labels/pseudo_labels_iter1_v3masked.parquet \
        --output_parquet /workspace/pseudo_labels/pseudo_labels_iter1_v3capped.parquet \
        --cap_per_class 300 \
        --drop_classes 45 \
        --seed 42
"""

import argparse
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",       required=True)
    p.add_argument("--input_parquet",   required=True)
    p.add_argument("--output_parquet",  required=True)
    p.add_argument("--cap_per_class",   type=int, default=300)
    p.add_argument("--drop_classes",    type=int, nargs="*", default=[],
                   help="Class indices to drop entirely (e.g. 45 for cls 65448).")
    p.add_argument("--seed",            type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    df_train = pd.read_csv(args.train_csv)
    counts   = df_train["primary_label"].value_counts()
    labels   = sorted(df_train["primary_label"].unique())

    df = pd.read_parquet(args.input_parquet)
    print(f"input rows: {len(df):,}")

    P = np.stack(df["soft_labels"].values).astype(np.float32)
    top1 = P.argmax(1)
    print(f"distinct top-1 classes (input): {len(np.unique(top1))}")

    # Step 1: drop suspected non-bird classes
    if args.drop_classes:
        drop_mask = np.isin(top1, args.drop_classes)
        dropped_names = [labels[c] for c in args.drop_classes]
        print(f"dropping classes {args.drop_classes} ({dropped_names}): "
              f"{int(drop_mask.sum()):,} rows")
        keep = ~drop_mask
        df   = df[keep].reset_index(drop=True)
        P    = P[keep]
        top1 = top1[keep]
        print(f"after drop: {len(df):,} rows")

    # Step 2: per-class cap (random sample without replacement)
    keep_idx_list = []
    print(f"\nper-class cap @ {args.cap_per_class}:")
    for cls in np.unique(top1):
        cls_idx = np.where(top1 == cls)[0]
        if len(cls_idx) > args.cap_per_class:
            sampled = rng.choice(cls_idx, size=args.cap_per_class, replace=False)
            keep_idx_list.append(sampled)
            tag = f"cap {len(cls_idx):>5} -> {args.cap_per_class}"
        else:
            keep_idx_list.append(cls_idx)
            tag = f"keep {len(cls_idx):>4}"
        lbl = labels[int(cls)]
        n_train = int(counts[lbl])
        print(f"  cls {int(cls):>3}: {lbl:<14} {tag}   ({n_train:>4} train samples)")

    keep_idx = np.concatenate(keep_idx_list)
    keep_idx.sort()

    df_out = df.iloc[keep_idx].reset_index(drop=True)
    P_out  = P[keep_idx]
    df_out["soft_labels"]    = [P_out[i].tolist() for i in range(len(P_out))]
    df_out["max_confidence"] = P_out.max(axis=1).astype("float64")
    df_out.to_parquet(args.output_parquet, index=False)
    print(f"\nwrote: {args.output_parquet}  ({len(df_out):,} rows)")

    # Verification
    print()
    print("=== top-1 distribution after capping ===")
    top1_out = P_out.argmax(1)
    u, c = np.unique(top1_out, return_counts=True)
    print(f"distinct top-1 classes: {len(u)}")
    for i in np.argsort(-c):
        cls = int(u[i])
        lbl = labels[cls]
        n_train = int(counts[lbl])
        pct = 100 * c[i] / len(top1_out)
        print(f"  cls {cls:>3}: {lbl:<14} {int(c[i]):>4} chunks "
              f"({pct:5.1f}%, {n_train:>4} train samples)")


if __name__ == "__main__":
    sys.exit(main())
