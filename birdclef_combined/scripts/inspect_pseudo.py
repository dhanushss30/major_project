"""Quick distribution sanity check for generated pseudo-labels.

Usage:
    python scripts/inspect_pseudo.py <parquet-path>
"""

import argparse
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquet")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"rows: {len(df)} | cols: {len(df.columns)}")

    meta_cols = {"filename", "chunk_idx", "start_sec", "end_sec",
                 "soundscape_id", "chunk_id", "file_id"}
    prob_cols = [c for c in df.columns if c not in meta_cols]
    P = df[prob_cols].values.astype(np.float32)

    print(f"prob matrix shape: {P.shape}")
    print(f"per-chunk range mean: {float((P.max(1) - P.min(1)).mean()):.4f}")
    print(f"per-chunk max mean:  {float(P.max(1).mean()):.4f}")
    print(f"per-chunk max std:   {float(P.max(1).std()):.4f}")
    print(f"overall min/max:     {float(P.min()):.4f} / {float(P.max()):.4f}")

    top1 = P.argmax(1)
    unique, counts = np.unique(top1, return_counts=True)
    order = np.argsort(-counts)[:8]
    print(f"distinct top-1 classes: {len(unique)} / {P.shape[1]}")
    print("top-8 species by top-1 share:")
    for i in order:
        cls = int(unique[i])
        pct = 100 * counts[i] / len(top1)
        print(f"  class {cls:>4}: {int(counts[i]):>6} chunks ({pct:5.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
