"""
merge_pseudo_labels.py — Merge pseudo-label parquets across iterations
=======================================================================
After each iteration, we merge the new pseudo-labels with previous ones.
Duplicates (same file + start_sec) are resolved by keeping the highest-
confidence label, since later iterations have better teachers.

Usage:
    python scripts/merge_pseudo_labels.py \\
        --inputs pseudo_labels_iter1_full.parquet pseudo_labels_iter2_full.parquet \\
        --output pseudo_labels_all_iters_merged.parquet

    python scripts/merge_pseudo_labels.py \\
        --inputs D:/pseudo/pseudo_labels_iter1_full.parquet \\
                 D:/pseudo/pseudo_labels_iter2_full.parquet \\
                 D:/pseudo/pseudo_labels_iter3_full.parquet \\
        --output D:/pseudo/pseudo_labels_all_iters_merged.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Input parquet files (one per iteration)")
    p.add_argument("--output", required=True,
                   help="Output merged parquet file path")
    args = p.parse_args()

    dfs = []
    for path in args.inputs:
        if not Path(path).exists():
            logger.warning(f"File not found, skipping: {path}")
            continue
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} rows from {path}")
        dfs.append(df)

    if not dfs:
        logger.error("No input files found. Check paths.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total before dedup: {len(merged):,}")

    # Keep highest-confidence entry for each (filename, start_sec) pair
    # Later iterations have better teachers → higher max_confidence usually
    merged = merged.sort_values("max_confidence", ascending=False)
    merged = merged.drop_duplicates(subset=["filename", "start_sec"], keep="first")
    merged = merged.reset_index(drop=True)

    logger.info(f"After dedup:        {len(merged):,}")
    logger.info(f"Unique files:       {merged['filename'].nunique():,}")

    # Show per-iteration breakdown
    if "iteration" in merged.columns:
        for it, grp in merged.groupby("iteration"):
            logger.info(f"  Iteration {it}: {len(grp):,} chunks")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    logger.info(f"Saved merged parquet: {args.output}")


if __name__ == "__main__":
    main()
