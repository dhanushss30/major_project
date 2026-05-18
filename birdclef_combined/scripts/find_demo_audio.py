"""find_demo_audio.py — Pick high-quality demo audio clips per species.

For demo day, you want clean recordings of common species the model knows
well. This script:
  1. Picks species from the top-N most-trained classes (>= --min_samples)
  2. For each, finds the highest-rated training file (XC quality 4-5)
  3. Copies the best clip per species to a demo/ folder

Usage:
    python scripts/find_demo_audio.py \
        --train_csv /workspace/birdclef-2025/train.csv \
        --audio_root /workspace/birdclef-2025/train_audio \
        --output_dir /workspace/demo_audio \
        --species_count 10 \
        --min_samples 200
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",     required=True)
    p.add_argument("--audio_root",    required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--species_count", type=int, default=10,
                   help="How many distinct species to pick demo clips for.")
    p.add_argument("--min_samples",   type=int, default=200,
                   help="Only consider species with at least this many training samples.")
    p.add_argument("--clips_per_species", type=int, default=2,
                   help="Number of clips to copy per species (best-rated first).")
    args = p.parse_args()

    df = pd.read_csv(args.train_csv)
    counts = df["primary_label"].value_counts()

    well_trained = counts[counts >= args.min_samples].index.tolist()
    print(f"Well-trained species (>= {args.min_samples} samples): {len(well_trained)}")

    output = Path(args.output_dir)
    output.mkdir(exist_ok=True, parents=True)

    rating_col = None
    for c in ("rating", "Rating", "quality"):
        if c in df.columns:
            rating_col = c
            break

    picked = []
    for species in well_trained[:args.species_count]:
        rows = df[df["primary_label"] == species]
        if rating_col:
            rows = rows.sort_values(rating_col, ascending=False)
        picks = rows.head(args.clips_per_species)
        for _, row in picks.iterrows():
            src = Path(args.audio_root) / row["filename"]
            if not src.exists():
                continue
            rating = row.get(rating_col, "?") if rating_col else "?"
            dst = output / f"{species}_rating{rating}_{src.name}"
            shutil.copy2(src, dst)
            picked.append({
                "species":  species,
                "n_train":  int(counts[species]),
                "rating":   rating,
                "filename": dst.name,
            })

    out_df = pd.DataFrame(picked)
    out_df.to_csv(output / "demo_manifest.csv", index=False)
    print()
    print(f"Copied {len(picked)} demo clips to {output}/")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
