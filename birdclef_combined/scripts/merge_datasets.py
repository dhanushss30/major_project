#!/usr/bin/env python3
"""
merge_datasets.py — Multi-Year BirdCLEF Dataset Merger
=======================================================
Downloads BirdCLEF 2021-2024 data and merges it with BirdCLEF 2025.
Only keeps species that exist in BirdCLEF 2025's 206-class label set.

This gives ~3-4× more training data for overlapping species,
especially helping rare species that have few 2025 samples.

Year stats (overlapping species with 2025):
  BirdCLEF 2021: ~62,000 recordings (many overlap)
  BirdCLEF 2022: ~14,000 recordings
  BirdCLEF 2023: ~16,000 recordings
  BirdCLEF 2024: ~24,000 recordings
  BirdCLEF 2025: ~24,000 recordings  ← base
  ─────────────────────────────────
  Combined:      ~100,000+ recordings (after species filter)

Expected AUC gain: +0.01–0.03 (especially for rare species)

Usage:
  python scripts/merge_datasets.py \\
      --data_root_2025 /workspace/birdclef-2025 \\
      --download_dir   /workspace/prior_years \\
      --output_dir     /workspace/birdclef-merged \\
      [--skip_download]   # if already downloaded

Output:
  /workspace/birdclef-merged/
  ├── train_merged.csv          ← unified metadata (use this for training)
  ├── train_audio/              ← symlinks to all audio files
  └── merge_summary.json        ← stats per year
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Year configurations ────────────────────────────────────────────────────

YEAR_CONFIGS = {
    2021: {
        "competition": "birdclef-2021",
        "metadata_candidates": ["train_metadata.csv", "train.csv"],
        "audio_dirs": ["train_short_audio", "train_audio"],
        "filename_col_candidates": ["filename"],
    },
    2022: {
        "competition": "birdclef-2022",
        "metadata_candidates": ["train_metadata.csv", "train.csv"],
        "audio_dirs": ["train_audio"],
        "filename_col_candidates": ["filename"],
    },
    2023: {
        "competition": "birdclef-2023",
        "metadata_candidates": ["train_metadata.csv", "train.csv"],
        "audio_dirs": ["train_audio"],
        "filename_col_candidates": ["filename"],
    },
    2024: {
        "competition": "birdclef-2024",
        "metadata_candidates": ["train_metadata.csv", "train.csv"],
        "audio_dirs": ["train_audio"],
        "filename_col_candidates": ["filename"],
    },
}

AUDIO_EXTENSIONS = {".ogg", ".mp3", ".wav", ".flac"}


# ── Helpers ────────────────────────────────────────────────────────────────

def run_cmd(cmd: str, check: bool = True) -> int:
    logger.info(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        logger.error(f"  Command failed with code {result.returncode}")
    return result.returncode


def download_year(year: int, download_dir: Path) -> bool:
    """Download a prior-year competition dataset using Kaggle CLI."""
    cfg = YEAR_CONFIGS[year]
    comp = cfg["competition"]
    year_dir = download_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    for meta_name in cfg["metadata_candidates"]:
        if (year_dir / meta_name).exists():
            logger.info(f"  {year}: already downloaded ({year_dir})")
            return True

    logger.info(f"  Downloading BirdCLEF {year} ({comp})...")
    rc = run_cmd(
        f"kaggle competitions download -c {comp} -p {year_dir} --quiet",
        check=False,
    )

    if rc != 0:
        logger.warning(f"  BirdCLEF {year}: download failed (rc={rc}). "
                       "You may need to accept competition rules at kaggle.com. "
                       "Skipping this year.")
        return False

    # Extract zip
    for zf in year_dir.glob("*.zip"):
        logger.info(f"  Extracting {zf.name}...")
        run_cmd(f"unzip -qo {zf} -d {year_dir}", check=False)
        break

    return True


def find_metadata(year_dir: Path, year: int) -> pd.DataFrame | None:
    """Find and load metadata CSV for a given year."""
    cfg = YEAR_CONFIGS[year]
    for name in cfg["metadata_candidates"]:
        path = year_dir / name
        if path.exists():
            logger.info(f"  Found metadata: {path}")
            return pd.read_csv(path)

    # Recursive search
    for csv_path in sorted(year_dir.rglob("train*.csv")):
        logger.info(f"  Found metadata (recursive): {csv_path}")
        return pd.read_csv(csv_path)

    return None


def find_audio_dir(year_dir: Path, year: int) -> Path | None:
    """Find audio directory for a given year."""
    cfg = YEAR_CONFIGS[year]
    for audio_dir_name in cfg["audio_dirs"]:
        p = year_dir / audio_dir_name
        if p.exists() and p.is_dir():
            return p
        # Recursive search
        matches = list(year_dir.rglob(audio_dir_name))
        if matches:
            return matches[0]
    return None


def normalize_metadata(df: pd.DataFrame, year: int, audio_dir: Path) -> pd.DataFrame:
    """
    Normalize a year's metadata to consistent column names.

    Required output columns:
        primary_label, secondary_labels, filename, rating, author, year
    """
    df = df.copy()

    # ── primary_label ────────────────────────────────────────────────────
    for col in ["primary_label", "ebird_code", "species_code"]:
        if col in df.columns:
            df["primary_label"] = df[col]
            break
    if "primary_label" not in df.columns:
        logger.warning(f"  {year}: no primary_label column found, skipping")
        return pd.DataFrame()

    # ── secondary_labels ─────────────────────────────────────────────────
    for col in ["secondary_labels", "secondary_label", "ebird_code_secondary"]:
        if col in df.columns:
            df["secondary_labels"] = df[col].fillna("").astype(str)
            break
    if "secondary_labels" not in df.columns:
        df["secondary_labels"] = ""

    # ── filename ─────────────────────────────────────────────────────────
    for col in ["filename", "filepath", "path"]:
        if col in df.columns:
            df["filename"] = df[col].astype(str)
            break
    if "filename" not in df.columns:
        logger.warning(f"  {year}: no filename column found")
        return pd.DataFrame()

    # Ensure filename is relative (species/XCxxxxx.ogg)
    df["filename"] = df["filename"].apply(
        lambda f: "/".join(Path(f).parts[-2:]) if len(Path(f).parts) >= 2
        else Path(f).name
    )

    # ── rating ───────────────────────────────────────────────────────────
    for col in ["rating", "quality_rating", "quality"]:
        if col in df.columns:
            df["rating"] = pd.to_numeric(df[col], errors="coerce").fillna(3.0)
            break
    if "rating" not in df.columns:
        df["rating"] = 3.0

    # ── author ───────────────────────────────────────────────────────────
    for col in ["author", "recordist", "recordist_name"]:
        if col in df.columns:
            df["author"] = df[col].fillna(f"unknown_{year}").astype(str)
            break
    if "author" not in df.columns:
        df["author"] = f"unknown_{year}"

    # ── year + audio_root ─────────────────────────────────────────────────
    df["year"] = year
    df["audio_root"] = str(audio_dir)

    # Keep only required columns
    keep = ["primary_label", "secondary_labels", "filename",
            "rating", "author", "year", "audio_root"]
    df = df[[c for c in keep if c in df.columns]].copy()

    return df.dropna(subset=["primary_label", "filename"])


def create_symlinks(
    df: pd.DataFrame,
    merged_audio_dir: Path,
) -> pd.DataFrame:
    """
    Create symlinks in merged_audio_dir for all audio files.
    Updates df["filename"] to reflect merged path.
    Returns df with only rows where audio file was found.
    """
    merged_audio_dir.mkdir(parents=True, exist_ok=True)
    valid_rows = []

    for _, row in df.iterrows():
        audio_root = Path(row["audio_root"])
        filename   = row["filename"]
        src = audio_root / filename

        # Try without subdir (some years store flat)
        if not src.exists():
            src = audio_root / Path(filename).name

        if not src.exists():
            continue

        # Create species subdirectory in merged dir
        dest_dir = merged_audio_dir / Path(filename).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = merged_audio_dir / filename

        if not dest.exists():
            try:
                os.symlink(src.resolve(), dest)
            except Exception:
                try:
                    shutil.copy2(str(src), str(dest))
                except Exception:
                    continue

        valid_rows.append(row)

    return pd.DataFrame(valid_rows)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Merge multi-year BirdCLEF datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root_2025", default="/workspace/birdclef-2025",
                   help="Path to BirdCLEF 2025 dataset")
    p.add_argument("--download_dir",   default="/workspace/prior_years",
                   help="Directory to download prior year data into")
    p.add_argument("--output_dir",     default="/workspace/birdclef-merged",
                   help="Output directory for merged dataset")
    p.add_argument("--years",          nargs="+", type=int,
                   default=[2021, 2022, 2023, 2024],
                   help="Prior years to include")
    p.add_argument("--skip_download",  action="store_true",
                   help="Skip kaggle download (data already in --download_dir)")
    p.add_argument("--min_rating",     type=float, default=0.0,
                   help="Minimum recording quality rating to include (0-5)")
    args = p.parse_args()

    data_root_2025 = Path(args.data_root_2025)
    download_dir   = Path(args.download_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load BirdCLEF 2025 label set ─────────────────────────────
    logger.info("Step 1: Loading BirdCLEF 2025 label set...")
    meta_2025_path = data_root_2025 / "train.csv"
    if not meta_2025_path.exists():
        meta_2025_path = data_root_2025 / "train_metadata.csv"
    if not meta_2025_path.exists():
        logger.error(f"BirdCLEF 2025 metadata not found at {data_root_2025}")
        sys.exit(1)

    df_2025 = pd.read_csv(meta_2025_path)
    label_set_2025 = set(df_2025["primary_label"].unique())
    logger.info(f"  BirdCLEF 2025: {len(df_2025):,} recordings, "
                f"{len(label_set_2025)} species")

    # Normalize 2025 metadata
    df_2025["year"]       = 2025
    df_2025["audio_root"] = str(data_root_2025 / "train_audio")
    if "author" not in df_2025.columns:
        df_2025["author"] = "unknown_2025"
    if "secondary_labels" not in df_2025.columns:
        df_2025["secondary_labels"] = ""
    if "rating" not in df_2025.columns:
        df_2025["rating"] = 3.0

    # ── Step 2: Download + process prior years ────────────────────────────
    logger.info(f"\nStep 2: Processing prior years {args.years}...")
    all_dfs = [df_2025]
    summary = {
        2025: {
            "n_raw": len(df_2025),
            "n_filtered": len(df_2025),
            "n_species_overlap": len(label_set_2025),
        }
    }

    for year in args.years:
        logger.info(f"\n── BirdCLEF {year} ──────────────────────────────────")
        year_dir = download_dir / str(year)

        if not args.skip_download:
            success = download_year(year, download_dir)
            if not success:
                logger.warning(f"  Skipping {year}")
                continue
        else:
            if not year_dir.exists():
                logger.warning(f"  {year_dir} not found, skipping {year}")
                continue

        # Load metadata
        df_year = find_metadata(year_dir, year)
        if df_year is None or len(df_year) == 0:
            logger.warning(f"  No metadata found for {year}, skipping")
            continue
        logger.info(f"  Raw metadata: {len(df_year):,} recordings")

        # Find audio directory
        audio_dir = find_audio_dir(year_dir, year)
        if audio_dir is None:
            logger.warning(f"  No audio directory found for {year}, skipping")
            continue
        logger.info(f"  Audio dir: {audio_dir}")

        # Normalize
        df_norm = normalize_metadata(df_year, year, audio_dir)
        if len(df_norm) == 0:
            continue

        # Filter to 2025 label set
        n_before = len(df_norm)
        df_norm = df_norm[df_norm["primary_label"].isin(label_set_2025)]
        n_overlap = df_norm["primary_label"].nunique()
        logger.info(f"  After species filter: {len(df_norm):,}/{n_before:,} "
                    f"({n_overlap} species overlap with 2025)")

        # Quality filter
        if args.min_rating > 0 and "rating" in df_norm.columns:
            df_norm = df_norm[df_norm["rating"] >= args.min_rating]
            logger.info(f"  After rating filter (≥{args.min_rating}): {len(df_norm):,}")

        if len(df_norm) == 0:
            logger.warning(f"  No recordings left after filtering for {year}")
            continue

        summary[year] = {
            "n_raw": n_before,
            "n_filtered": len(df_norm),
            "n_species_overlap": n_overlap,
        }

        all_dfs.append(df_norm)

    # ── Step 3: Create merged audio directory with symlinks ───────────────
    logger.info(f"\nStep 3: Creating merged audio directory...")
    merged_audio_dir = output_dir / "train_audio"

    # First symlink 2025 audio (or just point to original)
    if not merged_audio_dir.exists():
        try:
            os.symlink(
                (data_root_2025 / "train_audio").resolve(),
                merged_audio_dir,
            )
            logger.info(f"  Symlinked 2025 audio: {merged_audio_dir}")
            needs_manual_links = True
        except Exception:
            merged_audio_dir.mkdir(parents=True, exist_ok=True)
            needs_manual_links = True
    else:
        needs_manual_links = True

    # Create symlinks for prior year audio
    processed_dfs = [all_dfs[0]]  # 2025 always first, no symlink needed
    for df_year in all_dfs[1:]:
        year = df_year["year"].iloc[0]
        logger.info(f"  Linking audio for {year}...")

        # If merged_audio_dir is a symlink (points to 2025), we need a real dir
        if merged_audio_dir.is_symlink():
            real_merged = output_dir / "train_audio_merged"
            real_merged.mkdir(parents=True, exist_ok=True)
            # Copy 2025 symlinks into real dir
            src_2025 = data_root_2025 / "train_audio"
            for species_dir in src_2025.iterdir():
                if species_dir.is_dir():
                    dest_species = real_merged / species_dir.name
                    if not dest_species.exists():
                        try:
                            os.symlink(species_dir.resolve(), dest_species)
                        except Exception:
                            pass
            # Replace symlink with real dir
            merged_audio_dir.unlink()
            merged_audio_dir = real_merged
            output_dir_audio = real_merged

        df_linked = create_symlinks(df_year, merged_audio_dir)
        logger.info(f"  {year}: {len(df_linked):,} audio files linked")
        if len(df_linked) > 0:
            processed_dfs.append(df_linked)

    # ── Step 4: Merge all metadata ────────────────────────────────────────
    logger.info(f"\nStep 4: Merging metadata...")
    keep_cols = ["primary_label", "secondary_labels", "filename",
                 "rating", "author", "year"]

    merged_parts = []
    for df in processed_dfs:
        part = df[[c for c in keep_cols if c in df.columns]].copy()
        merged_parts.append(part)

    df_merged = pd.concat(merged_parts, ignore_index=True)

    # Deduplicate (same XC recording might appear in multiple years)
    n_before_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["filename"])
    logger.info(f"  Total after dedup: {len(df_merged):,} "
                f"(removed {n_before_dedup - len(df_merged):,} duplicates)")

    # Ensure required columns
    if "secondary_labels" not in df_merged.columns:
        df_merged["secondary_labels"] = ""
    if "rating" not in df_merged.columns:
        df_merged["rating"] = 3.0
    if "author" not in df_merged.columns:
        df_merged["author"] = "unknown"

    df_merged["secondary_labels"] = df_merged["secondary_labels"].fillna("").astype(str)

    # ── Step 5: Save ──────────────────────────────────────────────────────
    out_csv = output_dir / "train_merged.csv"
    df_merged.to_csv(out_csv, index=False)
    logger.info(f"\nSaved merged metadata: {out_csv}")

    # Copy soundscapes symlink
    sc_src = data_root_2025 / "train_soundscapes"
    sc_dst = output_dir / "train_soundscapes"
    if sc_src.exists() and not sc_dst.exists():
        try:
            os.symlink(sc_src.resolve(), sc_dst)
        except Exception:
            pass

    # Copy taxonomy files
    for fname in ["taxonomy.csv", "sample_submission.csv"]:
        src = data_root_2025 / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))

    # ── Summary ───────────────────────────────────────────────────────────
    summary["merged_total"]   = len(df_merged)
    summary["merged_species"] = df_merged["primary_label"].nunique()
    summary["output_csv"]     = str(out_csv)

    with open(output_dir / "merge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*60)
    logger.info("MERGE COMPLETE")
    logger.info(f"  Total recordings : {len(df_merged):,}")
    logger.info(f"  Total species    : {df_merged['primary_label'].nunique()}")
    logger.info(f"  Per year:")
    for yr, info in summary.items():
        if isinstance(yr, int):
            logger.info(f"    {yr}: {info.get('n_filtered', 0):,} recordings "
                        f"({info.get('n_species_overlap', 0)} species)")
    logger.info(f"\n  Output CSV: {out_csv}")
    logger.info(f"\n  Use for training with:")
    logger.info(f"    --data_root {output_dir}")
    logger.info(f"    metadata_file: train_merged.csv")
    logger.info("="*60)


if __name__ == "__main__":
    main()
