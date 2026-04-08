#!/usr/bin/env python3
"""
precompute_features.py — Convert Audio → HDF5
===============================================
IMPORTANT: Use --use_torchaudio for MP3 files (avoids librosa MP3 artifacts).
Paper: "convert audio files into .hdf5 format. If you have .mp3 compression, use --use_torchaudio"

Usage:
  python scripts/precompute_features.py /data/train_audio /data/train_hdf5 --n_cores 8 --use_torchaudio
  python scripts/precompute_features.py /data/soundscapes /data/soundscapes_hdf5 --n_cores 4
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.utils.hdf5_utils import encode_directory


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src_dir",  type=str)
    p.add_argument("dst_dir",  type=str)
    p.add_argument("--sample_rate",    type=int,  default=32_000)
    p.add_argument("--n_cores",        type=int,  default=8)
    p.add_argument("--use_torchaudio", action="store_true",
                   help="Use torchaudio for MP3 (recommended)")
    p.add_argument("--extensions",     type=str,
                   default=".ogg,.mp3,.wav,.flac")
    p.add_argument("--report",         type=str,  default=None)
    args = p.parse_args()

    exts   = tuple(e.strip() for e in args.extensions.split(","))
    logger.info(f"Converting {args.src_dir} → {args.dst_dir}")
    logger.info(f"Extensions: {exts} | Workers: {args.n_cores} | "
                f"SR: {args.sample_rate} | torchaudio: {args.use_torchaudio}")

    summary = encode_directory(
        src_dir        = args.src_dir,
        dst_dir        = args.dst_dir,
        extensions     = exts,
        target_sr      = args.sample_rate,
        use_torchaudio = args.use_torchaudio,
        n_workers      = args.n_cores,
    )

    logger.info(f"Done: {summary['success']}/{summary['total']} converted, "
                f"{summary['failed']} failed")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
