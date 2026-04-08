"""
hdf5_utils.py — Audio → HDF5 Pre-computation
===============================================
From paper Appendix B:
"To optimize data loading during training, audio files were converted to
 h5py format, enabling byte-wise access to random 5-second chunks."

Each file is stored as a separate HDF5 file at:
  dst_dir / relative_path.hdf5

IMPORTANT: Use --use_torchaudio for MP3 files (avoids librosa MP3 artifacts).
"""

import logging
import traceback
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import h5py
from tqdm import tqdm

logger = logging.getLogger(__name__)

TARGET_SR          = 32_000
COMPRESSION        = "gzip"
COMPRESSION_LEVEL  = 4
CHUNK_SHAPE_SECS   = 5       # HDF5 chunk size for byte-aligned access


def _load_single(
    path: str,
    target_sr: int     = TARGET_SR,
    use_torchaudio: bool = False,
) -> Optional[np.ndarray]:
    """Load one audio file and resample to target_sr."""
    try:
        if use_torchaudio:
            import torchaudio
            wave, sr = torchaudio.load(path)
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wave = torchaudio.functional.resample(wave, sr, target_sr)
            return wave.squeeze(0).numpy().astype(np.float32)
        else:
            import librosa
            audio, _ = librosa.load(path, sr=target_sr, mono=True)
            return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def _encode_file(
    args: tuple,
    src_root: str,
    dst_root: str,
    target_sr: int,
    use_torchaudio: bool,
) -> dict:
    """Worker: encode one audio file to HDF5."""
    rel_path, abs_path = args
    dst_path = Path(dst_root) / Path(rel_path).with_suffix(".hdf5")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    result = {"path": abs_path, "success": False}

    if dst_path.exists():
        result["success"]  = True
        result["skipped"]  = True
        return result

    audio = _load_single(abs_path, target_sr, use_torchaudio)
    if audio is None:
        result["error"] = "load_failed"
        return result

    try:
        chunk_samples = target_sr * CHUNK_SHAPE_SECS
        with h5py.File(str(dst_path), "w") as f:
            f.create_dataset(
                "audio",
                data              = audio,
                dtype             = np.float32,
                compression       = COMPRESSION,
                compression_opts  = COMPRESSION_LEVEL,
                chunks            = (min(chunk_samples, len(audio)),),
            )
            f.attrs["sample_rate"] = target_sr
            f.attrs["n_samples"]   = len(audio)
            f.attrs["duration"]    = len(audio) / target_sr
            f.attrs["source"]      = abs_path
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()

    return result


def encode_directory(
    src_dir:        Union[str, Path],
    dst_dir:        Union[str, Path],
    extensions:     tuple        = (".ogg", ".mp3", ".wav", ".flac"),
    target_sr:      int          = TARGET_SR,
    use_torchaudio: bool         = True,
    n_workers:      int          = 8,
) -> dict:
    """
    Convert all audio files in src_dir to HDF5 in dst_dir.
    Preserves directory structure. Skips already-encoded files.

    Returns summary dict with success/failure counts.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    for ext in extensions:
        all_files.extend(src_dir.rglob(f"*{ext}"))

    logger.info(f"Found {len(all_files):,} audio files in {src_dir}")

    args_list = [(str(f.relative_to(src_dir)), str(f)) for f in all_files]

    worker = partial(
        _encode_file,
        src_root       = str(src_dir),
        dst_root       = str(dst_dir),
        target_sr      = target_sr,
        use_torchaudio = use_torchaudio,
    )

    results = []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for r in tqdm(pool.imap(worker, args_list, chunksize=32),
                          total=len(args_list), desc="Encoding HDF5"):
                results.append(r)
    else:
        for args in tqdm(args_list, desc="Encoding HDF5"):
            results.append(worker(args))

    n_ok   = sum(r["success"] for r in results)
    n_fail = sum(not r["success"] for r in results)

    logger.info(f"Done: {n_ok:,} succeeded, {n_fail:,} failed")
    return {
        "total":        len(results),
        "success":      n_ok,
        "failed":       n_fail,
        "failed_paths": [r["path"] for r in results if not r["success"]],
    }
