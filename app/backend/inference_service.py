"""inference_service.py — Wraps the project's ensemble inference for the API.

Loads ckpts once at startup, provides predict() and helper methods.
Uses the existing ensemble_inference.py from birdclef_combined/scripts/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import config

# Add the birdclef_combined root (for `code_base` package) AND scripts dir
# (for direct imports of ensemble_inference, etc.)
BIRDCLEF_ROOT = config.SCRIPTS_DIR.parent     # → .../birdclef_combined
sys.path.insert(0, str(BIRDCLEF_ROOT))
sys.path.insert(0, str(config.SCRIPTS_DIR))

from ensemble_inference import (
    LoadedCkpt, load_ckpt, ensemble_predict, load_audio, chunk_with_tta,
    MelExtractor, SR, CHUNK_SAMPLES, N_CLASSES,
)
import ensemble_inference as _ei

# Override ensemble_inference.load_audio to bypass torchcodec dependency.
# We re-implement using soundfile (libsndfile), which handles WAV/OGG/FLAC
# natively on Windows without requiring ffmpeg or torchcodec.
import soundfile as _sf
import torch as _torch
import torchaudio as _ta


def _ei_load_audio(path):
    data, sr_in = _sf.read(str(path), always_2d=False, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    wav = _torch.from_numpy(data)
    if sr_in != SR:
        wav = _ta.functional.resample(wav.unsqueeze(0), sr_in, SR).squeeze(0)
    return wav


_ei.load_audio = _ei_load_audio


class InferenceService:
    """Singleton-style inference service. Loads ckpts once."""

    def __init__(self):
        self._models: List[LoadedCkpt] = []
        self._labels: List[str] = []
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        print("[inference_service] loading checkpoints...")
        valid_ckpts = [p for p in config.DEFAULT_CKPTS if p.exists()]
        if not valid_ckpts:
            raise RuntimeError(
                f"No checkpoints found in {config.CKPT_DIR}. "
                f"Expected files like best-epoch07-auc0.7756.ckpt."
            )
        for p in valid_ckpts:
            lc = load_ckpt(p, device=config.DEVICE)
            self._models.append(lc)
            tag = "FiLM" if lc.has_film else "clean"
            print(f"  loaded {p.name}  [{tag}]  val_auc={lc.val_auc}")

        # Load label list — prefer train.csv, fall back to per_class_auc.csv
        train_csv = Path(config.TRAIN_CSV)
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            self._labels = sorted(df["primary_label"].astype(str).unique())
            print(f"[inference_service] {len(self._labels)} labels from train.csv")
        elif Path(config.PER_CLASS_AUC_CSV).exists():
            df = pd.read_csv(config.PER_CLASS_AUC_CSV)
            self._labels = sorted(df["label"].astype(str).unique())
            print(f"[inference_service] {len(self._labels)} labels from per_class_auc.csv")
        else:
            raise RuntimeError(
                f"No label source found. Expected one of:\n"
                f"  train.csv at {config.TRAIN_CSV}\n"
                f"  per_class_auc.csv at {config.PER_CLASS_AUC_CSV}"
            )
        if len(self._labels) != N_CLASSES:
            print(f"[WARN] Expected {N_CLASSES} labels, got {len(self._labels)}")

        self._initialized = True
        print(f"[inference_service] ready — {len(self._models)} ckpts, "
              f"{len(self._labels)} labels")

    @property
    def models(self) -> List[LoadedCkpt]:
        return self._models

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def n_ckpts(self) -> int:
        return len(self._models)

    def predict(
        self,
        audio_path: str | Path,
        tta_hop_sec: float = None,
        noise_preprocess: bool = True,
        consistency_boost: float = None,
    ) -> np.ndarray:
        """Run ensemble inference. Returns (n_chunks, 206) probability matrix."""
        if not self._initialized:
            raise RuntimeError("InferenceService not initialized. Call initialize() first.")
        return ensemble_predict(
            audio_path             = audio_path,
            ckpts                  = self._models,
            device                 = config.DEVICE,
            tta_hop_sec            = tta_hop_sec or config.TTA_HOP_SEC,
            noise_preprocess       = noise_preprocess,
            consistency_vote_boost = (consistency_boost if consistency_boost is not None
                                      else config.CONSISTENCY_BOOST),
        )


# Module-level singleton
inference_service = InferenceService()
