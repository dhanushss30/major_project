"""
ensemble.py — 15-Model Ensemble + Optuna Selection + TopN Postprocessing
=========================================================================
From paper §5.7 (Ensembling):

"In the final ensemble, we performed a simple average over the predictions
 of all five folds across three selected experiments, resulting in an
 ensemble of 15 models. Postprocessing was then applied to the averaged
 predictions."

Model selection strategy:
  "Selecting three experiments using Optuna, with the objective of
   maximizing the OOF validation macro ROC AUC."

Final ensemble (best Private score: 0.929):
  Experiment A: ECA-NFNet-L0 (iter 3, full pseudo) × 5 folds
  Experiment B: EfficientNetV2-S (iter 2, OOF+full pseudo) × 5 folds
  Experiment C: EfficientNetV2-S (iter 2, Fernando's B5 variant) × 5 folds
  Total: 15 models

1st place additions:
  - TTA with ±2.5s temporal delta shifts
  - Separate insect/amphibian specialist blending
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─── TTA with temporal shifts (1st place) ────────────────────────────────

class TTAPredictor:
    """
    Test-Time Augmentation with temporal delta shifts.
    1st place: process each 5s chunk at [t-2.5s, t, t+2.5s].
    """

    def __init__(
        self,
        delta_shifts: List[float] = [-2.5, 0.0, 2.5],
        sr: int = 32_000,
    ):
        self.delta_shifts = delta_shifts
        self.sr = sr

    def get_tta_chunks(
        self,
        soundscape: np.ndarray,
        start_sec: float,
        duration: float = 5.0,
    ) -> List[np.ndarray]:
        """Get audio chunks for each TTA shift."""
        n_samples = int(duration * self.sr)
        total     = len(soundscape)
        chunks    = []

        for delta in self.delta_shifts:
            shifted = int((start_sec + delta) * self.sr)
            shifted = max(0, min(shifted, total - n_samples))
            chunk   = soundscape[shifted: shifted + n_samples]
            if len(chunk) < n_samples:
                chunk = np.pad(chunk, (0, n_samples - len(chunk)))
            chunks.append(chunk.astype(np.float32))

        return chunks


# ─── Single experiment (5 fold models) ───────────────────────────────────

class ExperimentEnsemble:
    """Wraps 5 fold models from one training experiment."""

    def __init__(
        self,
        models:    List[nn.Module],
        mel:       nn.Module,
        name:      str   = "exp",
        weight:    float = 1.0,
        device:    Optional[torch.device] = None,
    ):
        self.models  = models
        self.mel     = mel
        self.name    = name
        self.weight  = weight
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict_batch(self, audio: torch.Tensor) -> torch.Tensor:
        """Average predictions from all 5 fold models."""
        audio  = audio.to(self.device)
        mel    = self.mel.to(self.device)(audio)

        probs_list = []
        for model in self.models:
            model = model.to(self.device).eval()
            probs_list.append(model.get_probabilities(mel))

        return torch.stack(probs_list).mean(dim=0)


# ─── Full 15-model ensemble ───────────────────────────────────────────────

class BirdCLEFEnsemble:
    """
    Final 15-model ensemble with:
      - 3 experiments × 5 folds = 15 models
      - Arithmetic average (paper §5.7)
      - Optional TTA (1st place ±2.5s)
      - TopN post-processing (2nd place §5.6, N=1)

    Optuna experiment selection:
      - Maximize OOF ROC AUC (not Public LB) for better Private score
    """

    def __init__(
        self,
        experiments:   List[ExperimentEnsemble],
        tta_predictor: Optional[TTAPredictor] = None,
        postprocess_n: int   = 1,       # TopN with N=1 (paper best result)
        device:        Optional[torch.device] = None,
    ):
        self.experiments    = experiments
        self.tta             = tta_predictor
        self.postprocess_n   = postprocess_n
        self.device          = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict_soundscape(
        self,
        soundscape_path: str,
        class_names:     List[str],
        chunk_duration:  float = 5.0,
        sr:              int   = 32_000,
        batch_size:      int   = 32,
        apply_topn:      bool  = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full soundscape inference pipeline.

        Returns:
            timestamps  : (S,) start times in seconds
            predictions : (S, C) post-processed probabilities
        """
        import soundfile as sf

        audio, file_sr = sf.read(soundscape_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        chunk_samples = int(chunk_duration * sr)
        n_audio       = len(audio)

        # Build chunk positions (non-overlapping)
        starts     = list(range(0, n_audio - chunk_samples + 1, chunk_samples))
        timestamps = np.array([s / sr for s in starts], dtype=np.float32)

        if len(starts) == 0:
            return timestamps, np.zeros((0, len(class_names)))

        all_probs = []   # One array per chunk

        # Process in batches
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start: batch_start + batch_size]
            batch_chunks = []

            for s in batch_starts:
                if self.tta is not None:
                    # TTA: get multiple shifted variants
                    tta_chunks = self.tta.get_tta_chunks(audio, s / sr, chunk_duration)
                    batch_chunks.append(tta_chunks)
                else:
                    chunk = audio[s: s + chunk_samples]
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                    batch_chunks.append([chunk])

            # Flatten for batched inference
            flat_chunks = [c for variants in batch_chunks for c in variants]
            n_tta       = len(batch_chunks[0])

            batch_tensor = torch.tensor(np.stack(flat_chunks), dtype=torch.float32)

            # Average over all experiments
            exp_preds = []
            for exp in self.experiments:
                preds = exp.predict_batch(batch_tensor).cpu().numpy()   # (B*n_tta, C)
                exp_preds.append(preds)

            avg_preds = np.mean(exp_preds, axis=0)   # (B*n_tta, C)

            # Average TTA variants
            avg_preds = avg_preds.reshape(-1, n_tta, avg_preds.shape[-1]).mean(axis=1)

            all_probs.append(avg_preds)

        predictions = np.concatenate(all_probs, axis=0)   # (S, C)

        # TopN post-processing (paper §5.6)
        if apply_topn:
            from ..utils.postprocessing import topn_postprocessing
            predictions = topn_postprocessing(predictions, N=self.postprocess_n)

        return timestamps, predictions


# ─── Optuna experiment selector ───────────────────────────────────────────

def select_experiments_optuna(
    oof_predictions: Dict[str, np.ndarray],   # {exp_name → (N, C) OOF probs}
    oof_targets:     np.ndarray,              # (N, C) ground truth
    n_experiments:   int = 3,
    n_trials:        int = 200,
) -> List[str]:
    """
    Use Optuna to select the best combination of experiments for ensembling.
    Objective: maximize OOF padded macro ROC-AUC.

    From paper §5.7: "Optuna-based selection yielded the highest Private score,
    suggesting that validation scores may still provide useful guidance."

    Args:
        oof_predictions : Dict mapping experiment name to OOF predictions
        oof_targets     : Ground truth for OOF samples
        n_experiments   : Number of experiments to select
        n_trials        : Optuna optimization trials

    Returns:
        List of selected experiment names
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed. Returning all experiments.")
        return list(oof_predictions.keys())[:n_experiments]

    from ..utils.postprocessing import padded_auc_score

    exp_names = list(oof_predictions.keys())

    def objective(trial):
        # Sample which experiments to include
        chosen = [
            exp_names[i] for i in range(len(exp_names))
            if trial.suggest_categorical(f"exp_{i}", [True, False])
        ]
        if len(chosen) == 0:
            return 0.0

        # Average their OOF predictions
        avg = np.mean([oof_predictions[e] for e in chosen], axis=0)
        return padded_auc_score(oof_targets, avg)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    selected    = [exp_names[i] for i in range(len(exp_names))
                   if best_params.get(f"exp_{i}", False)]

    logger.info(f"Optuna selected: {selected} (OOF AUC: {study.best_value:.4f})")
    return selected
