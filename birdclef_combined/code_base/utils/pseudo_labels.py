"""
pseudo_labels.py — Pseudo-Label Generation & Selection (Exact 2nd Place)
=========================================================================
Implements the exact pseudo-labeling pipeline from §5.4:

Selection Logic (per 5-second chunk):
  1. Compute max predicted probability across all 206 classes
  2. Discard chunks where max_prob < 0.5   (threshold)
  3. For kept chunks: zero out class probs where prob < 0.1

Sampling Strategy (Figure 4 in paper):
  "For each training sample, if the corresponding class exists in the
   pseudo dataset, it is selected with 40% probability; otherwise,
   the original training sample is used."

Iteration statistics (Table 4):
  Iteration 1: 4,430 files → 19,405 chunks selected
  Iteration 2: 1,483 files →  3,750 chunks selected
  Iteration 3: 1,437 files →  4,108 chunks selected

OOF strategy (Figure 5):
  Split soundscapes by fold. Each fold predicted by models NOT trained on it.
  Re-generates pseudo-labels every iteration for freshness.

1st place additions:
  - Power transform: p_new = p^α  (compress teacher overconfidence)
    α: 0.6 → 0.5 → 0.4 across iterations
  - SED frame-level validation: filter chunks where SED says no species present
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─── Pseudo selection logic (exact 2nd place) ─────────────────────────────

def apply_pseudo_selection_logic(
    probs: np.ndarray,
    max_prob_threshold: float = 0.5,    # Discard if max prob < 0.5
    class_prob_threshold: float = 0.1,  # Zero out probs below 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From paper §5.4:
    "For each 5-second chunk, the maximum predicted class probability
     was computed. Chunks with a maximum probability below 0.5 were
     discarded. For retained chunks, all class probabilities below 0.1
     were zeroed out."

    Args:
        probs                : (N, C) model probabilities
        max_prob_threshold   : Discard threshold (default 0.5)
        class_prob_threshold : Zero-out threshold (default 0.1)

    Returns:
        keep_mask   : (N,) boolean mask of kept chunks
        soft_labels : (N, C) filtered soft label matrix
    """
    # Step 1: max probability per chunk
    max_probs = probs.max(axis=1)          # (N,)
    keep_mask = max_probs >= max_prob_threshold

    # Step 2: zero out low-confidence classes
    soft_labels = probs.copy()
    soft_labels[soft_labels < class_prob_threshold] = 0.0

    return keep_mask, soft_labels


# ─── Power transform (1st place) ──────────────────────────────────────────

def power_transform(
    probs: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Power transform from 1st place solution:
        p_new = p^α

    Purpose: compress overconfident teacher predictions.
    Iter 1: α=0.6  (mild compression)
    Iter 2: α=0.5  (moderate compression)
    Iter 3: α=0.4  (aggressive compression)

    When α < 1: probabilities closer to 0.5 are boosted, extreme values compressed.
    """
    return np.power(np.clip(probs, 1e-9, 1.0), alpha).astype(np.float32)


# ─── Main pseudo-label generator ─────────────────────────────────────────

class PseudoLabelGenerator:
    """
    Full pseudo-label generation pipeline.

    Supports both:
    - Full mode: predict all soundscapes with full ensemble → select
    - OOF mode:  per-fold prediction by held-out models → select (Figure 5)

    Usage:
        gen = PseudoLabelGenerator(models, mel_extractor, soundscape_ds)
        pseudo_df = gen.generate(
            iteration=1, max_prob_threshold=0.5, class_prob_threshold=0.1
        )
    """

    def __init__(
        self,
        models:             List[nn.Module],     # List of trained models (fold models)
        mel_extractor:      nn.Module,
        soundscape_dataset,                       # SoundscapeDataset instance
        device:             torch.device = None,
        batch_size:         int          = 64,
        num_workers:        int          = 4,
        n_classes:          int          = 206,
    ):
        self.models       = models
        self.mel          = mel_extractor
        self.ds           = soundscape_dataset
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.n_classes    = n_classes

    @torch.no_grad()
    def _run_inference(self, model: nn.Module) -> np.ndarray:
        """Run single model inference on all soundscape chunks."""
        model = model.to(self.device).eval()
        self.mel = self.mel.to(self.device).eval()

        loader = DataLoader(
            self.ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

        all_probs = []
        for batch in tqdm(loader, desc="Inference", leave=False):
            audio = batch["audio"].to(self.device)
            mel   = self.mel(audio)

            if hasattr(model, "get_probabilities"):
                prob = model.get_probabilities(mel)
            else:
                prob = torch.sigmoid(model(mel))

            all_probs.append(prob.cpu().numpy())

        return np.concatenate(all_probs, axis=0)   # (N, C)

    def generate(
        self,
        iteration:              int   = 1,
        max_prob_threshold:     float = 0.5,    # Paper: 0.5
        class_prob_threshold:   float = 0.1,    # Paper: 0.1
        power_alpha:            float = 1.0,    # 1.0 = no transform; <1 = 1st place
        output_path:            Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Generate pseudo-labels from ensemble prediction.

        Args:
            iteration               : Iteration number (for logging)
            max_prob_threshold      : Discard chunks below this max prob
            class_prob_threshold    : Zero out class probs below this
            power_alpha             : Power transform exponent (1st place)
            output_path             : Save parquet here if provided

        Returns:
            DataFrame with columns:
                filename, start_sec, soft_labels (list of C floats),
                max_confidence, iteration
        """
        logger.info(f"Generating pseudo-labels (iteration {iteration}) "
                    f"using {len(self.models)} models...")

        # Average ensemble predictions
        all_model_probs = []
        for model in self.models:
            probs = self._run_inference(model)
            all_model_probs.append(probs)

        ensemble_probs = np.mean(all_model_probs, axis=0)   # (N, C)
        logger.info(f"Ensemble shape: {ensemble_probs.shape}")

        # Apply power transform (1st place)
        if power_alpha != 1.0:
            ensemble_probs = power_transform(ensemble_probs, alpha=power_alpha)
            logger.info(f"Applied power transform α={power_alpha}")

        # Apply selection logic (2nd place)
        keep_mask, soft_labels = apply_pseudo_selection_logic(
            ensemble_probs,
            max_prob_threshold   = max_prob_threshold,
            class_prob_threshold = class_prob_threshold,
        )

        n_kept = keep_mask.sum()
        logger.info(
            f"Iteration {iteration}: {n_kept:,}/{len(keep_mask):,} chunks kept "
            f"({100*n_kept/max(len(keep_mask),1):.1f}%)"
        )

        # Collect metadata from dataset
        filenames  = []
        start_secs = []
        for item in self.ds:
            filenames.append(item["filename"])
            start_secs.append(float(item["start_sec"]))

        # Build DataFrame
        idx_arr = np.where(keep_mask)[0]
        pseudo_df = pd.DataFrame({
            "filename":       [filenames[i]              for i in idx_arr],
            "start_sec":      [start_secs[i]             for i in idx_arr],
            "soft_labels":    [soft_labels[i].tolist()   for i in idx_arr],
            "max_confidence": [float(ensemble_probs[i].max()) for i in idx_arr],
            "iteration":      iteration,
        })

        # Deduplicate: keep max-confidence chunk for each (file, start_sec)
        pseudo_df = pseudo_df.sort_values("max_confidence", ascending=False)
        pseudo_df = pseudo_df.drop_duplicates(subset=["filename", "start_sec"])

        logger.info(f"Final pseudo-label count: {len(pseudo_df):,}")

        if output_path:
            pseudo_df.to_parquet(str(output_path), index=False)
            logger.info(f"Saved to {output_path}")

        return pseudo_df

    def generate_oof(
        self,
        fold_models:            Dict[int, List[nn.Module]],
        iteration:              int   = 1,
        max_prob_threshold:     float = 0.5,
        class_prob_threshold:   float = 0.1,
        power_alpha:            float = 1.0,
        output_path:            Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        OOF (Out-Of-Fold) pseudo-label generation — paper Figure 5.

        "Soundscapes are split by fold. Each fold is predicted using
         models NOT trained on it."

        Args:
            fold_models : {fold_id → [model1, model2, ...]} mapping
                          Each fold's models are used to predict OTHER folds' soundscapes.
        """
        logger.info(f"OOF pseudo-label generation (iteration {iteration})")

        # Collect all chunk metadata first
        all_filenames  = [item["filename"]  for item in self.ds]
        all_start_secs = [float(item["start_sec"]) for item in self.ds]
        all_file_folds = [int(item["file_fold"]) for item in self.ds]

        n_total = len(all_filenames)
        ensemble_probs = np.zeros((n_total, self.n_classes), dtype=np.float32)

        from torch.utils.data import Subset

        # For each fold, predict its soundscapes using all OTHER folds' models
        unique_folds = sorted(fold_models.keys())
        for fold_id in unique_folds:
            # Indices of soundscape chunks belonging to this fold
            fold_indices = [i for i, ff in enumerate(all_file_folds) if ff == fold_id]
            if not fold_indices:
                continue

            # Models trained on all other folds
            oof_models = [m for f, ms in fold_models.items()
                          if f != fold_id for m in ms]
            if not oof_models:
                logger.warning(f"No OOF models for fold {fold_id}, skipping")
                continue

            # Predict this fold's soundscapes
            fold_ds = Subset(self.ds, fold_indices)
            fold_loader = DataLoader(
                fold_ds, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers
            )

            fold_preds = []
            for model in oof_models:
                model = model.to(self.device).eval()
                self.mel = self.mel.to(self.device).eval()
                model_preds = []

                with torch.no_grad():
                    for batch in fold_loader:
                        audio = batch["audio"].to(self.device)
                        mel   = self.mel(audio)
                        prob  = model.get_probabilities(mel)
                        model_preds.append(prob.cpu().numpy())

                fold_preds.append(np.concatenate(model_preds, axis=0))

            fold_avg = np.mean(fold_preds, axis=0)
            for local_i, global_i in enumerate(fold_indices):
                ensemble_probs[global_i] = fold_avg[local_i]

        # Apply power transform and selection
        if power_alpha != 1.0:
            ensemble_probs = power_transform(ensemble_probs, alpha=power_alpha)

        keep_mask, soft_labels = apply_pseudo_selection_logic(
            ensemble_probs, max_prob_threshold, class_prob_threshold
        )

        idx_arr   = np.where(keep_mask)[0]
        pseudo_df = pd.DataFrame({
            "filename":       [all_filenames[i]               for i in idx_arr],
            "start_sec":      [all_start_secs[i]              for i in idx_arr],
            "soft_labels":    [soft_labels[i].tolist()        for i in idx_arr],
            "max_confidence": [float(ensemble_probs[i].max()) for i in idx_arr],
            "iteration":      iteration,
            "mode":           "oof",
        })

        pseudo_df = pseudo_df.sort_values("max_confidence", ascending=False)
        pseudo_df = pseudo_df.drop_duplicates(subset=["filename", "start_sec"])
        logger.info(f"OOF iter {iteration}: {len(pseudo_df):,} chunks selected")

        if output_path:
            pseudo_df.to_parquet(str(output_path), index=False)

        return pseudo_df
