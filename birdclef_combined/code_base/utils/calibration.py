"""
calibration.py — Per-Class Adaptive Calibration for Pseudo-Labels
==================================================================
NOVEL CONTRIBUTION #3 — Not present in 1st or 2nd place solutions.

Problem: Both winning solutions use a GLOBAL threshold of 0.5 for
pseudo-label selection across all 206 species. This is deeply flawed:

  - RARE species (< 10 training samples, e.g., some Andean insects):
    The model has little confidence even when they ARE present.
    A global 0.5 threshold means they almost NEVER get pseudo-labeled,
    even in soundscapes where they genuinely occur. This perpetuates
    the imbalance — rare species stay under-represented forever.

  - COMMON species (e.g., Rufous-collared Sparrow, 200+ recordings):
    The model is overconfident. A 0.5 threshold lets through many
    false positives, degrading pseudo-label quality.

  - MODEL MISCALIBRATION: Neural networks are systematically
    overconfident (Guo et al. "On Calibration of Modern Neural
    Networks", ICML 2017). Raw probabilities ≠ true probabilities.

The 1st place "power transform" p_new = p^α compresses all probs
uniformly — it doesn't account for per-class confidence distributions.

Our solution: Per-Class Adaptive Calibration
  1. Temperature Scaling per class: fit T_c for each class c on OOF data
     calibrated_prob = sigmoid(logit / T_c)
  2. Per-class threshold: using Precision-Recall curve, select threshold
     for each class that achieves target precision (e.g., 80%)
  3. Apply calibrated thresholds in pseudo-label generation

Result: rare species with consistently lower raw confidence get
appropriately lower thresholds, ensuring they DO get pseudo-labeled.
Common species get higher thresholds to maintain quality.

References:
  - Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
  - Kull et al. "Beta calibration" (AISTATS 2017)
  - Platt, J. "Probabilistic Outputs for SVMs" (1999)

Expected gain: +0.005–0.010 AUC vs global threshold
Why: Better pseudo-labels for rare species directly improves AUC
     since competition metric is MACRO-averaged (rare species matter equally)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_auc_score

logger = logging.getLogger(__name__)


# ─── Per-Class Temperature Scaler ─────────────────────────────────────────

class PerClassTemperatureScaler(nn.Module):
    """
    Per-class temperature scaling for neural network calibration.

    Standard temperature scaling uses a single global T:
        calibrated = sigmoid(logit / T)

    We fit one T_c per class (206 temperatures total).
    This corrects for systematic per-species overconfidence.

    Fitting procedure:
        Minimize NLL on held-out OOF predictions:
        L(T_c) = -Σ [y_c * log(σ(z_c / T_c)) + (1-y_c) * log(1-σ(z_c / T_c))]

    Very fast: 206 independent 1D optimizations via LBFGS.

    Args:
        n_classes : Number of species (206 for BirdCLEF 2025)
    """

    def __init__(self, n_classes: int = 206):
        super().__init__()
        # Initialize all temperatures to 1.0 (no scaling initially)
        self.temperature = nn.Parameter(torch.ones(n_classes))
        self.n_classes   = n_classes

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, N_classes) — raw model logits
        Returns:
            (B, N_classes) — temperature-scaled probabilities
        """
        T = self.temperature.clamp(min=0.01, max=10.0).unsqueeze(0)   # (1, C)
        return torch.sigmoid(logits / T)

    def calibrate(
        self,
        oof_logits:  np.ndarray,    # (N, N_classes) — raw logits on OOF data
        oof_targets: np.ndarray,    # (N, N_classes) — ground truth labels
        n_iters:     int = 1000,
        lr:          float = 0.01,
    ) -> "PerClassTemperatureScaler":
        """
        Fit per-class temperatures using OOF predictions.

        Uses Adam optimizer with cross-entropy loss per class.
        Falls back gracefully if a class has no positive samples.

        Args:
            oof_logits  : (N, N_classes) raw logits
            oof_targets : (N, N_classes) binary targets (0/1 or smoothed)
            n_iters     : Optimization steps
            lr          : Learning rate (0.01 works well with Adam)

        Returns:
            self (for chaining)
        """
        logits_t  = torch.tensor(oof_logits,  dtype=torch.float32)
        targets_t = torch.tensor(oof_targets, dtype=torch.float32)

        # Optimize each class independently for stability
        temperatures = torch.ones(self.n_classes)

        for c in range(self.n_classes):
            pos_count = targets_t[:, c].sum().item()
            if pos_count < 2:
                # Too few positives — keep T=1.0 (no calibration)
                continue

            T_c = nn.Parameter(torch.tensor(1.0))
            opt = torch.optim.Adam([T_c], lr=lr)

            logits_c  = logits_t[:, c]
            targets_c = targets_t[:, c]

            for _ in range(n_iters):
                T_clamped = T_c.clamp(min=0.01, max=10.0)
                # Use logit-space BCE to avoid log(0) when sigmoid saturates
                # (sigmoid(x) can be exactly 0.0 or 1.0 in float32 for |x|>88)
                loss      = F.binary_cross_entropy_with_logits(
                    logits_c / T_clamped, targets_c
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

            temperatures[c] = T_c.detach().clamp(0.01, 10.0)

        with torch.no_grad():
            self.temperature.copy_(temperatures)

        T_np = temperatures.numpy()
        logger.info(
            f"Per-class calibration: "
            f"T mean={T_np.mean():.3f}, "
            f"T min={T_np.min():.3f} (class {T_np.argmin()}), "
            f"T max={T_np.max():.3f} (class {T_np.argmax()})"
        )
        logger.info("  T < 1.0 (was overconfident): "
                   f"{(T_np < 0.99).sum()} classes")
        logger.info("  T > 1.0 (was underconfident): "
                   f"{(T_np > 1.01).sum()} classes")

        return self

    def save(self, path: str):
        torch.save({"temperature": self.temperature.data,
                    "n_classes":   self.n_classes}, path)

    @classmethod
    def load(cls, path: str) -> "PerClassTemperatureScaler":
        ckpt = torch.load(path, map_location="cpu")
        obj  = cls(n_classes=ckpt["n_classes"])
        with torch.no_grad():
            obj.temperature.copy_(ckpt["temperature"])
        obj.eval()
        return obj


# ─── Per-Class Adaptive Threshold Selector ────────────────────────────────

class AdaptiveThresholdSelector:
    """
    Select per-class pseudo-label thresholds from OOF precision-recall curves.

    Instead of the global threshold = 0.5, we find the threshold for each
    class that achieves the target precision (e.g., 80%).

    This means:
      - Rare species with low model confidence: threshold could be 0.15–0.25
        → These species get pseudo-labeled (they were being ignored before)
      - Common species with high confidence: threshold could be 0.60–0.75
        → Higher quality pseudo-labels (less false positives)

    Args:
        target_precision : Desired precision at the selected threshold (default 0.80)
        min_threshold    : Never go below this (safety floor)
        max_threshold    : Never go above this (safety ceiling)
    """

    def __init__(
        self,
        target_precision: float = 0.80,
        min_threshold:    float = 0.10,
        max_threshold:    float = 0.90,
    ):
        self.target_precision = target_precision
        self.min_threshold    = min_threshold
        self.max_threshold    = max_threshold
        self.thresholds_:     Optional[np.ndarray] = None
        self.class_aucs_:     Optional[np.ndarray] = None

    def fit(
        self,
        oof_probs:   np.ndarray,    # (N, N_classes) — calibrated probabilities
        oof_targets: np.ndarray,    # (N, N_classes) — binary targets
    ) -> "AdaptiveThresholdSelector":
        """
        Fit per-class thresholds from calibrated OOF probabilities.

        For each class:
          1. Compute precision-recall curve
          2. Find threshold where precision first reaches target_precision
          3. Fall back to 0.5 if class has < 2 positive samples

        Returns self for chaining.
        """
        n_classes = oof_probs.shape[1]
        thresholds = np.full(n_classes, 0.5, dtype=np.float32)
        class_aucs = np.zeros(n_classes, dtype=np.float32)

        for c in range(n_classes):
            pos_count = oof_targets[:, c].sum()

            if pos_count < 2:
                # No positives — keep 0.5 (won't be used)
                class_aucs[c] = 0.5
                continue

            try:
                # AUC for this class
                class_aucs[c] = roc_auc_score(oof_targets[:, c], oof_probs[:, c])

                # Precision-recall curve
                precision, recall, pr_thresholds = precision_recall_curve(
                    oof_targets[:, c], oof_probs[:, c]
                )

                # Find smallest threshold where precision >= target
                # precision and recall are computed from high→low threshold
                # pr_thresholds has len = len(precision) - 1
                valid = np.where(precision[:-1] >= self.target_precision)[0]

                if len(valid) > 0:
                    # The last valid index has the lowest threshold achieving target precision
                    thresh = float(pr_thresholds[valid[-1]])
                    thresh = np.clip(thresh, self.min_threshold, self.max_threshold)
                else:
                    # Target precision never reached — use max threshold at max recall
                    thresh = self.max_threshold

                thresholds[c] = thresh

            except Exception:
                thresholds[c] = 0.5

        self.thresholds_ = thresholds
        self.class_aucs_ = class_aucs

        # Log distribution of adaptive thresholds
        logger.info(
            f"Adaptive thresholds fitted ({n_classes} classes):\n"
            f"  Mean threshold: {thresholds.mean():.3f} "
            f"(vs global 0.5)\n"
            f"  Below 0.3 (rare/low-conf classes): {(thresholds < 0.3).sum()}\n"
            f"  0.3–0.5 (moderate classes):         {((thresholds >= 0.3) & (thresholds < 0.5)).sum()}\n"
            f"  Above 0.5 (high-conf classes):      {(thresholds >= 0.5).sum()}\n"
            f"  Mean per-class AUC: {class_aucs.mean():.4f}"
        )

        return self

    def apply(
        self,
        probs:       np.ndarray,      # (N, N_classes) — probabilities
        return_mask: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply per-class thresholds to select pseudo-labels.

        Args:
            probs       : (N, N_classes) probabilities
            return_mask : If True, also return binary keep mask

        Returns:
            soft_labels : (N, N_classes) — probabilities for selected classes,
                          0.0 for classes below per-class threshold
            keep_mask   : (N,) bool — True if any class exceeds its threshold
        """
        assert self.thresholds_ is not None, "Call fit() first"

        soft_labels = probs.copy()

        # Zero out classes below their per-class threshold
        for c in range(probs.shape[1]):
            soft_labels[:, c] = np.where(
                probs[:, c] >= self.thresholds_[c],
                probs[:, c],
                0.0
            )

        # A sample is "kept" if at least one class is above its threshold
        keep_mask = (soft_labels.max(axis=1) > 0)

        return soft_labels, keep_mask

    def save(self, path: str):
        np.savez(path,
                 thresholds=self.thresholds_,
                 class_aucs=self.class_aucs_,
                 target_precision=np.array([self.target_precision]))

    @classmethod
    def load(cls, path: str) -> "AdaptiveThresholdSelector":
        data   = np.load(path)
        obj    = cls(target_precision=float(data["target_precision"][0]))
        obj.thresholds_ = data["thresholds"]
        obj.class_aucs_ = data["class_aucs"]
        return obj


# ─── Full Calibration Pipeline ─────────────────────────────────────────────

def run_calibration_pipeline(
    oof_logits:       np.ndarray,     # (N, N_classes) raw OOF logits
    oof_targets:      np.ndarray,     # (N, N_classes) ground truth
    save_dir:         str,
    target_precision: float = 0.80,
    n_temp_iters:     int   = 500,
) -> Tuple[PerClassTemperatureScaler, AdaptiveThresholdSelector]:
    """
    Full calibration pipeline: fit temperature scaler + adaptive thresholds.

    Call this once after supervised training (before pseudo-labeling).
    Save results for use during all pseudo-labeling iterations.

    Args:
        oof_logits       : Raw logits from ensemble on OOF validation data
        oof_targets      : Ground truth labels (from BirdCLEFDataset)
        save_dir         : Directory to save calibrator and threshold files
        target_precision : Precision target for threshold selection
        n_temp_iters     : Optimization steps for temperature calibration

    Returns:
        (temperature_scaler, threshold_selector)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Running per-class calibration pipeline")
    logger.info("="*60)

    # Step 1: Fit temperature scaler
    logger.info("Step 1/3: Fitting per-class temperature scaling...")
    scaler = PerClassTemperatureScaler(n_classes=oof_logits.shape[1])
    scaler.calibrate(oof_logits, oof_targets, n_iters=n_temp_iters)
    scaler.save(str(save_path / "temperature_scaler.pt"))

    # Step 2: Get calibrated probabilities
    logger.info("Step 2/3: Computing calibrated probabilities...")
    with torch.no_grad():
        calibrated_probs = scaler(
            torch.tensor(oof_logits, dtype=torch.float32)
        ).numpy()

    # Log calibration quality
    raw_probs = 1 / (1 + np.exp(-oof_logits))   # sigmoid
    for name, p in [("Raw", raw_probs), ("Calibrated", calibrated_probs)]:
        auc_vals = []
        for c in range(oof_logits.shape[1]):
            if oof_targets[:, c].sum() >= 2:
                try:
                    auc_vals.append(roc_auc_score(oof_targets[:, c], p[:, c]))
                except Exception:
                    pass
        logger.info(f"  {name} macro AUC: {np.mean(auc_vals):.4f}")

    # Step 3: Fit adaptive thresholds on calibrated probs
    logger.info("Step 3/3: Fitting per-class adaptive thresholds...")
    selector = AdaptiveThresholdSelector(target_precision=target_precision)
    selector.fit(calibrated_probs, oof_targets)
    selector.save(str(save_path / "adaptive_thresholds.npz"))

    logger.info(f"Calibration artifacts saved to: {save_path}")
    return scaler, selector


def apply_calibration_to_pseudo_labels(
    raw_probs:   np.ndarray,           # (N, N_classes) raw model probabilities
    scaler:      PerClassTemperatureScaler,
    selector:    AdaptiveThresholdSelector,
    power_alpha: float = 0.6,          # Power transform (kept for compatibility)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply full calibration pipeline to pseudo-label predictions.
    Replaces the fixed-threshold apply_pseudo_selection_logic() in pseudo_labels.py.

    Pipeline:
        raw_probs
          → temperature scaling (per-class calibration)
          → power transform (compatibility with 1st place)
          → adaptive thresholding (per-class, replaces global 0.5)
          → soft_labels, keep_mask

    Args:
        raw_probs   : (N, N_classes) raw ensemble probabilities
        scaler      : Fitted PerClassTemperatureScaler
        selector    : Fitted AdaptiveThresholdSelector
        power_alpha : Power transform exponent (0.4–0.6 across iterations)

    Returns:
        soft_labels : (N, N_classes) — calibrated soft labels (0 where below threshold)
        keep_mask   : (N,) bool — whether each sample has ≥1 confident class
    """
    # Step 1: Temperature calibration
    # Convert raw probs back to logits, apply per-class temperature
    logits = np.log(raw_probs.clip(1e-7, 1 - 1e-7) / (1 - raw_probs.clip(1e-7, 1 - 1e-7)))
    with torch.no_grad():
        calibrated = scaler(
            torch.tensor(logits, dtype=torch.float32)
        ).numpy()

    # Step 2: Power transform (consistent with 1st place iterations)
    # alpha < 1 compresses overconfident predictions toward 0.5
    calibrated = np.power(calibrated.clip(0, 1), power_alpha)

    # Step 3: Adaptive per-class thresholding
    soft_labels, keep_mask = selector.apply(calibrated)

    n_kept = keep_mask.sum()
    logger.info(f"Calibrated pseudo-label selection: "
               f"{n_kept}/{len(keep_mask)} samples kept "
               f"({n_kept/len(keep_mask)*100:.1f}%), "
               f"power_alpha={power_alpha}")

    return soft_labels, keep_mask
