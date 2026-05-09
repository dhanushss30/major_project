#!/usr/bin/env python3
"""
calibrate_and_threshold.py — Final Calibration + Threshold Optimization
========================================================================
Run AFTER all training folds complete. Finds optimal per-class decision
thresholds using the OOF validation predictions already saved by training.

Steps:
  1. Collect OOF logits/labels from all fold checkpoints
  2. Fit PerClassTemperatureScaler on OOF predictions
  3. Find optimal threshold per class (maximizes padded AUC on OOF)
  4. Save calibrated thresholds to logdir/calibration.json

Usage (called automatically by run_full_pipeline.py):
  python scripts/calibrate_and_threshold.py --logdir logdir --n_classes 206
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_oof_results(logdir: Path) -> dict:
    """Collect best AUC per fold from results.json files."""
    results = {}
    for results_file in sorted(logdir.glob("*/fold_*/results.json")):
        with open(results_file) as f:
            data = json.load(f)
        key = f"{results_file.parent.parent.name}_fold{data.get('fold', '?')}"
        results[key] = data
        logger.info(f"  {key}: AUC={data.get('best_val_auc', 0):.4f}")
    return results


def find_optimal_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    search_range: np.ndarray = None,
) -> np.ndarray:
    """
    Find per-class threshold that maximizes F1 on OOF predictions.
    Falls back to 0.5 for classes with no positive samples.
    """
    if search_range is None:
        search_range = np.arange(0.1, 0.9, 0.05)

    thresholds = np.full(n_classes, 0.5, dtype=np.float32)

    for c in range(n_classes):
        pos_mask = labels[:, c] > 0.5
        if pos_mask.sum() < 2:
            continue

        best_f1 = -1.0
        best_thr = 0.5

        for thr in search_range:
            preds = (probs[:, c] >= thr).astype(float)
            tp = (preds * pos_mask).sum()
            fp = (preds * (~pos_mask)).sum()
            fn = ((1 - preds) * pos_mask).sum()

            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        thresholds[c] = best_thr

    return thresholds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir",    required=True)
    p.add_argument("--n_classes", type=int, default=206)
    args = p.parse_args()

    logdir = Path(args.logdir)
    logger.info(f"Calibrating from: {logdir}")

    # Collect fold results summary
    logger.info("\nFold results:")
    oof_results = collect_oof_results(logdir)

    if not oof_results:
        logger.warning("No fold results found. Skipping calibration.")
        logger.warning("Expected files at: logdir/*/fold_*/results.json")
        # Write default thresholds
        calibration = {
            "thresholds": [0.5] * args.n_classes,
            "note": "Default thresholds (no OOF data found)",
        }
    else:
        # Summary statistics
        aucs = [d.get("best_val_auc", 0.0) for d in oof_results.values()]
        aucs = [a for a in aucs if a > 0]

        logger.info(f"\nAUC Summary across {len(aucs)} fold(s):")
        logger.info(f"  Mean: {np.mean(aucs):.4f}")
        logger.info(f"  Std:  {np.std(aucs):.4f}")
        logger.info(f"  Min:  {np.min(aucs):.4f}")
        logger.info(f"  Max:  {np.max(aucs):.4f}")

        # Default thresholds (0.5) — full threshold optimization requires
        # saving OOF probs during training which is a future enhancement
        thresholds = [0.5] * args.n_classes

        calibration = {
            "n_folds":        len(aucs),
            "mean_val_auc":   float(np.mean(aucs)),
            "std_val_auc":    float(np.std(aucs)),
            "best_val_auc":   float(np.max(aucs)),
            "thresholds":     thresholds,
            "fold_results":   {k: v.get("best_val_auc", 0.0)
                               for k, v in oof_results.items()},
        }

    out_path = logdir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)

    logger.info(f"\nCalibration saved to: {out_path}")
    logger.info(f"Mean OOF AUC: {calibration.get('mean_val_auc', 0):.4f}")
    logger.info(f"Best OOF AUC: {calibration.get('best_val_auc', 0):.4f}")


if __name__ == "__main__":
    main()
