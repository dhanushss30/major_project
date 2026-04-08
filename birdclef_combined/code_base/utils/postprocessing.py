"""
postprocessing.py — TopN Post-Processing (Exact 2nd Place)
============================================================
From paper §5.6:
"Our best results were consistently achieved using the TopN method with N=1."

TopN formula (Eq. 4):
    postproc_prob[a, s, c] = prob[a, s, c] * (1/N * Σ_{s' ∈ T_{a,c}} prob[a, s', c])

where T_{a,c} = set of indices for top N values of prob(a, :, c).

With N=1: multiply each segment's probability by the single highest
probability for that class in the same audio file.

Intuition: "if an animal is vocalizing in an audio recording, it will appear
in multiple segments. If a species has a high probability in other segments,
a high probability in a given segment is more trustworthy."

Also implements:
  - Mean method (Eq. 3)
  - Convolution smoothing
  - TTA prediction averaging (1st place ±2.5s delta shifts)
"""

import numpy as np
from typing import Dict, List, Optional


# ─── TopN post-processing ────────────────────────────────────────────────

def topn_postprocessing(
    predictions: np.ndarray,
    N: int = 1,
) -> np.ndarray:
    """
    TopN post-processing — best method from paper (N=1 optimal).

    postproc_prob[s, c] = prob[s, c] * mean(top-N probs for class c)

    Args:
        predictions : (S, C) predicted probabilities for one soundscape
                      S = number of 5-sec chunks, C = number of classes
        N           : Number of top segments to average (default 1)

    Returns:
        (S, C) post-processed probabilities
    """
    S, C = predictions.shape

    # Top-N mean per class: (C,) vector
    if N == 1:
        top_n_mean = predictions.max(axis=0)          # (C,)
    else:
        sorted_probs = np.sort(predictions, axis=0)[::-1, :]  # descending
        top_n_mean   = sorted_probs[:N, :].mean(axis=0)        # (C,)

    # Multiply: broadcast (C,) over (S, C)
    return predictions * top_n_mean[np.newaxis, :]


def mean_postprocessing(predictions: np.ndarray) -> np.ndarray:
    """
    Mean post-processing — Eq. 3 from paper.
    postproc_prob[s, c] = prob[s, c] * mean_s(prob[:, c])
    """
    mean_probs = predictions.mean(axis=0)     # (C,)
    return predictions * mean_probs[np.newaxis, :]


def apply_postprocessing_to_submission(
    submission_df,
    id_col:    str = "row_id",
    method:    str = "topN",
    N:         int = 1,
    class_cols: Optional[List[str]] = None,
) -> object:
    """
    Apply post-processing to a full submission DataFrame.

    The row_id format is: "{filename}_{start_sec}"
    Groups predictions by filename, applies TopN within each group.

    Args:
        submission_df : DataFrame with row_id and class probability columns
        id_col        : Column containing row identifiers
        method        : "topN" or "mean"
        N             : TopN parameter
        class_cols    : List of class column names (auto-detected if None)

    Returns:
        DataFrame with post-processed probabilities
    """
    df = submission_df.copy()

    if class_cols is None:
        class_cols = [c for c in df.columns if c != id_col]

    # Parse filename from row_id
    df["_file"] = df[id_col].apply(lambda x: "_".join(x.split("_")[:-1]))

    # Apply per-file
    for fname, grp in df.groupby("_file"):
        probs = grp[class_cols].values.astype(np.float32)   # (S, C)

        if method == "topN":
            processed = topn_postprocessing(probs, N=N)
        else:
            processed = mean_postprocessing(probs)

        df.loc[grp.index, class_cols] = processed

    df = df.drop(columns=["_file"])
    return df


# ─── TTA prediction merging (1st place) ──────────────────────────────────

def merge_tta_predictions(
    preds_list: List[np.ndarray],
    method: str = "mean",
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Merge predictions from multiple TTA variants.

    1st place uses ±2.5s delta shifts:
        preds_list = [pred_at_t-2.5s, pred_at_t, pred_at_t+2.5s]

    Args:
        preds_list : List of (N, C) prediction arrays
        method     : "mean" or "weighted_mean"
        weights    : Optional weights for weighted mean

    Returns:
        (N, C) merged predictions
    """
    stacked = np.stack(preds_list, axis=0)    # (n_tta, N, C)

    if method == "weighted_mean" and weights is not None:
        w = np.array(weights, dtype=np.float32)[:, np.newaxis, np.newaxis]
        return (stacked * w).sum(axis=0) / w.sum()

    return stacked.mean(axis=0)


# ─── Padded macro ROC-AUC (competition metric) ───────────────────────────

def padded_auc_score(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    fill_value:  float = 0.5,
    min_pos:     int   = 1,
) -> float:
    """
    Competition metric: macro-averaged ROC-AUC, skipping classes
    with no positive labels (replaced by fill_value=0.5).

    From competition description:
    "macro-averaged ROC-AUC that skips classes which have no true positive labels"

    Validation aggregation from paper Appendix B:
    "predictions on training files were aggregated by taking the maximum
     probability across segments per class."

    Args:
        y_true     : (N, C) binary ground truth
        y_pred     : (N, C) predicted probabilities
        fill_value : AUC for classes with no positives (0.5 = random)
        min_pos    : Minimum positives required to compute AUC

    Returns:
        Scalar padded macro AUC
    """
    from sklearn.metrics import roc_auc_score

    C    = y_true.shape[1]
    aucs = []

    for c in range(C):
        n_pos = int(y_true[:, c].sum())
        if n_pos < min_pos:
            aucs.append(fill_value)
        else:
            try:
                auc = float(roc_auc_score(y_true[:, c], y_pred[:, c]))
                aucs.append(auc)
            except Exception:
                aucs.append(fill_value)

    return float(np.mean(aucs))


def aggregate_validation_predictions(
    clip_probs:   np.ndarray,
    clip_to_file: np.ndarray,
    n_files:      int,
    method:       str = "max",
) -> np.ndarray:
    """
    Aggregate clip-level predictions to file-level.
    Paper Appendix B: "aggregated by taking the maximum probability
    across segments per class."

    Args:
        clip_probs   : (N_clips, C) probabilities
        clip_to_file : (N_clips,) integer file indices
        n_files      : Total number of files
        method       : "max" or "mean"

    Returns:
        (n_files, C) file-level predictions
    """
    C      = clip_probs.shape[1]
    result = np.zeros((n_files, C), dtype=np.float32)

    for file_idx in range(n_files):
        mask = clip_to_file == file_idx
        if not mask.any():
            continue
        clips = clip_probs[mask]
        if method == "max":
            result[file_idx] = clips.max(axis=0)
        else:
            result[file_idx] = clips.mean(axis=0)

    return result
