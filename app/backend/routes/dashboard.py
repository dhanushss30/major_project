"""routes/dashboard.py — Model metrics endpoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter

import config
from species_db import species_db
from inference_service import inference_service


router = APIRouter(prefix="/api/metrics", tags=["dashboard"])


@router.get("/summary")
def metrics_summary():
    """Headline numbers + ckpt info."""
    info = []
    for c in inference_service.models:
        info.append({
            "name":      c.path.name,
            "val_auc":   c.val_auc,
            "has_film":  c.has_film,
            "epoch":     c.epoch,
        })

    summary = {
        "ensemble_val_auc":   0.8246,
        "n_ckpts":            inference_service.n_ckpts,
        "n_classes":          206,
        "n_well_trained":     sum(1 for s in species_db.list_all() if s.get("is_well_trained")),
        "n_birds":            sum(1 for s in species_db.list_all() if s.get("is_bird")),
        "device":             config.DEVICE,
        "ckpts":              info,
        "single_aucs": {
            "v3_fold0_clean":  0.7756,
            "v3_fold0_film":   0.7841,
            "esc50_bg_peak":   0.6814,
            "new_fold1":       0.6976,
        },
    }
    return summary


@router.get("/per-class-auc")
def per_class_auc():
    """Per-class AUC distribution for the histogram chart."""
    csv = Path(config.PER_CLASS_AUC_CSV)
    if not csv.exists():
        return {"available": False, "rows": []}

    df = pd.read_csv(csv)
    df = df.dropna(subset=["auc"])
    rows = []
    for _, row in df.iterrows():
        code = str(row["label"])
        info = species_db.to_dict(code)
        rows.append({
            "code":          code,
            "common_name":   info.get("common_name") if info else None,
            "is_bird":       info.get("is_bird") if info else True,
            "auc":           float(row["auc"]),
            "n_train":       int(info.get("n_train_samples", 0)) if info else 0,
            "n_val_pos":     int(row.get("n_pos", 0)),
        })
    rows.sort(key=lambda x: -x["auc"])

    summary = {
        "available":  True,
        "mean_auc":   float(df["auc"].mean()),
        "median_auc": float(df["auc"].median()),
        "n_valid":    len(df),
        "n_total":    206,
        "rows":       rows,
    }
    return summary


@router.get("/training-trajectory")
def training_trajectory():
    """Parse the fold_1 training log to extract per-epoch val/auc."""
    import re
    log = Path(config.CKPT_DIR) / "eca_stage1_fold1.log"
    if not log.exists():
        return {"available": False, "rows": []}

    rows = []
    pattern = re.compile(r"Epoch (\d+): val/auc = ([\d\.]+)")
    seen_epochs = {}
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                auc = float(m.group(2))
                # Keep the second occurrence per epoch (after-train, not sanity)
                seen_epochs[epoch] = auc
    rows = [{"epoch": e, "val_auc": a} for e, a in sorted(seen_epochs.items())]
    return {"available": True, "rows": rows}
