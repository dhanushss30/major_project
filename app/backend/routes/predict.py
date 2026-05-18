"""routes/predict.py — Main prediction endpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import config
from inference_service import inference_service
from species_db import species_db


router = APIRouter(prefix="/api", tags=["predict"])


class SpeciesScore(BaseModel):
    code:           str
    common_name:    str | None
    prob:           float
    is_bird:        bool
    well_trained:   bool


class ChunkPrediction(BaseModel):
    chunk_idx:      int
    start_sec:      float
    end_sec:        float
    max_prob:       float
    is_bird:        bool
    top_species:    list[SpeciesScore]


class PredictResponse(BaseModel):
    n_chunks:        int
    n_birds:         int
    n_no_bird:       int
    duration_sec:    float
    no_bird_thresh:  float
    chunks:          list[ChunkPrediction]
    overall_top:     list[SpeciesScore]   # aggregated across all chunks


@router.post("/predict", response_model=PredictResponse)
async def predict(
    audio:              UploadFile = File(..., description="Audio file (WAV/MP3/OGG/FLAC)"),
    top_k:              int        = Form(5),
    no_bird_thresh:     float      = Form(0.10),
    tta_hop:            float      = Form(2.5),
    noise_preprocess:   bool       = Form(True),
    consistency_boost:  float      = Form(0.05),
    only_well_trained:  bool       = Form(False),
):
    """Predict species per 5-sec chunk on uploaded audio.

    only_well_trained=True filters predictions to species with >=50 training samples
    (excludes rare iNat numeric taxa, gives a cleaner demo).
    """
    suffix = Path(audio.filename or "uploaded.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        probs = inference_service.predict(
            audio_path        = tmp_path,
            tta_hop_sec       = tta_hop,
            noise_preprocess  = noise_preprocess,
            consistency_boost = consistency_boost,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    labels = inference_service.labels
    n_chunks, n_classes = probs.shape

    # Mask non-well-trained species if requested
    if only_well_trained:
        well_trained_idxs = set()
        for i, code in enumerate(labels):
            info = species_db.get(code)
            if info and info.is_well_trained and info.is_bird:
                well_trained_idxs.add(i)
        mask = np.zeros(n_classes, dtype=bool)
        for i in well_trained_idxs:
            mask[i] = True
        probs = probs * mask[None, :]

    chunks: list[ChunkPrediction] = []
    n_birds = 0
    agg_probs = probs.mean(axis=0)   # aggregated for overall_top

    for i in range(n_chunks):
        order   = np.argsort(-probs[i])[:top_k]
        max_p   = float(probs[i, order[0]])
        is_bird = max_p >= no_bird_thresh
        if is_bird:
            n_birds += 1
        top_species = []
        for c in order:
            info = species_db.get(labels[c])
            top_species.append(SpeciesScore(
                code         = labels[c],
                common_name  = info.common_name if info else None,
                prob         = float(probs[i, c]),
                is_bird      = bool(info.is_bird) if info else True,
                well_trained = bool(info.is_well_trained) if info else False,
            ))
        chunks.append(ChunkPrediction(
            chunk_idx   = i,
            start_sec   = i * 5.0,
            end_sec     = (i + 1) * 5.0,
            max_prob    = max_p,
            is_bird     = is_bird,
            top_species = top_species,
        ))

    # Overall top species (averaged across chunks)
    overall_order = np.argsort(-agg_probs)[:top_k]
    overall_top = []
    for c in overall_order:
        info = species_db.get(labels[c])
        overall_top.append(SpeciesScore(
            code         = labels[c],
            common_name  = info.common_name if info else None,
            prob         = float(agg_probs[c]),
            is_bird      = bool(info.is_bird) if info else True,
            well_trained = bool(info.is_well_trained) if info else False,
        ))

    return PredictResponse(
        n_chunks       = n_chunks,
        n_birds        = n_birds,
        n_no_bird      = n_chunks - n_birds,
        duration_sec   = n_chunks * 5.0,
        no_bird_thresh = no_bird_thresh,
        chunks         = chunks,
        overall_top    = overall_top,
    )
