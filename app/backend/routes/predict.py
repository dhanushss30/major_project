"""routes/predict.py - Main prediction endpoint.

Two-stage decision per 5-sec chunk:

  1. AST gate (bird_gate) — pretrained AudioSet classifier, 527 classes.
     Computes bird_score (sum over 22 bird-related AudioSet classes) and
     identifies the top non-bird category (Speech, Music, Vehicle, ...).
     This is the OPEN-SET layer: it answers "is this even a bird sound?"
     since our 206-class species softmax has no notion of "not a bird".

  2. Species head — the 4-ckpt 206-class ensemble. Only consulted when the
     gate says the chunk is plausibly a bird.

Rule: a chunk is labeled non-bird ONLY when the gate is confidently non-bird
AND its top non-bird category is in a hard-rejection list (Speech, Music,
Vehicle, Sine wave, Static, Silence, ...). Otherwise the species top-K is
returned. This dispenses with the manual no-bird-threshold the user used to
have to tune.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import config
from bird_gate import bird_gate, HARD_NON_BIRD_LABELS
from inference_service import inference_service
from species_db import species_db


router = APIRouter(prefix="/api", tags=["predict"])


# ─── Gate decision thresholds ───────────────────────────────────────────
# Derived empirically (see smoke tests in bird_gate.py).
# - bird_score on real bird recordings: ~0.1–0.95 (varies per chunk).
# - bird_score on speech / music / noise / dogs / synthetic: < 0.01.
# - top_non_bird_score on hard distractors (Speech, Music, ...): > 0.15 typically.
GATE_BIRD_HARD_FLOOR    = 0.02   # bird_score below this → non-bird regardless of label
GATE_BIRD_CEILING       = 0.10   # bird_score below this AND a hard label wins → non-bird
GATE_NONBIRD_FLOOR      = 0.15   # min confidence of a hard-non-bird label to reject

# Fallback: only when AST is uncertain (gate doesn't fire), apply a very low
# safety threshold on the species max-prob so we still reject obviously
# off-distribution chunks the gate missed.
SPECIES_SAFETY_THRESH   = 0.02


class SpeciesScore(BaseModel):
    code:           str
    common_name:    str | None
    prob:           float
    is_bird:        bool
    well_trained:   bool


class ChunkPrediction(BaseModel):
    chunk_idx:           int
    start_sec:           float
    end_sec:             float
    max_prob:            float
    is_bird:             bool
    top_species:         list[SpeciesScore]
    # Gate fields — added so the UI can show *what* the non-bird sound is
    gate_bird_score:     float
    gate_top_label:      str
    non_bird_label:      str | None = None     # set when is_bird=False
    non_bird_score:      float | None = None


class PredictResponse(BaseModel):
    n_chunks:        int
    n_birds:         int
    n_no_bird:       int
    duration_sec:    float
    chunks:          list[ChunkPrediction]
    overall_top:     list[SpeciesScore]


@router.post("/predict", response_model=PredictResponse)
async def predict(
    audio:              UploadFile = File(..., description="Audio file (WAV/MP3/OGG/FLAC)"),
    top_k:              int        = Form(5),
    tta_hop:            float      = Form(2.5),
    noise_preprocess:   bool       = Form(True),
    consistency_boost:  float      = Form(0.05),
    only_well_trained:  bool       = Form(False),
):
    """Per-5-sec species prediction with AST-based non-bird rejection.

    The gate runs first per chunk; only chunks that survive go through the
    species classifier. The legacy `no_bird_thresh` parameter is gone — the
    gate handles open-set rejection automatically.
    """
    suffix = Path(audio.filename or "uploaded.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    # ── Species inference (slow path) ──────────────────────────────────
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

    # ── AST gate (fast path, ~0.6s / chunk on CPU) ─────────────────────
    try:
        gate_scores = bird_gate.score_file(tmp_path, n_chunks=n_chunks)
    except Exception as e:
        # Soft-fail: if the gate breaks, fall back to species-only with a
        # conservative species threshold rather than 500ing.
        print(f"[predict] bird_gate failed: {e}; falling back to species-only")
        gate_scores = [{
            "bird_score": 1.0, "top_non_bird_label": "",
            "top_non_bird_score": 0.0, "best_hard_label": "",
            "best_hard_score": 0.0, "top_label": "", "top_score": 0.0,
        } for _ in range(n_chunks)]

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
    agg_probs = probs.mean(axis=0)

    for i in range(n_chunks):
        order   = np.argsort(-probs[i])[:top_k]
        max_p   = float(probs[i, order[0]])
        g       = gate_scores[i]

        # ── Decision tree ─────────────────────────────────────────────
        # Tier 1: bird_score is essentially zero → confident non-bird,
        #   regardless of which non-bird category wins. This catches dog
        #   barks, livestock, etc. where the top class is the generic
        #   "Animal" parent (not in HARD list) but bird-score is ~0.001.
        gate_very_confident = g["bird_score"] < GATE_BIRD_HARD_FLOOR
        # Tier 2: bird_score is low and a recognized hard distractor wins.
        gate_hard_match = (
            g["bird_score"]         <  GATE_BIRD_CEILING
            and g["top_non_bird_label"] in HARD_NON_BIRD_LABELS
            and g["top_non_bird_score"] >= GATE_NONBIRD_FLOOR
        )
        species_too_weak = max_p < SPECIES_SAFETY_THRESH
        is_bird = not (gate_very_confident or gate_hard_match or species_too_weak)

        if is_bird:
            n_birds += 1

        # Display label for non-bird: prefer a specific hard label
        # (e.g. "Dog") over a generic parent like "Animal" when both fire.
        display_label = g["top_non_bird_label"]
        display_score = float(g["top_non_bird_score"])
        if (not is_bird) and g.get("best_hard_score", 0) >= 0.05:
            display_label = g["best_hard_label"]
            display_score = float(g["best_hard_score"])

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
            chunk_idx        = i,
            start_sec        = i * 5.0,
            end_sec          = (i + 1) * 5.0,
            max_prob         = max_p,
            is_bird          = is_bird,
            top_species      = top_species,
            gate_bird_score  = float(g["bird_score"]),
            gate_top_label   = g["top_label"],
            non_bird_label   = (display_label if not is_bird else None),
            non_bird_score   = (display_score if not is_bird else None),
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
        chunks         = chunks,
        overall_top    = overall_top,
    )
