"""fastapi_backend.py — REST API for BirdCLEF 2025 5-second predictions.

Endpoints:
  GET  /             — health check
  GET  /info         — model + ensemble info
  GET  /species      — full 206-class list
  POST /predict      — upload audio, get per-5s predictions

Usage on server:
    uvicorn scripts.fastapi_backend:app --host 0.0.0.0 --port 8081

Usage from client (curl):
    curl -X POST http://localhost:8081/predict \\
         -F "audio=@/path/to/file.wav" \\
         -F "top_k=3" -F "no_bird_thresh=0.30"
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_inference import ensemble_predict, load_ckpt
from predict_cli import DEFAULT_CKPTS, DEFAULT_TRAIN_CSV, load_label_names


# ============================================================================
# Globals (loaded once at startup)
# ============================================================================
app = FastAPI(
    title       = "BirdCLEF 2025 Species API",
    description = "Per-5-second bird species prediction with open-set rejection",
    version     = "1.0.0",
)

_models = None
_labels = None


@app.on_event("startup")
def startup_event():
    global _models, _labels
    available = [c for c in DEFAULT_CKPTS if Path(c).exists()]
    if not available:
        raise RuntimeError("No checkpoints found. Train models first.")
    _models = [load_ckpt(c, device="cuda") for c in available]
    _labels = load_label_names(DEFAULT_TRAIN_CSV)
    print(f"[startup] loaded {len(_models)} ckpts, {len(_labels)} labels")


# ============================================================================
# Response schemas
# ============================================================================
class SpeciesScore(BaseModel):
    label: str
    prob:  float


class ChunkPrediction(BaseModel):
    chunk_idx:    int
    start_sec:    float
    end_sec:      float
    max_prob:     float
    is_bird:      bool
    top_species:  list[SpeciesScore]


class PredictResponse(BaseModel):
    n_chunks:        int
    n_birds:         int
    n_no_bird:       int
    no_bird_thresh:  float
    chunks:          list[ChunkPrediction]


class InfoResponse(BaseModel):
    backbone:        str
    n_classes:       int
    sr:              int
    chunk_sec:       float
    n_ckpts:         int
    ensemble_ckpts:  list[str]


# ============================================================================
# Endpoints
# ============================================================================
@app.get("/")
def root():
    return {"status": "ok", "service": "birdclef-2025-api"}


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        backbone       = "eca_nfnet_l0",
        n_classes      = 206,
        sr             = 32_000,
        chunk_sec      = 5.0,
        n_ckpts        = len(_models),
        ensemble_ckpts = [c.path.name for c in _models],
    )


@app.get("/species")
def species():
    return {"labels": _labels, "n": len(_labels)}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    audio:              UploadFile = File(..., description="Audio file (WAV/MP3/OGG/FLAC)"),
    top_k:              int        = Form(3),
    no_bird_thresh:     float      = Form(0.30),
    tta_hop:            float      = Form(2.5),
    noise_preprocess:   bool       = Form(True),
    consistency_boost:  float      = Form(0.05),
):
    if _models is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    suffix = Path(audio.filename or "uploaded.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        probs = ensemble_predict(
            audio_path             = tmp_path,
            ckpts                  = _models,
            device                 = "cuda",
            tta_hop_sec            = tta_hop,
            noise_preprocess       = noise_preprocess,
            consistency_vote_boost = consistency_boost,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    n_chunks, _ = probs.shape
    chunks = []
    n_birds = 0
    for i in range(n_chunks):
        order   = np.argsort(-probs[i])[:top_k]
        max_p   = float(probs[i, order[0]])
        is_bird = max_p >= no_bird_thresh
        if is_bird:
            n_birds += 1
        chunks.append(ChunkPrediction(
            chunk_idx   = i,
            start_sec   = i * 5.0,
            end_sec     = (i + 1) * 5.0,
            max_prob    = max_p,
            is_bird     = is_bird,
            top_species = [
                SpeciesScore(label=_labels[c], prob=float(probs[i, c]))
                for c in order
            ],
        ))

    return PredictResponse(
        n_chunks       = n_chunks,
        n_birds        = n_birds,
        n_no_bird      = n_chunks - n_birds,
        no_bird_thresh = no_bird_thresh,
        chunks         = chunks,
    )
