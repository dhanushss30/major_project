"""main.py — FastAPI entry point for WildEar Research Lab app."""

from __future__ import annotations

# ── Force torchaudio to use soundfile backend (works on Windows without ffmpeg)
# torchaudio 2.11+ defaults to torchcodec which requires ffmpeg; soundfile is simpler.
import torchaudio
_orig_torchaudio_load = torchaudio.load
def _patched_torchaudio_load(*args, **kwargs):
    if "backend" not in kwargs:
        kwargs["backend"] = "soundfile"
    return _orig_torchaudio_load(*args, **kwargs)
torchaudio.load = _patched_torchaudio_load

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from bird_gate import bird_gate
from inference_service import inference_service
from species_db import species_db
from rag_service import rag_service
from routes import (
    predict_router,
    species_router,
    chat_router,
    dashboard_router,
    audio_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 70)
    print(" WildEar Research Lab API — Starting")
    print("=" * 70)
    print(f" Project root: {config.ROOT_DIR}")
    print(f" Device:       {config.DEVICE}")
    print(f" Train CSV:    {config.TRAIN_CSV}")
    print(f" Ckpt dir:     {config.CKPT_DIR}")
    print()

    species_db.initialize()
    rag_service.initialize()
    inference_service.initialize()
    bird_gate.initialize()

    print()
    print("=" * 70)
    print(" Ready — docs at /docs")
    print("=" * 70)
    yield
    # Shutdown
    print("[main] shutting down")


app = FastAPI(
    title       = "WildEar Research Lab API",
    description = "Backend for the WildEar ensemble bird species classifier",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = config.CORS_ALLOW_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Routers
app.include_router(predict_router)
app.include_router(species_router)
app.include_router(chat_router)
app.include_router(dashboard_router)
app.include_router(audio_router)


@app.get("/")
def root():
    return {
        "service":     "wildear-research-lab",
        "version":     "1.0.0",
        "docs":        "/docs",
        "n_ckpts":     inference_service.n_ckpts,
        "n_species":   len(species_db.all_codes()),
        "ai_enabled":  rag_service.enabled,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = config.HOST,
        port    = config.PORT,
        reload  = False,
    )
