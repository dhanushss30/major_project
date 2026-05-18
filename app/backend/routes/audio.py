"""routes/audio.py — Audio visualization endpoints (waveform + spectrogram)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Form

import audio_utils


router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/waveform")
async def waveform(audio: UploadFile = File(...), n_points: int = Form(1024)):
    """Return downsampled waveform for plotting."""
    suffix = Path(audio.filename or "uploaded.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        return audio_utils.waveform_data(tmp_path, n_points=n_points)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode failed: {e}")


@router.post("/spectrogram")
async def spectrogram(
    audio:         UploadFile = File(...),
    start_sec:     float      = Form(0.0),
    duration_sec: float       = Form(5.0),
    as_image:      bool       = Form(False, description="If True, return base64 PNG; else raw array"),
):
    suffix = Path(audio.filename or "uploaded.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        if as_image:
            img_b64 = audio_utils.spectrogram_png_base64(
                tmp_path, start_sec=start_sec, duration_sec=duration_sec,
            )
            return {"image": img_b64, "format": "base64_png"}
        else:
            return audio_utils.spectrogram_array(
                tmp_path, start_sec=start_sec, duration_sec=duration_sec,
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Spectrogram failed: {e}")
