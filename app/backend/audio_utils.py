"""audio_utils.py — Mel-spectrogram + waveform image generation for the UI."""

from __future__ import annotations

import io
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio


def _load_audio_mono_via_soundfile(path: str | Path) -> tuple[torch.Tensor, int]:
    """Bypass torchaudio.load (which needs torchcodec on 2.11+) by reading
    directly via libsndfile / soundfile. Returns (waveform, sample_rate).
    Handles WAV/FLAC/OGG natively; MP3 needs ffmpeg or pre-conversion."""
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    return wav, sr


SR        = 32_000
N_FFT     = 2_048
HOP_LEN   = 512
N_MELS    = 128
FMIN      = 20.0
TOP_DB    = 80.0


def _load_audio_mono(path: str | Path) -> torch.Tensor:
    # Bypass torchaudio.load (torchcodec dependency on 2.11+) → use soundfile
    wav, sr_in = _load_audio_mono_via_soundfile(path)
    if sr_in != SR:
        # torchaudio.functional.resample only needs torch tensors; no decoder
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr_in, SR).squeeze(0)
    return wav


def waveform_data(path: str | Path, n_points: int = 1024) -> dict:
    """Return downsampled waveform for plotting in the frontend.

    Returns:
        {
            "duration_sec": float,
            "samples": List[float],  # downsampled to n_points
            "sr": 32000,
        }
    """
    wav = _load_audio_mono(path).numpy()
    duration = len(wav) / SR
    if len(wav) > n_points:
        # Decimate
        idx = np.linspace(0, len(wav) - 1, n_points).astype(int)
        samples = wav[idx]
    else:
        samples = wav
    # Normalize to [-1, 1] for plotting
    peak = max(abs(samples.max()), abs(samples.min()), 1e-6)
    samples = samples / peak
    return {
        "duration_sec": float(duration),
        "samples":      samples.astype(float).tolist(),
        "sr":           SR,
    }


def spectrogram_png_base64(
    path: str | Path,
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
    figsize: tuple = (10, 4),
    cmap: str = "magma",
) -> str:
    """Generate mel-spectrogram image as base64 PNG for embedding in JSON.

    Returns a 'data:image/png;base64,...' string ready for <img src="...">.
    """
    wav = _load_audio_mono(path)
    s = int(start_sec * SR)
    e = int((start_sec + duration_sec) * SR)
    seg = wav[s:e]
    if len(seg) < 2048:
        # Pad short audio
        seg = torch.nn.functional.pad(seg, (0, 2048 - len(seg)))

    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate = SR,
        n_fft       = N_FFT,
        hop_length  = HOP_LEN,
        f_min       = FMIN,
        f_max       = SR // 2,
        n_mels      = N_MELS,
        power       = 2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB)

    mel = mel_extractor(seg.unsqueeze(0))   # (1, n_mels, T')
    mel = amp_to_db(mel).squeeze(0).numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mel, aspect="auto", origin="lower", cmap=cmap,
        extent=[0, duration_sec, FMIN, SR // 2],
    )
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("frequency (Hz)", fontsize=10)
    ax.set_yscale("symlog", linthresh=1_000)
    fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    ax.set_title(f"Mel-Spectrogram  ({start_sec:.1f}–{start_sec + duration_sec:.1f}s)",
                 fontsize=11, fontweight="bold")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def spectrogram_array(
    path: str | Path,
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
) -> dict:
    """Return raw mel-spec array for frontend-rendered heatmap (Plotly).

    Returns:
        {
            "z": List[List[float]],  # (n_mels, n_frames)
            "n_mels": 128,
            "n_frames": int,
            "duration_sec": float,
            "fmin": 20.0,
            "fmax": 16000.0,
        }
    """
    wav = _load_audio_mono(path)
    s = int(start_sec * SR)
    e = int((start_sec + duration_sec) * SR)
    seg = wav[s:e]
    if len(seg) < 2048:
        seg = torch.nn.functional.pad(seg, (0, 2048 - len(seg)))

    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate = SR,
        n_fft       = N_FFT,
        hop_length  = HOP_LEN,
        f_min       = FMIN,
        f_max       = SR // 2,
        n_mels      = N_MELS,
        power       = 2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB)

    mel = mel_extractor(seg.unsqueeze(0))
    mel = amp_to_db(mel).squeeze(0).numpy()
    return {
        "z":            mel.astype(float).tolist(),
        "n_mels":       N_MELS,
        "n_frames":     int(mel.shape[1]),
        "duration_sec": float(duration_sec),
        "fmin":         float(FMIN),
        "fmax":         float(SR // 2),
    }
