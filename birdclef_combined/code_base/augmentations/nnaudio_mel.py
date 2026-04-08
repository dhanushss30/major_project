"""
nnaudio_mel.py — GPU Mel-Spectrogram Extraction (nnAudio)
===========================================================
Exact parameters from 2nd place paper Appendix B:

  sample_rate  = 32,000
  n_mels       = 128
  f_min        = 20 Hz
  n_fft        = 2,048
  hop_length   = 512
  normalized   = True
  top_db       = 80
  amin         = 1e-10

Post-extraction: AmplitudeToDb → standardize → scale to [0,1]

nnAudio performs all STFT + mel filterbank operations as a 1-D
convolution on GPU, avoiding the CPU↔GPU bottleneck of librosa.
Falls back to torchaudio if nnAudio is not installed.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# ─── nnAudio import with graceful fallback ─────────────────────────────────

try:
    from nnAudio.features import MelSpectrogram as NNAudioMel
    NNAUDIO_AVAILABLE = True
except ImportError:
    NNAUDIO_AVAILABLE = False

try:
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


# ─── Exact 2nd-place spectrogram config ────────────────────────────────────

DEFAULT_MEL_CONFIG = dict(
    sr          = 32_000,
    n_mels      = 128,
    fmin        = 20,
    fmax        = None,      # Default: sr / 2 = 16,000
    n_fft       = 2_048,
    hop_length  = 512,
    normalized  = True,
    trainable   = False,
)

TOP_DB   = 80.0
AMIN     = 1e-10


# ─── Main extractor class ──────────────────────────────────────────────────

class MelExtractor(nn.Module):
    """
    GPU mel-spectrogram extractor with the exact parameters from
    Sydorskyi & Gonçalves (2025), Appendix B.

    Pipeline:
        audio (B, T) → mel amplitude (B, n_mels, T')
                     → dB via AmplitudeToDb(top_db=80, amin=1e-10)
                     → standardise μ=0, σ=1  (per-sample)
                     → scale to [0, 1]
                     → (B, 1, n_mels, T')   [channel dim for CNN]

    Args:
        sr          : Sample rate (default 32,000)
        n_mels      : Number of mel bins (default 128)
        fmin        : Min frequency (default 20)
        fmax        : Max frequency (default None → sr/2)
        n_fft       : FFT size (default 2,048)
        hop_length  : Hop size (default 512)
        top_db      : Max dB range (default 80)
        amin        : Min amplitude before log (default 1e-10)
        trainable   : Whether mel filters are learnable (default False)
    """

    def __init__(
        self,
        sr: int = 32_000,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: Optional[float] = None,
        n_fft: int = 2_048,
        hop_length: int = 512,
        top_db: float = 80.0,
        amin: float = 1e-10,
        trainable: bool = False,
    ):
        super().__init__()
        self.sr         = sr
        self.n_mels     = n_mels
        self.fmin       = fmin
        self.fmax       = fmax or sr // 2
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.top_db     = top_db
        self.amin       = amin
        self.trainable  = trainable

        self._build_transform()

    def _build_transform(self):
        if NNAUDIO_AVAILABLE:
            self.mel_transform = NNAudioMel(
                sr         = self.sr,
                n_mels     = self.n_mels,
                fmin       = self.fmin,
                fmax       = self.fmax,
                n_fft      = self.n_fft,
                hop_length = self.hop_length,
                trainable_mel   = self.trainable,
                trainable_STFT  = self.trainable,
                verbose    = False,
            )
            self._backend = "nnaudio"
        elif TORCHAUDIO_AVAILABLE:
            self.mel_transform = T.MelSpectrogram(
                sample_rate    = self.sr,
                n_fft          = self.n_fft,
                hop_length     = self.hop_length,
                n_mels         = self.n_mels,
                f_min          = self.fmin,
                f_max          = self.fmax,
                normalized     = True,
                window_fn      = torch.hann_window,
            )
            self._backend = "torchaudio"
        else:
            # Pure numpy fallback — runs on CPU, used only if nothing else available
            self._backend = "librosa"

    def _amplitude_to_db(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel amplitude spectrogram to dB.
        Equivalent to librosa.power_to_db with top_db=80, amin=1e-10.

        Note: mel from nnAudio normalized=True is already power spectrum.
        """
        mel = torch.clamp(mel, min=self.amin)
        mel_db = 10.0 * torch.log10(mel)
        # Apply top_db: clip to [max - top_db, max]
        max_db = mel_db.amax(dim=(-2, -1), keepdim=True)
        mel_db = torch.maximum(mel_db, max_db - self.top_db)
        return mel_db

    def _normalize_to_01(self, mel_db: torch.Tensor) -> torch.Tensor:
        """
        Standardise then scale to [0, 1].
        From the paper: "standardized and scaled to the [0, 1] range."
        """
        # Per-sample standardization
        mean = mel_db.mean(dim=(-2, -1), keepdim=True)
        std  = mel_db.std(dim=(-2, -1), keepdim=True) + 1e-6
        mel_db = (mel_db - mean) / std

        # Scale to [0, 1]
        mn = mel_db.amin(dim=(-2, -1), keepdim=True)
        mx = mel_db.amax(dim=(-2, -1), keepdim=True)
        mel_db = (mel_db - mn) / (mx - mn + 1e-6)
        return mel_db

    def _librosa_fallback(self, audio: torch.Tensor) -> torch.Tensor:
        """CPU numpy fallback using librosa."""
        import librosa
        mels = []
        for wave in audio.cpu().numpy():
            mel = librosa.feature.melspectrogram(
                y          = wave,
                sr         = self.sr,
                n_fft      = self.n_fft,
                hop_length = self.hop_length,
                n_mels     = self.n_mels,
                fmin       = self.fmin,
                fmax       = self.fmax,
            )
            mels.append(mel)
        return torch.tensor(np.stack(mels), dtype=torch.float32, device=audio.device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio : (B, T) or (T,) float32 waveform

        Returns:
            spec  : (B, 1, n_mels, T') float32, values in [0, 1]
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # --- Compute mel amplitude ---
        if self._backend in ("nnaudio", "torchaudio"):
            mel = self.mel_transform(audio)        # (B, n_mels, T')
        else:
            mel = self._librosa_fallback(audio)    # (B, n_mels, T')

        # --- Amplitude → dB ---
        mel_db = self._amplitude_to_db(mel)

        # --- Normalise to [0, 1] ---
        spec = self._normalize_to_01(mel_db)

        # --- Add channel dimension ---
        return spec.unsqueeze(1)                   # (B, 1, n_mels, T')

    def get_output_shape(self, audio_length: int) -> tuple:
        """Return (n_mels, T') for a given input length."""
        T_out = 1 + (audio_length - self.n_fft) // self.hop_length
        return (self.n_mels, T_out)
