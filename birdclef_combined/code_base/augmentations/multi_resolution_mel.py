"""
multi_resolution_mel.py — Multi-Window Spectral Attention (Novel #8)
======================================================================
Computes mel-spectrograms at three different time-frequency resolutions
and fuses them via learned channel attention, providing the backbone
with multi-scale spectral information in a single forward pass.

Motivation: Different taxonomic groups have discriminative features at
different time-frequency resolutions:
  - Insects (17 spp): high-frequency, transient stridulation patterns
    → fine temporal resolution (n_fft=1024, hop=256)
  - Birds (146 spp): mid-frequency, structured calls/songs
    → standard resolution (n_fft=2048, hop=512)
  - Amphibians (34 spp): low-frequency, tonal calls
    → fine frequency resolution (n_fft=4096, hop=1024)

Instead of choosing one resolution, we stack all three as a 3-channel
"image" and let the model learn which resolution is most informative
per species. This naturally leverages ImageNet-pretrained backbones
which expect 3-channel input.

Alternative mode: channel attention dynamically weights the three
resolutions before merging to a single channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from nnAudio.features.mel import MelSpectrogram as nnMelSpec
    NNAUDIO_AVAILABLE = True
except ImportError:
    NNAUDIO_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class MultiResolutionMelExtractor(nn.Module):
    """
    Computes mel spectrograms at 3 resolutions and stacks/fuses them.

    Resolution configs:
        Fine:   n_fft=1024, hop=256  → 125 frames/sec (insects)
        Medium: n_fft=2048, hop=512  → 62.5 frames/sec (birds)
        Coarse: n_fft=4096, hop=1024 → 31.25 frames/sec (amphibians)

    Output modes:
        "stack":     (B, 3, n_mels, T) — 3-channel input for backbone
        "attention": (B, 1, n_mels, T) — attention-weighted single channel

    All three spectrograms are resampled to match the medium resolution's
    time axis for consistent tensor shapes.

    Args:
        sr:         Sample rate (32000)
        n_mels:     Number of mel bands (128)
        fmin:       Min frequency (20 Hz)
        top_db:     Amplitude clipping (80 dB)
        amin:       Floor value (1e-10)
        mode:       "stack" or "attention"
    """

    RESOLUTIONS = {
        "fine":   {"n_fft": 1024, "hop_length": 256},
        "medium": {"n_fft": 2048, "hop_length": 512},
        "coarse": {"n_fft": 4096, "hop_length": 1024},
    }

    def __init__(
        self,
        sr:     int   = 32_000,
        n_mels: int   = 128,
        fmin:   float = 20.0,
        fmax:   Optional[float] = None,
        top_db: float = 80.0,
        amin:   float = 1e-10,
        mode:   str   = "stack",
    ):
        super().__init__()
        self.sr     = sr
        self.n_mels = n_mels
        self.top_db = top_db
        self.amin   = amin
        self.mode   = mode

        self.mel_transforms = nn.ModuleDict()

        for name, params in self.RESOLUTIONS.items():
            if NNAUDIO_AVAILABLE:
                mel = nnMelSpec(
                    sr         = sr,
                    n_fft      = params["n_fft"],
                    hop_length = params["hop_length"],
                    n_mels     = n_mels,
                    fmin       = fmin,
                    fmax       = fmax or sr // 2,
                    trainable_mel = False,
                    trainable_STFT = False,
                )
            elif TORCHAUDIO_AVAILABLE:
                mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate = sr,
                    n_fft       = params["n_fft"],
                    hop_length  = params["hop_length"],
                    n_mels      = n_mels,
                    f_min       = fmin,
                    f_max       = fmax or sr // 2,
                )
            else:
                raise ImportError("Need nnAudio or torchaudio for mel extraction")

            self.mel_transforms[name] = mel

        if mode == "attention":
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(3, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1),
            )

    def _to_db_normalize(self, mel: torch.Tensor) -> torch.Tensor:
        mel_db = 10.0 * torch.log10(mel.clamp(min=self.amin))

        if self.top_db is not None:
            max_val = mel_db.amax(dim=(-2, -1), keepdim=True)
            mel_db  = torch.maximum(mel_db, max_val - self.top_db)

        mean = mel_db.mean(dim=(-2, -1), keepdim=True)
        std  = mel_db.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        mel_db = (mel_db - mean) / std

        mel_db = (mel_db - mel_db.amin(dim=(-2, -1), keepdim=True))
        max_val = mel_db.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        mel_db = mel_db / max_val

        return mel_db

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, T) raw waveform at self.sr

        Returns:
            mode="stack":     (B, 3, n_mels, T') — 3-channel spectrogram
            mode="attention": (B, 1, n_mels, T') — attention-fused single channel
        """
        mels = {}
        target_t = None

        for name in ["fine", "medium", "coarse"]:
            mel = self.mel_transforms[name](audio)

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if mel.dim() == 3:
                mel = mel.unsqueeze(1)

            mel = self._to_db_normalize(mel)

            if name == "medium":
                target_t = mel.shape[-1]
            mels[name] = mel

        for name in mels:
            if mels[name].shape[-1] != target_t:
                mels[name] = F.interpolate(
                    mels[name], size=(self.n_mels, target_t),
                    mode="bilinear", align_corners=False,
                )

        stacked = torch.cat(
            [mels["fine"], mels["medium"], mels["coarse"]], dim=1
        )

        if self.mode == "stack":
            return stacked

        attn_input = stacked
        weights = self.channel_attention(attn_input)
        weights = weights.view(-1, 3, 1, 1)
        fused = (stacked * weights).sum(dim=1, keepdim=True)
        return fused
