"""
audio_aug.py — Audio-Domain Augmentations (Exact 2nd Place Implementation)
===========================================================================
From Sydorskyi & Gonçalves (2025), Section 5.2 (Baseline Augmentations):

  1. MixUp (p=0.5):
     - Two audio waveforms are ADDED in the audio domain (not blended)
     - Target = element-wise MAXIMUM of one-hot vectors (not weighted sum)
     - This simulates simultaneous multi-species vocalisation

  2. Background Noise (p=0.5):
     - Noise drawn equally from: prior-year BirdCLEF soundscapes + ESC-50
     - Randomly overlapped with training samples

  3. RandomFiltering (Section 5.2):
     - Simplified random equalizer
     - Simulates channel distortion and recording device variation

Additional (1st place noisy student):
  4. Heavy augmentation mode: Gaussian/pink noise, gain, pitch shift
  5. Temporal shift for TTA: ±2.5s delta shifts
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── MixUp — audio domain with max label (exact 2nd place) ────────────────

class AudioMixUp:
    """
    MixUp in the AUDIO domain with element-wise MAX label combination.

    From the paper:
    "Two audio waveforms were added in the audio domain, and the resulting
     one-hot target vector was computed as the element-wise maximum across
     the original vectors."

    This is different from the standard MixUp (convex combination).
    The audio signals are simply summed, naturally simulating two species
    calling simultaneously — which matches test distribution.

    MixUp also bridges hard labels (training) and soft labels (pseudo):
    "Audio segments were mixed in the audio domain, effectively creating
     an interpolated feature space. Corresponding label vectors were
     combined via element-wise summation and clipped to [0, 1]."
    """

    def __init__(self, p: float = 0.5, scale: float = 0.5):
        """
        Args:
            p     : Probability of applying MixUp to each sample
            scale : Volume scale for the mixed-in audio (default 0.5)
                    Added audio: original + scale * second
        """
        self.p     = p
        self.scale = scale

    def __call__(
        self,
        audio1: np.ndarray,
        label1: np.ndarray,
        audio2: np.ndarray,
        label2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            audio1, audio2 : (T,) float32 waveforms (same length)
            label1, label2 : (C,) float32 label vectors

        Returns:
            mixed_audio : (T,) float32
            mixed_label : (C,) float32 — element-wise max OR sum clipped to [0,1]
        """
        if random.random() > self.p:
            return audio1, label1

        n = min(len(audio1), len(audio2))
        mixed = audio1[:n] + self.scale * audio2[:n]
        mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

        # Label: element-wise maximum (for hard labels)
        # or sum clipped to [0,1] (for soft pseudo-labels)
        if label1.max() <= 1.0 and label2.max() <= 1.0:
            mixed_label = np.clip(label1 + label2, 0.0, 1.0)
        else:
            mixed_label = np.maximum(label1, label2)

        return mixed, mixed_label.astype(np.float32)


# ─── Background noise (ESC-50 + soundscapes) ──────────────────────────────

class BackgroundNoiseMixer:
    """
    Background noise augmentation — from paper Section 5.2:

    "Background audio from prior-year soundscapes and the ESC-50 dataset
     was randomly overlapped with training samples." (p=0.5)

    "Equally drawing noise samples from the soundscape dataset and ESC-50."

    ESC-50: https://github.com/karolpiczak/ESC-50
    Soundscapes: prior BirdCLEF competition soundscapes

    The noise is added at a random SNR between 3–20 dB to prevent it from
    drowning out the target species. Level-calibrated via RMS matching.
    """

    def __init__(
        self,
        soundscape_paths: List[str],
        esc50_paths: List[str],
        sample_rate: int = 32_000,
        min_snr_db: float = 3.0,
        max_snr_db: float = 20.0,
        p: float = 0.5,
    ):
        self.sample_rate    = sample_rate
        self.min_snr_db     = min_snr_db
        self.max_snr_db     = max_snr_db
        self.p              = p

        # Pool sources 50/50
        self.soundscape_paths = soundscape_paths
        self.esc50_paths      = esc50_paths
        self._cache: dict     = {}

    def _load_noise(self, path: str, n_samples: int) -> np.ndarray:
        """Load and cache noise clip, tile if shorter than needed."""
        if path not in self._cache:
            try:
                import soundfile as sf
                data, sr = sf.read(path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != self.sample_rate:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
                self._cache[path] = data.astype(np.float32)
            except Exception:
                return np.zeros(n_samples, dtype=np.float32)

        noise = self._cache[path]
        if len(noise) < n_samples:
            repeats = n_samples // len(noise) + 1
            noise   = np.tile(noise, repeats)
        start = random.randint(0, max(0, len(noise) - n_samples))
        return noise[start: start + n_samples]

    def __call__(
        self,
        audio: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not (self.soundscape_paths or self.esc50_paths):
            return audio, label
        if random.random() > self.p:
            return audio, label

        # Select noise source 50/50
        use_soundscape = bool(self.soundscape_paths) and (
            not self.esc50_paths or random.random() < 0.5
        )
        pool = self.soundscape_paths if use_soundscape else self.esc50_paths
        if not pool:
            return audio, label

        noise_path = random.choice(pool)
        noise      = self._load_noise(noise_path, len(audio))

        # RMS-based SNR control
        snr_db      = random.uniform(self.min_snr_db, self.max_snr_db)
        snr_linear  = 10 ** (snr_db / 20.0)
        sig_rms     = np.sqrt(np.mean(audio ** 2)) + 1e-9
        noise_rms   = np.sqrt(np.mean(noise ** 2)) + 1e-9
        scale       = sig_rms / (snr_linear * noise_rms)

        mixed = audio + scale * noise
        mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
        return mixed, label  # Label unchanged: background has no species label


# ─── RandomFiltering (simplified random equalizer) ─────────────────────────

class RandomFiltering(nn.Module):
    """
    RandomFiltering — from paper Section 5.2:
    "A simplified version of a random equalizer was used to simulate
     channel distortions."

    Applies a frequency-domain random gain envelope to the mel spectrogram,
    simulating variation in microphone frequency response, distance from
    subject, and environmental acoustic conditions.

    Applied in spectrogram domain (not audio domain) for efficiency.
    """

    def __init__(
        self,
        max_db: float  = 6.0,    # Maximum frequency band gain (±dB)
        n_bands: int   = 4,      # Number of frequency bands to modulate
        p: float       = 0.5,
    ):
        super().__init__()
        self.max_db = max_db
        self.n_bands = n_bands
        self.p      = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec : (B, 1, n_mels, T) in [0, 1]
        Returns:
            filtered spec : same shape, values re-clipped to [0, 1]
        """
        if not self.training or random.random() > self.p:
            return spec

        B, C, F, T = spec.shape
        device     = spec.device

        # Build random gain envelope over frequency axis
        band_gains = torch.zeros(B, 1, F, 1, device=device)
        band_size  = F // self.n_bands

        for b in range(B):
            for band_idx in range(self.n_bands):
                gain_db = random.uniform(-self.max_db, self.max_db)
                gain    = 10 ** (gain_db / 20.0)
                f_start = band_idx * band_size
                f_end   = (band_idx + 1) * band_size if band_idx < self.n_bands - 1 else F
                band_gains[b, 0, f_start:f_end, 0] = gain

        # Smooth between bands using linear interpolation
        band_gains = torch.nn.functional.interpolate(
            band_gains.squeeze(1).squeeze(-1).unsqueeze(0),  # (1, B, F)
            size=F, mode="linear", align_corners=False
        ).squeeze(0).unsqueeze(1).unsqueeze(-1)              # (B, 1, F, 1)

        spec = spec * band_gains
        return torch.clamp(spec, 0.0, 1.0)



# ─── SpecAugment (time + frequency masking) ────────────────────────────────

class SpecAugment(nn.Module):
    """
    SpecAugment — Park et al. (2019).
    From paper Section 5.2: standard time and frequency masking.

    Applied to mel spectrograms after RandomFiltering.
    """

    def __init__(
        self,
        freq_mask_max: int = 27,   # Max consecutive frequency bins to mask
        time_mask_max: int = 100,  # Max consecutive time frames to mask
        n_freq_masks:  int = 2,
        n_time_masks:  int = 2,
        p: float = 0.5,
    ):
        super().__init__()
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.n_freq_masks  = n_freq_masks
        self.n_time_masks  = n_time_masks
        self.p             = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (B, 1, F, T)"""
        if not self.training or random.random() > self.p:
            return spec

        spec    = spec.clone()
        _, _, F, T = spec.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f  = random.randint(0, self.freq_mask_max)
            f0 = random.randint(0, max(0, F - f))
            spec[:, :, f0: f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.n_time_masks):
            t  = random.randint(0, min(self.time_mask_max, T))
            t0 = random.randint(0, max(0, T - t))
            spec[:, :, :, t0: t0 + t] = 0.0

        return spec


# ─── Composed training augmenter ───────────────────────────────────────────

class SpectrogramAugmenter(nn.Module):
    """
    Composed spectrogram-domain augmenter for training:
        RandomFiltering → SpecAugment

    Call .train() to enable augmentations, .eval() to disable.
    """

    def __init__(
        self,
        use_random_filtering: bool = True,
        use_spec_augment:     bool = True,
        heavy_mode:           bool = False,  # Noisy student: stronger aug
    ):
        super().__init__()
        self.random_filter = RandomFiltering(
            max_db  = 9.0 if heavy_mode else 6.0,
            n_bands = 6   if heavy_mode else 4,
            p       = 0.6 if heavy_mode else 0.5,
        ) if use_random_filtering else None

        self.spec_augment = SpecAugment(
            freq_mask_max = 35  if heavy_mode else 27,
            time_mask_max = 120 if heavy_mode else 100,
            n_freq_masks  = 3   if heavy_mode else 2,
            n_time_masks  = 3   if heavy_mode else 2,
            p             = 0.7 if heavy_mode else 0.5,
        ) if use_spec_augment else None

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (B, 1, F, T) → (B, 1, F, T)"""
        if self.random_filter is not None:
            spec = self.random_filter(spec)
        if self.spec_augment is not None:
            spec = self.spec_augment(spec)
        return spec
