"""
domain_adaptation.py — Gradient Reversal Domain Adaptation (DANN)
==================================================================
NOVEL CONTRIBUTION #1 — Not present in 1st or 2nd place solutions.

Problem: BirdCLEF's hardest challenge is DOMAIN SHIFT.
  - Training data: clean, close-up Xeno-Canto recordings (one species, quiet)
  - Test data:     noisy 1-minute soundscapes (multiple species, far-away, wind)

Current solutions (both winning teams) address this via:
  - Data augmentation (background noise, MixUp)
  - Pseudo-labeling (noisy student)

What they do NOT do: force the backbone to learn features that are
domain-invariant, i.e., features where XC recordings and soundscapes
look identical in embedding space.

Our solution: Domain-Adversarial Neural Networks (DANN)
  Ganin et al. "Domain-Adversarial Training of Neural Networks" (JMLR 2016)

Architecture:
    audio → backbone → features
                           ↓                ↓
                   species head         domain classifier
                   (species loss)       (domain loss, REVERSED gradient)

The gradient reversal layer flips the sign of the gradient flowing
back into the backbone from the domain classifier. This forces the
backbone to make its features MAXIMALLY CONFUSING to the domain
classifier — i.e., domain-invariant.

Training signal:
    total_loss = species_loss + λ(p) * domain_loss
    λ(p) = 2 / (1 + exp(-10p)) - 1   where p = step/total_steps
    (λ ramps from 0→1 during training, stabilising early learning)

At inference: domain classifier is discarded. Only backbone + species head used.

Why this helps: The backbone learns features that are useful for species ID
but independent of whether the audio comes from a close-up XC recording or
a distant soundscape. This directly bridges the domain gap.

Reference implementations:
  - github.com/fungtion/DANN (original PyTorch)
  - github.com/jvanvugt/pytorch-domain-adaptation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ─── Gradient Reversal Layer ──────────────────────────────────────────────

class GradientReversalFunction(Function):
    """
    Custom autograd function that:
      - Forward pass:  x → x          (identity)
      - Backward pass: grad → -λ*grad  (reversal with scaling)

    This is mathematically equivalent to a gradient ascent step
    on the domain classifier loss with respect to the backbone.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lambda_, = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Wraps GradientReversalFunction as a Module so it can be placed
    inside nn.Sequential or used in forward().

    Args:
        lambda_ : Gradient reversal scale (controls adaptation strength).
                  Set dynamically during training via set_lambda().
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


# ─── DANN λ schedule ──────────────────────────────────────────────────────

def compute_dann_lambda(p: float, max_lambda: float = 1.0) -> float:
    """
    Compute DANN adaptation coefficient λ(p).

    Formula from Ganin et al.:
        λ(p) = 2 / (1 + exp(-10p)) - 1

    p = current_step / total_steps  ∈ [0, 1]

    At p=0.0 → λ≈0.0 (no domain adaptation, stable early training)
    At p=0.5 → λ≈0.82 (moderate)
    At p=1.0 → λ≈1.0 (full adaptation)

    Args:
        p          : Training progress ∈ [0, 1]
        max_lambda : Scale factor (default 1.0; use 0.1–0.3 for fine-tuning)
    """
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ─── Domain Classifier ────────────────────────────────────────────────────

class DomainClassifier(nn.Module):
    """
    Binary classifier: XC recording (0) vs Soundscape (1).

    Placed after the gradient reversal layer so that the backbone
    learns to produce domain-CONFUSED features.

    Architecture: features → GRL → FC(512) → BN → ReLU → FC(256) → FC(1)

    Note: Batch Normalization is critical here — it stabilises training
    when gradient reversal makes the signal noisy in early epochs.

    Args:
        in_features : Backbone output dimension (1280 for NFNet/EffNetV2)
        hidden_dim  : Hidden layer size
    """

    def __init__(self, in_features: int, hidden_dim: int = 512):
        super().__init__()

        self.grl = GradientReversalLayer(lambda_=1.0)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_lambda(self, lambda_: float):
        """Update gradient reversal strength. Called each training step."""
        self.grl.set_lambda(lambda_)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (B, in_features) — pooled backbone features
        Returns:
            (B, 1) — domain logits (XC=0, soundscape=1)
        """
        reversed_features = self.grl(features)
        return self.net(reversed_features)


# ─── Domain Adaptation Loss ───────────────────────────────────────────────

class DomainAdaptationLoss(nn.Module):
    """
    Combines domain classification loss for both labeled (XC) and
    unlabeled (soundscape) samples.

    domain_label:
        0 = XC recording (source domain)
        1 = Soundscape   (target domain)

    Loss = BCE(domain_logits_xc,        zeros)
         + BCE(domain_logits_soundscape, ones)

    The backbone receives reversed gradients from both, encouraging
    it to produce features that are indistinguishable across domains.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        domain_logits_xc: torch.Tensor,           # (B_xc, 1)
        domain_logits_soundscape: torch.Tensor,   # (B_sc, 1)
    ) -> torch.Tensor:
        device = domain_logits_xc.device

        labels_xc = torch.zeros(domain_logits_xc.size(0), 1, device=device)
        labels_sc = torch.ones(domain_logits_soundscape.size(0), 1, device=device)

        loss_xc = F.binary_cross_entropy_with_logits(domain_logits_xc, labels_xc)
        loss_sc = F.binary_cross_entropy_with_logits(domain_logits_soundscape, labels_sc)

        return (loss_xc + loss_sc) / 2.0


# ─── Soundscape Audio Buffer (for DANN training) ──────────────────────────

class SoundscapeAudioBuffer:
    """
    Provides random soundscape audio chunks during supervised training
    for domain adaptation — without requiring labels.

    Usage: In each training step, call sample() to get a batch of
    soundscape audio tensors (no labels needed — only used for
    domain classification).

    Caches audio chunks in memory for fast repeated access.
    """

    def __init__(
        self,
        soundscape_dir: str,
        sr: int             = 32_000,
        chunk_duration: float = 5.0,
        max_cache_size: int = 2000,    # Max chunks to keep in memory
    ):
        from pathlib import Path
        import random as _random
        self._random = _random
        self.sr             = sr
        self.chunk_samples  = int(chunk_duration * sr)
        self.max_cache_size = max_cache_size
        self._cache: list   = []

        root = Path(soundscape_dir)
        self._files = []
        for ext in (".ogg", ".mp3", ".wav", ".flac"):
            self._files.extend(sorted(root.rglob(f"*{ext}")))

        if not self._files:
            raise FileNotFoundError(f"No audio files found in {soundscape_dir}")

    def _load_chunk(self, path) -> Optional[torch.Tensor]:
        try:
            import soundfile as sf
            import numpy as np

            info  = sf.info(str(path))
            n_total = int(info.duration * self.sr)
            if n_total < self.chunk_samples:
                return None

            start = self._random.randint(0, n_total - self.chunk_samples)
            audio, sr = sf.read(str(path), start=start,
                                frames=self.chunk_samples,
                                dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) < self.chunk_samples:
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
            return torch.tensor(audio[:self.chunk_samples], dtype=torch.float32)
        except Exception:
            return None

    def _fill_cache(self, n: int = 500):
        paths = self._random.choices(self._files, k=n * 3)
        for p in paths:
            if len(self._cache) >= self.max_cache_size:
                break
            chunk = self._load_chunk(p)
            if chunk is not None:
                self._cache.append(chunk)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            (batch_size, chunk_samples) tensor of soundscape audio
        """
        if len(self._cache) < batch_size * 2:
            self._fill_cache(500)

        indices = self._random.choices(range(len(self._cache)), k=batch_size)
        chunks  = [self._cache[i] for i in indices]
        return torch.stack(chunks, dim=0).to(device)
