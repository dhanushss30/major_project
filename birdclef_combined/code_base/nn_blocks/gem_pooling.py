"""
gem_pooling.py — Generalized Mean Pooling
==========================================
From Radenović et al. "Fine-tuning CNN Image Retrieval with No Human Annotation."
Referenced in the 2nd place paper Appendix B:
"To aggregate CNN embeddings across the frequency dimension, GeM pooling was employed."

The classification head architecture (exact from paper):
  CNN backbone output → GeM pool → 512 hidden (ReLU) → n_classes
  With: dropout 0.25 post-CNN, dropout 0.5 post-hidden
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling.

    GeMPool(x, p) = (1/|x|  Σ x_i^p) ^ (1/p)

    Special cases:
      p = 1   → Average pooling
      p → ∞   → Max pooling
      p ≈ 3   → Typically optimal for fine-grained recognition (learned)

    Args:
        p         : Initial exponent (default 3.0, learnable)
        eps       : Stability epsilon
        trainable : Whether p is a learnable parameter
    """

    def __init__(
        self,
        p: float          = 3.0,
        eps: float        = 1e-6,
        trainable: bool   = True,
    ):
        super().__init__()
        self.eps = eps
        self.p   = nn.Parameter(torch.tensor(float(p)), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)  feature map
        Returns:
            (B, C)  pooled features
        """
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1,
        ).pow(1.0 / self.p).view(x.size(0), -1)

    def extra_repr(self) -> str:
        return f"p={self.p.item():.1f}, eps={self.eps}, trainable={self.p.requires_grad}"


class ClassificationHead(nn.Module):
    """
    Exact classification head from 2nd place paper Appendix B:

        CNN features (C,)
          → Dropout(0.25)
          → Linear(C → 512)
          → ReLU
          → Dropout(0.5)
          → Linear(512 → n_classes)

    Note: No sigmoid here — raw logits for BCE/Focal loss training.
    Sigmoid applied during inference only.
    """

    def __init__(
        self,
        in_features: int,
        n_classes:   int,
        hidden_dim:  int   = 512,
        dropout1:    float = 0.25,    # Post-CNN encoder dropout
        dropout2:    float = 0.5,     # Post-hidden layer dropout
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the 512-dim hidden embedding (after first Linear+ReLU, before classifier).
        Used by PrototypicalHead and build_prototypes.py.

        Args:
            x : (B, in_features)
        Returns:
            (B, hidden_dim)
        """
        # head = [Dropout, Linear, ReLU, Dropout, Linear]
        # indices 0-2 = Dropout → Linear → ReLU
        h = x
        for layer in list(self.head.children())[:3]:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_features) pooled CNN features
        Returns:
            (B, n_classes) logits
        """
        return self.head(x)


class SEDHead(nn.Module):
    """
    Sound Event Detection head — from 1st place approach.
    Provides both clip-level and frame-level predictions.

    Architecture (inspired by Adavanne & Virtanen, 2017):
        Time-distributed features (B, T, D)
          → Frame classifier: Linear(D → n_classes)   [frame logits]
          → Attention net:    Linear(D → n_classes)    [attention weights]
          → Clip prediction:  weighted sum over time

    This allows training on weak (clip-level) labels while producing
    strong (frame-level) predictions at inference for pseudo-label refinement.
    """

    def __init__(
        self,
        in_features:  int,
        n_classes:    int,
        hidden_dim:   int   = 256,
        dropout:      float = 0.3,
    ):
        super().__init__()

        # Project backbone features to hidden dim
        self.proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Per-frame species classifier
        self.frame_cls = nn.Linear(hidden_dim, n_classes)

        # Per-frame attention weights
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        """
        Args:
            x : (B, T, D) time-distributed features

        Returns dict:
            "clip_logits"    : (B, n_classes)    clip-level prediction
            "frame_logits"   : (B, T, n_classes) per-frame logits
            "attn_weights"   : (B, T, n_classes) temporal attention weights
        """
        h = self.proj(x)                              # (B, T, hidden)

        frame_logits = self.frame_cls(h)              # (B, T, n_classes)
        attn_logits  = self.attn(h)                   # (B, T, n_classes)
        attn_weights = torch.softmax(attn_logits, dim=1)  # softmax over time

        clip_logits  = (frame_logits * attn_weights).sum(dim=1)  # (B, n_classes)

        return {
            "clip_logits":  clip_logits,
            "frame_logits": frame_logits,
            "attn_weights": attn_weights,
        }
