"""
cooccurrence_graph.py — Species Co-occurrence Graph Refinement
==============================================================
NOVEL CONTRIBUTION #2 — Not present in 1st or 2nd place solutions.

Problem: Standard classifiers treat all 206 species independently.
In reality, species co-occur in predictable ecological communities.

For example, in Colombia's cloud forests:
  - If Rufous-naped Wren (XC recordings) is detected, other cloud
    forest species are more likely to also be present.
  - If a species that never overlaps with another in Colombia is
    predicted highly, suppressing the other is ecologically valid.

Neither the 1st nor 2nd place solution modelled species interactions.

Our solution: Species Co-occurrence Graph Convolutional Network (GCN)
  - Nodes: 206 species
  - Edge weights: co-occurrence frequency in soundscape predictions
  - Node features: raw classifier probabilities
  - GCN output: refined probabilities (takes ecology into account)

This is inspired by:
  - Chen et al. "Multi-Label Image Recognition with Graph
    Convolutional Networks" (CVPR 2019) — SSGRL
  - ML-GCN: graph-based multi-label classification
  But applied to BIOACOUSTIC SPECIES CO-OCCURRENCE, which is novel.

Two-stage approach:
  Stage 1: Build co-occurrence matrix from OOF soundscape predictions
            (after supervised training, before pseudo-labeling)
  Stage 2: Train a small 2-layer GCN to refine predictions
            (supervised by OOF AUC loss, 10-20 epochs, very fast)

At inference:
  raw_probs → GCN(raw_probs, A_normalized) → refined_probs

Expected gain: +0.005–0.012 AUC
Computational cost: Negligible (GCN is tiny, <1M parameters)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─── Co-occurrence Matrix Builder ────────────────────────────────────────

def build_cooccurrence_matrix(
    predictions: np.ndarray,       # (N_files × N_clips, N_classes) probabilities
    file_ids:    np.ndarray,        # (N_files × N_clips,) file index per clip
    threshold:   float  = 0.3,     # Confidence threshold for "present"
    smoothing:   float  = 1.0,     # Laplace smoothing (avoids zero edges)
    normalize:   bool   = True,    # Row-normalize adjacency matrix
) -> torch.Tensor:
    """
    Build species co-occurrence adjacency matrix from soundscape predictions.

    For each soundscape file, compute which species are predicted as
    "present" (max probability across clips > threshold). Two species
    that co-occur in the same soundscape file get an edge.

    A[i, j] = P(species j present | species i present)
            = count(files where both i and j present) /
              count(files where i present)

    Args:
        predictions : (N_clips, N_classes) — model probabilities per clip
        file_ids    : (N_clips,) — which file each clip belongs to
        threshold   : Minimum probability to consider a species "present"
        smoothing   : Laplace smoothing constant
        normalize   : If True, row-normalize A (GCN convention)

    Returns:
        A_norm : (N_classes, N_classes) normalized adjacency matrix
    """
    n_classes  = predictions.shape[1]
    n_files    = int(file_ids.max()) + 1

    # For each file, take max probability across all clips
    file_probs = np.zeros((n_files, n_classes), dtype=np.float32)
    for clip_idx, file_idx in enumerate(file_ids):
        file_probs[file_idx] = np.maximum(file_probs[file_idx],
                                          predictions[clip_idx])

    # Binary presence matrix: (N_files, N_classes)
    presence = (file_probs >= threshold).astype(np.float32)

    # Co-occurrence matrix: A[i, j] = count files where both i and j present
    # Using matrix multiplication: presence.T @ presence = (N_classes, N_classes)
    cooc = presence.T @ presence    # (N_classes, N_classes)

    # Add Laplace smoothing
    cooc += smoothing

    # Normalize: A[i, j] = P(j | i) = cooc[i,j] / sum_k cooc[i,k]
    row_sum = cooc.sum(axis=1, keepdims=True)
    A = cooc / (row_sum + 1e-8)

    # Symmetric normalization for GCN: D^(-1/2) * A * D^(-1/2)
    if normalize:
        deg    = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(deg) + 1e-8))
        A = D_inv_sqrt @ A @ D_inv_sqrt

    logger.info(f"Co-occurrence matrix built: {n_classes}×{n_classes}, "
                f"sparsity={np.mean(A < 1e-4):.2%}, "
                f"n_files={n_files}")

    return torch.tensor(A, dtype=torch.float32)


# ─── Graph Convolutional Layer ─────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """
    Standard Graph Convolutional layer (Kipf & Welling 2017):
        H' = σ(A_hat * H * W)

    where:
        A_hat = pre-computed normalized adjacency matrix (fixed)
        H     = node feature matrix (species probabilities or hidden)
        W     = learnable weight matrix
        σ     = ReLU or identity

    Args:
        in_features  : Input feature dim per node
        out_features : Output feature dim per node
        bias         : Use bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H : (N_classes, in_features)  — node features
            A : (N_classes, N_classes)    — normalized adjacency (fixed)
        Returns:
            (N_classes, out_features)
        """
        support = H @ self.weight      # (N_classes, out_features)
        output  = A @ support          # aggregate from neighbors
        if self.bias is not None:
            output = output + self.bias
        return output


# ─── Species Co-occurrence GCN ────────────────────────────────────────────

class SpeciesCooccurrenceGCN(nn.Module):
    """
    2-layer GCN that refines species probability predictions using
    the ecological co-occurrence graph.

    Architecture:
        raw_probs (N_classes,) as initial node features
          → GCN Layer 1: (N_classes, 1) → (N_classes, hidden_dim) + ReLU
          → Dropout
          → GCN Layer 2: (N_classes, hidden_dim) → (N_classes, 1)
          → Sigmoid → refined_probs (N_classes,)

    This treats each species as a node with feature = its raw probability,
    and refines each probability by aggregating information from
    ecologically co-occurring species.

    Training: Minimize AUC surrogate loss (or BCE) on OOF predictions.
    Inference: raw_probs → GCN → refined_probs (no change to main model)
    """

    def __init__(
        self,
        n_classes:  int   = 206,
        hidden_dim: int   = 128,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.gcn1    = GraphConvLayer(1, hidden_dim)
        self.gcn2    = GraphConvLayer(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

        # Residual connection: output = GCN(x) + α * x
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        # Fixed adjacency (set via register_buffer, not trainable)
        self.register_buffer("adj", torch.eye(n_classes))

    def set_adjacency(self, A: torch.Tensor):
        """Set the co-occurrence adjacency matrix (called once after building)."""
        self.adj = A.to(self.adj.device)
        logger.info(f"GCN adjacency matrix set: {A.shape}")

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs : (B, N_classes) — raw sigmoid probabilities in [0, 1]
        Returns:
            (B, N_classes) — refined probabilities in [0, 1]
        """
        B, C    = probs.shape
        A       = self.adj              # (C, C)
        refined = []

        # Process each sample (adjacency is shared across the batch)
        for i in range(B):
            # Node features: (C, 1) — one feature per species = its probability
            H = probs[i].unsqueeze(1)               # (C, 1)

            # Layer 1
            H = self.gcn1(H, A)                     # (C, hidden)
            H = F.relu(H)
            H = self.dropout(H)

            # Layer 2
            H = self.gcn2(H, A)                     # (C, 1)
            delta = H.squeeze(1)                     # (C,)

            # Residual: refined = sigmoid(logit(raw) + δ)
            # More stable than directly adding delta to probability
            logit_raw     = torch.logit(probs[i].clamp(1e-6, 1 - 1e-6))
            logit_refined = logit_raw + self.residual_weight * delta
            refined.append(torch.sigmoid(logit_refined))

        return torch.stack(refined, dim=0)      # (B, C)


# ─── GCN Training ─────────────────────────────────────────────────────────

def train_cooccurrence_gcn(
    gcn:         SpeciesCooccurrenceGCN,
    oof_probs:   np.ndarray,        # (N_clips, N_classes) OOF predictions
    oof_targets: np.ndarray,        # (N_clips, N_classes) ground truth
    n_epochs:    int    = 30,
    lr:          float  = 1e-3,
    batch_size:  int    = 512,
    device:      str    = "cuda",
) -> SpeciesCooccurrenceGCN:
    """
    Train the GCN on out-of-fold predictions to minimise BCE loss.

    Args:
        gcn         : Initialized SpeciesCooccurrenceGCN with adjacency set
        oof_probs   : (N_clips, N_classes) — raw ensemble probabilities (OOF)
        oof_targets : (N_clips, N_classes) — true labels
        n_epochs    : Training epochs (30 is enough — GCN is tiny)
        lr          : Learning rate
        batch_size  : Mini-batch size for clips
        device      : Training device

    Returns:
        Trained GCN
    """
    from ..utils.postprocessing import padded_auc_score

    gcn = gcn.to(device)
    gcn.train()

    optimizer = torch.optim.Adam(gcn.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    probs_t   = torch.tensor(oof_probs,   dtype=torch.float32)
    targets_t = torch.tensor(oof_targets, dtype=torch.float32)
    N         = len(probs_t)

    best_auc      = 0.0
    best_state    = None

    for epoch in range(n_epochs):
        gcn.train()
        perm   = torch.randperm(N)
        losses = []

        for start in range(0, N, batch_size):
            idx   = perm[start: start + batch_size]
            x     = probs_t[idx].to(device)
            y     = targets_t[idx].to(device)

            refined = gcn(x)
            loss    = F.binary_cross_entropy(refined, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gcn.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        # Validation AUC every 5 epochs
        if (epoch + 1) % 5 == 0:
            gcn.eval()
            with torch.no_grad():
                all_refined = []
                for start in range(0, N, batch_size):
                    x       = probs_t[start: start + batch_size].to(device)
                    refined = gcn(x)
                    all_refined.append(refined.cpu().numpy())
                all_refined = np.concatenate(all_refined, axis=0)

            auc = padded_auc_score(oof_targets, all_refined)
            raw_auc = padded_auc_score(oof_targets, oof_probs)
            logger.info(f"GCN Epoch {epoch+1}/{n_epochs}: "
                       f"loss={np.mean(losses):.4f}, "
                       f"AUC={auc:.4f} (raw={raw_auc:.4f}, "
                       f"delta={auc-raw_auc:+.4f})")

            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.clone() for k, v in gcn.state_dict().items()}

    if best_state is not None:
        gcn.load_state_dict(best_state)
        logger.info(f"GCN training complete. Best AUC: {best_auc:.4f}")

    return gcn


# ─── Save / Load GCN ──────────────────────────────────────────────────────

def save_gcn(gcn: SpeciesCooccurrenceGCN, path: str):
    torch.save({
        "state_dict": gcn.state_dict(),
        "n_classes":  gcn.n_classes,
    }, path)
    logger.info(f"GCN saved to {path}")


def load_gcn(path: str, device: str = "cpu") -> SpeciesCooccurrenceGCN:
    ckpt = torch.load(path, map_location=device)
    gcn  = SpeciesCooccurrenceGCN(n_classes=ckpt["n_classes"])
    gcn.load_state_dict(ckpt["state_dict"])
    gcn.eval()
    logger.info(f"GCN loaded from {path}")
    return gcn
