"""
focal_bce.py — Loss Functions
===============================
2nd place: Linear combination of BCE + Focal loss (Section 5.2)
1st place: SoftAUC — differentiable pairwise AUC optimization

Paper quote (2nd place):
"optimized via a linear combination of Binary Cross Entropy (BCE) and
 Focal loss"

Paper quote (1st place overview article):
"Since we are evaluated on AUC at BirdCLEF, it makes sense to optimize it
 directly. SoftAUCLoss — implemented class SoftAUCLoss(nn.Module), which
 computes pairwise differences and log-loss, resistant to overfitting,
 but supporting soft labels."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── BCE + Focal combined loss (2nd place) ────────────────────────────────

class BCEFocalLoss(nn.Module):
    """
    Linear combination: α*BCE + (1-α)*FocalLoss

    From paper Section 5.2 baseline:
    "multi-label classification problem, optimized via a linear combination
     of Binary Cross Entropy (BCE) and Focal loss."

    Focal loss focuses on hard examples by down-weighting easy ones.
    Works well with label smoothing for noisy weak labels.
    """

    def __init__(
        self,
        bce_weight:   float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma:  float = 2.0,
        focal_alpha:  float = 0.25,
        reduction:    str   = "mean",
        pos_weight:   Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight   = bce_weight
        self.focal_weight = focal_weight
        self.gamma        = focal_gamma
        self.alpha        = focal_alpha
        self.reduction    = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def _bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight = self.pos_weight,
            reduction  = self.reduction,
        )

    def _focal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob  = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t   = prob * targets + (1 - prob) * (1 - targets)
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss  = alpha * (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B, C) labels in [0, 1] (may be soft)
        Returns:
            scalar loss
        """
        loss = torch.tensor(0.0, device=logits.device)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self._bce(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self._focal(logits, targets)
        return loss


# ─── SoftAUC loss (1st place innovation) ─────────────────────────────────

class SoftAUCLoss(nn.Module):
    """
    Differentiable pairwise AUC loss — 1st place solution.

    AUC = P(score(positive) > score(negative)) over all pos/neg pairs.

    Approximation via sigmoid:
        L = E_{(i+, i-)} [ -log σ(γ * (s_{i+} - s_{i-} - margin)) ]

    Supports soft labels: a "positive" is any sample with label > pos_thresh.
    This is critical for pseudo-label training where labels are probabilities.

    Resistance to overfitting: doesn't push scores to 0/1 extremes,
    only requires ordering (positive scores > negative scores).

    Args:
        gamma       : Sigmoid slope (higher = sharper boundary)
        margin      : Hinge margin (0 for pure AUC)
        pos_thresh  : Soft label threshold to define "positive"
        neg_thresh  : Soft label threshold to define "negative"
        n_pairs     : Max pairs to sample per class per batch
        reduction   : "mean" or "sum"
    """

    def __init__(
        self,
        gamma:      float = 1.0,
        margin:     float = 0.0,
        pos_thresh: float = 0.5,
        neg_thresh: float = 0.1,
        n_pairs:    int   = 64,
        reduction:  str   = "mean",
    ):
        super().__init__()
        self.gamma      = gamma
        self.margin     = margin
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.n_pairs    = n_pairs
        self.reduction  = reduction

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B, C) soft labels in [0, 1]
        Returns:
            Scalar AUC surrogate loss
        """
        probs   = torch.sigmoid(logits)
        B, C    = probs.shape
        device  = logits.device

        total_loss = torch.zeros(1, device=device)
        n_terms    = 0

        for c in range(C):
            p_c   = probs[:, c]      # (B,)
            t_c   = targets[:, c]    # (B,)

            pos_idx = (t_c >= self.pos_thresh).nonzero(as_tuple=True)[0]
            neg_idx = (t_c <= self.neg_thresh).nonzero(as_tuple=True)[0]

            n_pos, n_neg = len(pos_idx), len(neg_idx)
            if n_pos == 0 or n_neg == 0:
                continue

            n_pairs = min(self.n_pairs, n_pos * n_neg)
            pi = torch.randint(0, n_pos, (n_pairs,), device=device)
            ni = torch.randint(0, n_neg, (n_pairs,), device=device)

            s_pos = p_c[pos_idx[pi]]
            s_neg = p_c[neg_idx[ni]]

            # Confidence-weighted: weight by label confidence of each element
            pos_conf = t_c[pos_idx[pi]].detach()
            neg_conf = (1 - t_c[neg_idx[ni]]).detach()
            weights  = pos_conf * neg_conf

            diff = s_pos - s_neg - self.margin
            pair_loss = -torch.log(torch.sigmoid(self.gamma * diff) + 1e-8)
            pair_loss = (pair_loss * weights).sum()

            total_loss = total_loss + pair_loss
            n_terms   += n_pairs

        if n_terms == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        if self.reduction == "mean":
            return (total_loss / n_terms).squeeze()
        return total_loss.squeeze()


# ─── Combined loss ────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Final combined loss:
        w_focal * BCEFocal + w_auc * SoftAUC

    Stage 1 (clean labels): w_auc = 0 (BCE+Focal only)
    Stage 2 (pseudo):       w_auc = 0.3 (optimize AUC directly)
    """

    def __init__(
        self,
        bce_weight:   float = 0.5,
        focal_weight: float = 0.5,
        auc_weight:   float = 0.0,
        focal_gamma:  float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.total_bce_focal = bce_weight + focal_weight
        self.auc_weight      = auc_weight

        self.bce_focal = BCEFocalLoss(
            bce_weight   = bce_weight   / max(bce_weight + focal_weight, 1e-6),
            focal_weight = focal_weight / max(bce_weight + focal_weight, 1e-6),
            focal_gamma  = focal_gamma,
        )

        self.soft_auc = SoftAUCLoss() if auc_weight > 0 else None

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Apply label smoothing to targets before loss
        # (smoothing already applied in dataset, this is a no-op if done there)
        loss = self.total_bce_focal * self.bce_focal(logits, targets)

        if self.soft_auc is not None and self.auc_weight > 0:
            loss = loss + self.auc_weight * self.soft_auc(logits, targets)

        return loss
