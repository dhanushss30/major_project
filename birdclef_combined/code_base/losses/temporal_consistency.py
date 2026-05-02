"""
temporal_consistency.py — Temporal Consistency Regularization (Novel #6)
=========================================================================
Enforces smooth, ecologically coherent predictions across adjacent
5-second segments of the same soundscape recording during pseudo-label
training.

Motivation: Birds vocalize in temporal patterns — if a species is
detected in one segment, it is likely present in adjacent segments.
Standard pseudo-labeling treats each 5-sec chunk independently, which
introduces noisy, temporally incoherent predictions. TCR penalizes
sharp prediction discontinuities between adjacent chunks from the
same file via symmetric KL-divergence.

This is applied ONLY during pseudo-label / noisy-student stages,
where the model processes soundscape-derived data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    """
    Symmetric KL-divergence between predictions on adjacent 5-sec chunks.

    Given a batch of audio segments, pairs that originate from the same
    soundscape file and are temporally adjacent (|t1 - t2| <= gap) are
    penalized for prediction disagreement.

    The loss is weighted by prediction confidence: high-confidence pairs
    contribute more, preventing the regularizer from pulling confident
    predictions toward uncertain neighbors.

    Args:
        weight:     Loss weight (λ_tcr in the paper)
        max_gap:    Maximum temporal gap (seconds) to consider "adjacent"
        temperature: Softmax temperature for sharpening predictions
    """

    def __init__(
        self,
        weight:      float = 0.1,
        max_gap:     float = 5.0,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.weight      = weight
        self.max_gap     = max_gap
        self.temperature = temperature

    def forward(
        self,
        logits:     torch.Tensor,
        filenames:  list,
        start_secs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:     (B, C) raw logits from the model
            filenames:  list of B filename strings
            start_secs: (B,) tensor of segment start times in seconds

        Returns:
            Scalar TCR loss (0 if no adjacent pairs found)
        """
        B, C = logits.shape
        device = logits.device

        if B < 2:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        p = F.softmax(logits / self.temperature, dim=-1)
        log_p = F.log_softmax(logits / self.temperature, dim=-1)

        total_loss = torch.zeros(1, device=device)
        n_pairs = 0

        file_groups = {}
        for i, fn in enumerate(filenames):
            if fn not in file_groups:
                file_groups[fn] = []
            file_groups[fn].append(i)

        for fn, indices in file_groups.items():
            if len(indices) < 2:
                continue

            for i_pos, i in enumerate(indices):
                for j in indices[i_pos + 1:]:
                    gap = abs(start_secs[i].item() - start_secs[j].item())
                    if gap > self.max_gap:
                        continue

                    kl_ij = F.kl_div(log_p[j], p[i], reduction="sum")
                    kl_ji = F.kl_div(log_p[i], p[j], reduction="sum")
                    sym_kl = 0.5 * (kl_ij + kl_ji)

                    conf_i = logits[i].sigmoid().max()
                    conf_j = logits[j].sigmoid().max()
                    conf_weight = (conf_i * conf_j).detach()

                    total_loss = total_loss + conf_weight * sym_kl
                    n_pairs += 1

        if n_pairs == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        return (self.weight * total_loss / n_pairs).squeeze()


class TemporalConsistencyBatchMixer:
    """
    Utility to construct mini-batches that contain adjacent segments
    from the same soundscape file, enabling TCR computation.

    During pseudo-label training, instead of random sampling, this
    sampler groups consecutive segments from the same file into batches.
    """

    def __init__(
        self,
        pseudo_df,
        group_size: int = 4,
    ):
        self.groups = []
        grouped = pseudo_df.groupby("filename")
        for fn, group in grouped:
            sorted_group = group.sort_values("start_sec")
            indices = sorted_group.index.tolist()
            for i in range(0, len(indices), group_size):
                chunk = indices[i:i + group_size]
                if len(chunk) >= 2:
                    self.groups.append(chunk)

    def sample_group(self):
        import random
        return random.choice(self.groups) if self.groups else []
