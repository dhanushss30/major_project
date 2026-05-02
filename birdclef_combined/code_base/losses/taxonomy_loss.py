"""
taxonomy_loss.py — Taxonomy-Aware Hierarchical Loss (Novel #7)
================================================================
BirdCLEF+ 2025 expanded beyond birds to include insects, amphibians,
and mammals (Table 1: 146 Aves, 34 Amphibia, 17 Insecta, 9 Mammalia).

No existing solution exploits this multi-taxonomic structure.

This module adds:
1. An auxiliary taxonomic classification head predicting the 4 super-classes
2. A hierarchical consistency penalty that ensures species-level predictions
   are consistent with taxonomic-level predictions
3. A taxonomic confusion penalty that increases the cost of confusing
   species across taxonomic boundaries (e.g., confusing a bird with an insect
   is worse than confusing two bird species)

The auxiliary loss acts as a structured prior, particularly helping
rare species that have few training samples but share acoustic properties
with their taxonomic siblings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


TAXONOMY_MAP = {
    "Aves":     0,
    "Amphibia": 1,
    "Insecta":  2,
    "Mammalia": 3,
}

N_TAXA = len(TAXONOMY_MAP)

SPECIES_TO_TAXON = {
    "gocfly1": 0, "devwoo1": 0, "yebfly2": 0, "blbwre1": 0, "grttin1": 0,
    "pavpig2": 0, "rufgro1": 0, "whhsaw1": 0, "rutfly6": 0, "gretin1": 0,
    "grbcam1": 0, "blbgra1": 0, "littin1": 0, "rugdov": 0, "blbant1": 0,
    "whbant1": 0, "blchaw1": 0, "yercac1": 0, "bobher1": 0, "anhing": 0,
    "bkcdon": 0, "bawwre1": 0, "reshaw1": 0, "trokin1": 0, "smbani": 0,
}


class TaxonomyMapper:
    """
    Maps species indices to taxonomic group indices.
    Built from the competition metadata.
    """

    def __init__(
        self,
        label_to_idx:   Dict[str, int],
        species_to_taxon: Optional[Dict[str, int]] = None,
        default_taxon:  int = 0,
    ):
        self.n_classes = len(label_to_idx)
        self.n_taxa    = N_TAXA

        self.species_to_group = torch.zeros(self.n_classes, dtype=torch.long)

        if species_to_taxon:
            for species, idx in label_to_idx.items():
                taxon = species_to_taxon.get(species, default_taxon)
                self.species_to_group[idx] = taxon

        self.group_masks = torch.zeros(self.n_taxa, self.n_classes, dtype=torch.float32)
        for c in range(self.n_classes):
            g = self.species_to_group[c].item()
            self.group_masks[g, c] = 1.0

    def get_taxon_labels(self, species_labels: torch.Tensor) -> torch.Tensor:
        """
        Convert species-level labels (B, n_classes) to taxon-level labels (B, n_taxa).
        A taxon is positive if ANY species in that taxon is positive.
        """
        masks = self.group_masks.to(species_labels.device)
        taxon_labels = torch.matmul(species_labels, masks.t())
        return taxon_labels.clamp(0.0, 1.0)

    def to(self, device):
        self.species_to_group = self.species_to_group.to(device)
        self.group_masks = self.group_masks.to(device)
        return self


class TaxonomyHead(nn.Module):
    """
    Lightweight auxiliary head for taxonomic group classification.
    Shares the backbone feature extractor but has its own classifier.
    """

    def __init__(self, in_features: int, n_taxa: int = N_TAXA, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_taxa),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class TaxonomyAwareLoss(nn.Module):
    """
    Combined taxonomic loss with three components:

    1. Auxiliary classification loss: BCE on taxon-level predictions
       Forces the model to first identify WHAT kind of animal it is.

    2. Hierarchical consistency loss: penalizes species predictions
       that are inconsistent with taxon predictions. If the model says
       "not a bird" but predicts a bird species, that's penalized.

    3. Cross-taxon confusion penalty: increases the cost of confusing
       species across taxonomic boundaries vs. within the same group.

    Args:
        aux_weight:         Weight of auxiliary taxon classification loss
        consistency_weight: Weight of hierarchical consistency penalty
        confusion_weight:   Weight of cross-taxon confusion penalty
    """

    def __init__(
        self,
        aux_weight:         float = 0.2,
        consistency_weight: float = 0.1,
        confusion_weight:   float = 0.05,
    ):
        super().__init__()
        self.aux_weight         = aux_weight
        self.consistency_weight = consistency_weight
        self.confusion_weight   = confusion_weight

    def forward(
        self,
        species_logits: torch.Tensor,
        taxon_logits:   torch.Tensor,
        species_labels: torch.Tensor,
        taxon_labels:   torch.Tensor,
        mapper:         TaxonomyMapper,
    ) -> torch.Tensor:
        device = species_logits.device
        loss = torch.zeros(1, device=device)

        if self.aux_weight > 0:
            aux_loss = F.binary_cross_entropy_with_logits(
                taxon_logits, taxon_labels, reduction="mean"
            )
            loss = loss + self.aux_weight * aux_loss

        if self.consistency_weight > 0:
            species_probs = species_logits.sigmoid()
            taxon_probs   = taxon_logits.sigmoid()

            group_masks = mapper.group_masks.to(device)

            for g in range(mapper.n_taxa):
                mask = group_masks[g]
                species_in_group = (species_probs * mask.unsqueeze(0)).max(dim=1).values
                taxon_g = taxon_probs[:, g]
                inconsistency = F.relu(species_in_group - taxon_g)
                loss = loss + self.consistency_weight * inconsistency.mean()

        if self.confusion_weight > 0:
            species_probs = species_logits.sigmoid()
            group_assignments = mapper.species_to_group.to(device)

            B, C = species_probs.shape
            top2_vals, top2_idx = species_probs.topk(2, dim=1)

            pred1_group = group_assignments[top2_idx[:, 0]]
            pred2_group = group_assignments[top2_idx[:, 1]]
            cross_taxon = (pred1_group != pred2_group).float()

            confusion = cross_taxon * top2_vals[:, 1]
            loss = loss + self.confusion_weight * confusion.mean()

        return loss.squeeze()
