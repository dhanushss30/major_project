"""
backbone.py — CNN Backbone Models
===================================
Backbones used in the combined solution:

2nd place primary:
  eca_nfnet_l0          — NFNet-L0, RAdam lr=1e-3
  tf_efficientnetv2_s   — EfficientNetV2-S (in21k pretrain), AdamW lr=1e-4

1st place ensemble members:
  efficientnet_b3, efficientnet_b4
  efficientnet_b0_ns    — NoisyStudent pretrained
  regnety_016, regnety_008

All loaded from timm, with:
  - global_pool = ""  (we add GeM ourselves)
  - num_classes = 0   (feature extractor only)
  - in_chans = 1      (mono mel spectrogram)

The full model:  backbone → GeM pool → ClassificationHead (512 hidden)
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..nn_blocks.gem_pooling import GeMPooling, ClassificationHead, SEDHead
from .noise_conditioning import NoisyConditionedHead, NoiseProfileExtractor

logger = logging.getLogger(__name__)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.error("timm not installed. Run: pip install timm")


# ─── Backbone registry ────────────────────────────────────────────────────

# Maps config name → (timm model name, feature channels)
BACKBONE_REGISTRY = {
    # 2nd place primary
    "eca_nfnet_l0":            ("eca_nfnet_l0",                    2304),
    "tf_efficientnetv2_s":     ("tf_efficientnetv2_s.in21k",       1280),
    # 1st place
    "efficientnet_b0":         ("efficientnet_b0",                 1280),
    "efficientnet_b3":         ("efficientnet_b3",                 1536),
    "efficientnet_b4":         ("efficientnet_b4",                 1792),
    "efficientnet_b0_ns":      ("tf_efficientnet_b0.ns_jft_in1k",  1280),
    "efficientnet_b3_ns":      ("tf_efficientnet_b3.ns_jft_in1k",  1536),
    "regnety_016":             ("regnety_016",                      888),
    "regnety_008":             ("regnety_008",                      768),
    # Pretraining backbone
    "convnext_small":          ("convnext_small",                   768),
}


def get_feature_dim(backbone_name: str) -> int:
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                         f"Choose from: {list(BACKBONE_REGISTRY.keys())}")
    return BACKBONE_REGISTRY[backbone_name][1]


def create_timm_backbone(
    backbone_name: str,
    pretrained:    bool          = True,
    in_chans:      int           = 1,
) -> nn.Module:
    """Create timm backbone as pure feature extractor."""
    assert TIMM_AVAILABLE, "pip install timm"
    timm_name = BACKBONE_REGISTRY[backbone_name][0]
    model = timm.create_model(
        timm_name,
        pretrained  = pretrained,
        num_classes = 0,
        global_pool = "",       # Remove global pool, we use GeM
        in_chans    = in_chans,
    )
    logger.info(f"Created backbone: {backbone_name} ({timm_name}), "
                f"pretrained={pretrained}, in_chans={in_chans}")
    return model


# ─── Full BirdCLEF Model ──────────────────────────────────────────────────

class BirdCLEFModel(nn.Module):
    """
    Full classification model (2nd place architecture):

        Mel spectrogram (B, 1, n_mels, T)
          → CNN backbone          (B, C, H', W')
          → GeM pooling           (B, C)
          → Dropout(0.25)
          → Linear(C → 512) + ReLU
          → Dropout(0.5)
          → Linear(512 → n_classes)
          → logits (B, n_classes)

    Supports both CNN mode and SED mode:
      - CNN mode: standard clip-level prediction
      - SED mode: frame-level + clip-level via SEDHead
    """

    def __init__(
        self,
        backbone_name:         str           = "eca_nfnet_l0",
        n_classes:             int           = 206,
        pretrained:            bool          = True,
        pretrained_path:       Optional[str] = None,
        in_chans:              int           = 1,
        hidden_dim:            int           = 512,
        dropout1:              float         = 0.25,
        dropout2:              float         = 0.5,
        gem_p:                 float         = 3.0,
        use_sed_head:          bool          = False,
        use_noise_conditioning: bool         = False,   # Novel #5
        noise_dim:             int           = 128,     # Novel #5
        use_multi_res_mel:     bool          = False,   # Novel #8
    ):
        super().__init__()

        self.backbone_name          = backbone_name
        self.n_classes              = n_classes
        self.use_sed_head           = use_sed_head
        self.use_noise_conditioning = use_noise_conditioning

        actual_in_chans = 3 if use_multi_res_mel else in_chans

        # Backbone
        self.backbone = create_timm_backbone(
            backbone_name,
            pretrained = pretrained and pretrained_path is None,
            in_chans   = actual_in_chans,
        )

        # GeM pooling over (H', W') → (C,)
        self.gem = GeMPooling(p=gem_p, trainable=True)

        # Auto-detect actual feature dimension (registry values can be wrong)
        with torch.no_grad():
            dummy = torch.zeros(1, actual_in_chans, 128, 128)
            dummy_feat = self.backbone(dummy)
            if dummy_feat.dim() == 4:
                feat_dim = dummy_feat.shape[1]   # (B, C, H, W) → C
            else:
                feat_dim = dummy_feat.shape[-1]
        logger.info(f"Auto-detected feature dim: {feat_dim}")
        # Exposed so train_module can size DANN/taxonomy/causal modules using
        # the true backbone output dim, not the (sometimes-wrong) registry value.
        self.feat_dim = feat_dim

        # Classification head — Novel #5 replaces standard head when enabled
        if use_sed_head:
            self.head = SEDHead(
                in_features = feat_dim,
                n_classes   = n_classes,
                hidden_dim  = hidden_dim,
                dropout     = dropout1,
            )
        elif use_noise_conditioning:
            # Novel #5: FiLM-conditioned head
            self.head = NoisyConditionedHead(
                in_features = feat_dim,
                n_classes   = n_classes,
                hidden_dim  = hidden_dim,
                noise_dim   = noise_dim,
                dropout1    = dropout1,
                dropout2    = dropout2,
            )
            logger.info(f"NoisyConditionedHead enabled (noise_dim={noise_dim})")
        else:
            self.head = ClassificationHead(
                in_features = feat_dim,
                n_classes   = n_classes,
                hidden_dim  = hidden_dim,
                dropout1    = dropout1,
                dropout2    = dropout2,
            )

        # Load custom pretrained backbone
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        """Load backbone weights from custom checkpoint (strips 'backbone.' prefix)."""
        state = torch.load(path, map_location="cpu")

        # Unwrap nested state dicts
        for key in ("state_dict", "model", "backbone_state_dict"):
            if key in state:
                state = state[key]
                break

        # Try to load backbone only
        backbone_state = {
            k.replace("backbone.", "").replace("model.backbone.", ""): v
            for k, v in state.items()
            if "backbone" in k or "feature" in k
        }

        if not backbone_state:
            backbone_state = state  # Assume entire checkpoint is the backbone

        missing, unexpected = self.backbone.load_state_dict(
            backbone_state, strict=False
        )
        logger.info(f"Loaded backbone from {path}")
        if missing:
            logger.debug(f"  Missing: {missing[:5]}")
        if unexpected:
            logger.debug(f"  Unexpected: {unexpected[:5]}")

    def extract_features(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel : (B, 1, n_mels, T)
        Returns:
            features : (B, feat_dim) or (B, T', feat_dim) for SED
        """
        feat = self.backbone(mel)     # (B, C, H', W')

        if self.use_sed_head:
            # For SED: pool over frequency (H') axis, keep time (W')
            # (B, C, H', W') → mean over H' → (B, C, W') → permute → (B, W', C)
            feat = feat.mean(dim=2)           # (B, C, W')
            feat = feat.permute(0, 2, 1)      # (B, T', C)
        else:
            feat = self.gem(feat)             # (B, C)

        return feat

    def get_embedding(
        self,
        mel:           torch.Tensor,           # (B, 1, n_mels, T)
        noise_profile: Optional[torch.Tensor] = None,  # (B, noise_dim) or None
    ) -> torch.Tensor:
        """
        Returns 512-dim hidden embedding (after first Linear+ReLU, before classifier).
        Used by PrototypicalHead (Novel #4) and build_prototypes.py.

        Works for both ClassificationHead and NoisyConditionedHead.
        """
        feat = self.extract_features(mel)           # (B, feat_dim)
        if self.use_noise_conditioning and hasattr(self.head, "get_embedding"):
            return self.head.get_embedding(feat, noise_profile)  # (B, 512)
        elif hasattr(self.head, "get_embedding"):
            return self.head.get_embedding(feat)    # ClassificationHead
        else:
            raise AttributeError("Head does not expose get_embedding()")

    def forward(
        self,
        mel:           torch.Tensor,           # (B, 1, n_mels, T)
        noise_profile: Optional[torch.Tensor] = None,  # (B, noise_dim) or None
    ):
        """
        Args:
            mel           : (B, 1, n_mels, T)
            noise_profile : (B, noise_dim) — soundscape noise descriptor.
                            Only used when use_noise_conditioning=True.
                            Pass None to use identity FiLM (unconditioned).
        Returns:
            CNN mode: (B, n_classes) logits
            SED mode: dict with clip_logits, frame_logits, attn_weights
        """
        feat = self.extract_features(mel)

        if self.use_sed_head:
            return self.head(feat)
        elif self.use_noise_conditioning:
            return self.head(feat, noise_profile)   # NoisyConditionedHead
        else:
            return self.head(feat)                  # ClassificationHead

    def get_logits(
        self,
        mel:           torch.Tensor,
        noise_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Always returns clip-level logits (B, n_classes)."""
        out = self.forward(mel, noise_profile)
        if isinstance(out, dict):
            return out["clip_logits"]
        return out

    def get_probabilities(
        self,
        mel:           torch.Tensor,
        noise_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns sigmoid probabilities for inference."""
        return torch.sigmoid(self.get_logits(mel, noise_profile))
