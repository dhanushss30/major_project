"""
train_module.py — PyTorch Lightning Training Module
=====================================================
Exact training setup from paper Appendix B:

  EfficientNetV2-S: AdamW, lr=1e-4, ε=1e-8, β=(0.9,0.999)
  NFNet-L0:         RAdam, lr=1e-3
  Schedule:         Cosine down to 1e-6, NO warmup
  Batch size:       64
  Epochs:           50

MixUp applied in audio domain during collate (50% probability).
SpecAugment + RandomFiltering applied in spectrogram domain.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import lightning as L
    from lightning.pytorch.utilities.types import STEP_OUTPUT
except ImportError:
    import pytorch_lightning as L

from .backbone import BirdCLEFModel
from .domain_adaptation import (
    DomainClassifier, DomainAdaptationLoss,
    SoundscapeAudioBuffer, compute_dann_lambda,
)
from .prototypical_head import PrototypicalHead, PrototypicalLoss
from .noise_conditioning import NoiseProfileExtractor
from ..augmentations.nnaudio_mel import MelExtractor
from ..augmentations.multi_resolution_mel import MultiResolutionMelExtractor
from ..augmentations.audio_aug import AudioMixUp, BackgroundNoiseMixer, SpectrogramAugmenter
from ..losses.focal_bce import CombinedLoss
from ..losses.temporal_consistency import TemporalConsistencyLoss
from ..losses.taxonomy_loss import (
    TaxonomyMapper, TaxonomyHead, TaxonomyAwareLoss, N_TAXA,
)
from ..utils.postprocessing import padded_auc_score

logger = logging.getLogger(__name__)


class BirdCLEFModule(L.LightningModule):
    """
    Full training module with exact 2nd place hyperparameters.

    Supports three stages:
      "supervised"    — clean labels, BCE+Focal only
      "pseudo"        — pseudo labels added, SoftAUC enabled
      "noisy_student" — heavy augmentation, SoftAUC

    The MixUp augmentation is applied in the forward pass:
    audio1 + audio2 → mixed_audio, max(label1, label2) → mixed_label
    (not in the dataset to allow batch-level pairing)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Build model
        self.model = BirdCLEFModel(
            backbone_name          = cfg["backbone"],
            n_classes              = cfg.get("n_classes", 206),
            pretrained             = cfg.get("pretrained", True),
            pretrained_path        = cfg.get("pretrained_path", None),
            hidden_dim             = cfg.get("hidden_dim", 512),
            dropout1               = cfg.get("dropout1", 0.25),
            dropout2               = cfg.get("dropout2", 0.5),
            gem_p                  = cfg.get("gem_p", 3.0),
            use_sed_head           = cfg.get("use_sed_head", False),
            use_noise_conditioning = cfg.get("use_noise_conditioning", False),  # Novel #5
            noise_dim              = cfg.get("noise_dim", 128),
            use_multi_res_mel      = cfg.get("use_multi_res_mel", False),      # Novel #8
        )

        # ── Mel extractor — standard or multi-resolution (Novel #8) ──────
        self.use_multi_res_mel = cfg.get("use_multi_res_mel", False)
        if self.use_multi_res_mel:
            self.mel = MultiResolutionMelExtractor(
                sr     = cfg.get("sr", 32_000),
                n_mels = cfg.get("n_mels", 128),
                fmin   = cfg.get("fmin", 20.0),
                fmax   = cfg.get("fmax", None),
                top_db = cfg.get("top_db", 80.0),
                amin   = cfg.get("amin", 1e-10),
                mode   = cfg.get("multi_res_mode", "stack"),
            )
            logger.info("Multi-Resolution Mel: ENABLED "
                       f"(mode={cfg.get('multi_res_mode', 'stack')})")
        else:
            self.mel = MelExtractor(
                sr         = cfg.get("sr", 32_000),
                n_mels     = cfg.get("n_mels", 128),
                fmin       = cfg.get("fmin", 20.0),
                fmax       = cfg.get("fmax", None),
                n_fft      = cfg.get("n_fft", 2_048),
                hop_length = cfg.get("hop_length", 512),
                top_db     = cfg.get("top_db", 80.0),
                amin       = cfg.get("amin", 1e-10),
            )

        # Audio-domain MixUp (50% probability, paper §5.2)
        self.mixup = AudioMixUp(
            p     = cfg.get("mixup_p", 0.5),
            scale = cfg.get("mixup_scale", 0.5),
        )

        # Background noise augmentation (paper §5.2, p=0.5)
        bg_soundscape = cfg.get("bg_soundscape_paths", [])
        bg_esc50      = cfg.get("bg_esc50_paths", [])
        if bg_soundscape or bg_esc50:
            self.bg_noise = BackgroundNoiseMixer(
                soundscape_paths = bg_soundscape,
                esc50_paths      = bg_esc50,
                sample_rate      = cfg.get("sr", 32_000),
                p                = 0.5,
            )
        else:
            self.bg_noise = None

        # Spectrogram-domain augmentations
        heavy = cfg.get("stage", "supervised") == "noisy_student"
        self.spec_aug = SpectrogramAugmenter(
            use_random_filtering = True,
            use_spec_augment     = True,
            heavy_mode           = heavy,
        )

        # Loss
        self.loss_fn = CombinedLoss(
            bce_weight    = cfg.get("bce_weight",   0.5),
            focal_weight  = cfg.get("focal_weight", 0.5),
            auc_weight    = cfg.get("auc_weight",   0.0),
            focal_gamma   = cfg.get("focal_gamma",  2.0),
        )

        # ── NOVEL: DANN Domain Adaptation (Contribution #1) ───────────────
        # Enabled when "soundscape_root" is set and "use_dann" is True
        self.use_dann      = cfg.get("use_dann", False)
        self.dann_lambda   = cfg.get("dann_lambda_max", 0.3)
        self._dann_buffer  = None
        self._domain_clf   = None
        self._domain_loss  = None

        if self.use_dann and cfg.get("soundscape_root"):
            # Auto-detect feature dim from the backbone registry
            from .backbone import BACKBONE_REGISTRY
            feat_dim = BACKBONE_REGISTRY.get(cfg["backbone"], (None, 1280))[1]

            self._domain_clf  = DomainClassifier(
                in_features = feat_dim,
                hidden_dim  = cfg.get("dann_hidden_dim", 512),
            )
            self._domain_loss = DomainAdaptationLoss()

            try:
                self._dann_buffer = SoundscapeAudioBuffer(
                    soundscape_dir = cfg["soundscape_root"],
                    sr             = cfg.get("sr", 32_000),
                    chunk_duration = cfg.get("chunk_duration", 5.0),
                )
                logger.info("DANN domain adaptation: ENABLED "
                           f"(lambda_max={self.dann_lambda})")
            except FileNotFoundError as e:
                logger.warning(f"DANN disabled — soundscape dir error: {e}")
                self.use_dann     = False
                self._domain_clf  = None
                self._domain_loss = None
        # ─────────────────────────────────────────────────────────────────

        # ── NOVEL #4: Prototypical Head for Rare Species ──────────────────
        # Enabled when "use_prototypical" is True.
        # Requires class_counts_path pointing to a saved numpy array of shape (C,)
        # Built automatically by main_train.py from the dataset after fold split.
        self.use_prototypical  = cfg.get("use_prototypical", False)
        self._proto_head       = None
        self._proto_loss_fn    = None

        if self.use_prototypical:
            hidden_dim = cfg.get("hidden_dim", 512)
            n_classes  = cfg.get("n_classes", 206)
            self._proto_head    = PrototypicalHead(
                embedding_dim = hidden_dim,
                n_classes     = n_classes,
                temperature   = cfg.get("proto_temperature", 0.05),
            )
            self._proto_loss_fn = PrototypicalLoss(
                margin         = cfg.get("proto_margin",    0.3),
                rare_threshold = cfg.get("proto_rare_thr",  0.2),
                weight         = cfg.get("proto_loss_weight", 0.3),
            )
            logger.info("Prototypical head: ENABLED "
                       f"(temperature={cfg.get('proto_temperature', 0.05)}, "
                       f"loss_weight={cfg.get('proto_loss_weight', 0.3)})")
        # ─────────────────────────────────────────────────────────────────

        # ── NOVEL #5: Noise-Conditioned Feature Adaptation ────────────────
        # Enabled when "use_noise_conditioning" is True.
        # NoiseProfileExtractor runs on soundscape audio from the DANN buffer.
        # If DANN buffer is absent, zeros are used (identity = no conditioning).
        self.use_noise_conditioning = cfg.get("use_noise_conditioning", False)
        self._noise_extractor       = None

        if self.use_noise_conditioning:
            self._noise_extractor = NoiseProfileExtractor(
                n_mels     = cfg.get("n_mels",    128),
                output_dim = cfg.get("noise_dim", 128),
            )
            logger.info(
                f"Noise conditioning: ENABLED (noise_dim={cfg.get('noise_dim', 128)})"
            )
        # ─────────────────────────────────────────────────────────────────

        # ── NOVEL #6: Temporal Consistency Regularization ──────────────────
        self.use_tcr = cfg.get("use_tcr", False)
        self._tcr_loss = None

        if self.use_tcr:
            self._tcr_loss = TemporalConsistencyLoss(
                weight      = cfg.get("tcr_weight", 0.1),
                max_gap     = cfg.get("tcr_max_gap", 5.0),
                temperature = cfg.get("tcr_temperature", 2.0),
            )
            logger.info(f"Temporal Consistency Regularization: ENABLED "
                       f"(weight={cfg.get('tcr_weight', 0.1)})")
        # ─────────────────────────────────────────────────────────────────

        # ── NOVEL #7: Taxonomy-Aware Hierarchical Loss ───────────────────
        self.use_taxonomy = cfg.get("use_taxonomy", False)
        self._taxonomy_head = None
        self._taxonomy_loss = None
        self._taxonomy_mapper = None

        if self.use_taxonomy:
            from .backbone import BACKBONE_REGISTRY
            feat_dim = BACKBONE_REGISTRY.get(cfg["backbone"], (None, 1280))[1]

            self._taxonomy_head = TaxonomyHead(
                in_features = feat_dim,
                n_taxa      = N_TAXA,
                hidden_dim  = cfg.get("taxonomy_hidden_dim", 128),
            )
            self._taxonomy_loss = TaxonomyAwareLoss(
                aux_weight         = cfg.get("taxonomy_aux_weight", 0.2),
                consistency_weight = cfg.get("taxonomy_consistency_weight", 0.1),
                confusion_weight   = cfg.get("taxonomy_confusion_weight", 0.05),
            )
            logger.info("Taxonomy-Aware Loss: ENABLED "
                       f"(aux={cfg.get('taxonomy_aux_weight', 0.2)}, "
                       f"consistency={cfg.get('taxonomy_consistency_weight', 0.1)})")
        # ─────────────────────────────────────────────────────────────────

        # EMA
        self.use_ema    = cfg.get("use_ema", True)
        self.ema_decay  = cfg.get("ema_decay", 0.999)
        self._ema_model = None

        # Validation cache
        self._val_probs:   List[np.ndarray] = []
        self._val_targets: List[np.ndarray] = []

    # ── Taxonomy mapper initialisation ──────────────────────────────────

    def set_taxonomy_mapper(self, label_to_idx: dict, species_to_taxon: dict):
        if self._taxonomy_loss is not None:
            self._taxonomy_mapper = TaxonomyMapper(
                label_to_idx     = label_to_idx,
                species_to_taxon = species_to_taxon,
            )

    # ── Prototype rarity initialisation ──────────────────────────────────

    def set_class_counts(self, class_counts: torch.Tensor):
        """
        Call this from main_train.py after building the dataset, passing
        the per-class training sample counts so rarity weights are correct.

        Args:
            class_counts : (C,) int tensor of per-class clip counts
        """
        if self._proto_head is not None:
            self._proto_head.set_rarity_weights(
                class_counts,
                half_life  = self.cfg.get("proto_half_life",  50),
                max_weight = self.cfg.get("proto_max_weight", 0.75),
            )

    # ── EMA ───────────────────────────────────────────────────────────────

    def on_train_start(self):
        if self.use_ema:
            import copy
            self._ema_model = copy.deepcopy(self.model)
            for p in self._ema_model.parameters():
                p.requires_grad_(False)
            self._ema_model.eval()

    def _update_ema(self):
        if self._ema_model is None:
            return
        with torch.no_grad():
            for ema_p, model_p in zip(
                self._ema_model.parameters(), self.model.parameters()
            ):
                ema_p.mul_(self.ema_decay).add_(model_p, alpha=1 - self.ema_decay)

    def _inference_model(self) -> nn.Module:
        """Return EMA model if available, else main model."""
        return self._ema_model if self._ema_model is not None else self.model

    # ── Training step ─────────────────────────────────────────────────────

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        audio  = batch["audio"]      # (B, T)
        labels = batch["label"]      # (B, C)

        # MixUp in audio domain (50% probability per sample, vectorized)
        if self.mixup.p > 0:
            perm   = torch.randperm(audio.size(0), device=audio.device)
            audio2 = audio[perm]
            labels2 = labels[perm]

            # Vectorized: decide per-sample whether to mix
            do_mix = torch.rand(audio.size(0)) < self.mixup.p  # (B,) bool
            if do_mix.any():
                mixed_audio  = audio.clone()
                mixed_labels = labels.clone()
                idx = do_mix.nonzero(as_tuple=True)[0]
                mixed_audio[idx]  = (audio[idx] + self.mixup.scale * audio2[idx]).clamp(-1.0, 1.0)
                mixed_labels[idx] = (labels[idx] + labels2[idx]).clamp(0.0, 1.0)
                audio  = mixed_audio
                labels = mixed_labels

        # Background noise augmentation (audio domain, before mel)
        if self.bg_noise is not None and self.training:
            audio_np = audio.cpu().numpy()
            labels_np = labels.cpu().numpy()
            mixed_audio_list = []
            mixed_label_list = []
            for i in range(audio_np.shape[0]):
                a, l = self.bg_noise(audio_np[i], labels_np[i])
                mixed_audio_list.append(a)
                mixed_label_list.append(l)
            audio  = torch.tensor(np.stack(mixed_audio_list), device=audio.device)
            labels = torch.tensor(np.stack(mixed_label_list), device=labels.device)

        # Mel extraction (GPU, nnAudio)
        mel = self.mel(audio)         # (B, 1, n_mels, T')

        # Spectrogram augmentations (training mode)
        self.spec_aug.train()
        mel = self.spec_aug(mel)

        # ── NOVEL #5: Noise profile from soundscape buffer ────────────────
        noise_profile = None
        if self.use_noise_conditioning and self._noise_extractor is not None:
            if self._dann_buffer is not None:
                sc_audio_noise = self._dann_buffer.sample(audio.size(0), audio.device)
                sc_mel_noise   = self.mel(sc_audio_noise)                  # (B, 1, n_mels, T')
                with torch.no_grad():
                    noise_profile = self._noise_extractor(sc_mel_noise)    # (B, noise_dim)
            else:
                # No soundscape buffer: use zeros (identity FiLM — no conditioning)
                noise_profile = torch.zeros(
                    audio.size(0), self.cfg.get("noise_dim", 128), device=audio.device
                )
        # ─────────────────────────────────────────────────────────────────

        # Forward — extract features for species head, DANN, and prototypes
        features = self.model.extract_features(mel)                        # (B, feat_dim)

        # Pass noise profile to head (NoisyConditionedHead uses it; others ignore it)
        if self.use_noise_conditioning:
            logits = self.model.head(features, noise_profile)             # (B, n_classes)
        else:
            logits = self.model.head(features)                            # (B, n_classes)

        # Species classification loss (BCE + Focal + optional SoftAUC)
        loss = self.loss_fn(logits, labels)

        # ── NOVEL: DANN Domain Adaptation (Contribution #1) ───────────────
        if self.use_dann and self._domain_clf is not None and self._dann_buffer is not None:
            # Compute current DANN λ based on training progress
            total_steps   = self.trainer.estimated_stepping_batches
            current_step  = self.global_step
            p             = min(current_step / max(total_steps, 1), 1.0)
            dann_lambda   = compute_dann_lambda(p, max_lambda=self.dann_lambda)
            self._domain_clf.set_lambda(dann_lambda)

            # Sample unlabeled soundscape audio (no labels needed)
            sc_audio    = self._dann_buffer.sample(audio.size(0), audio.device)
            sc_mel      = self.mel(sc_audio)                     # (B, 1, n_mels, T')
            sc_features = self.model.extract_features(sc_mel)    # (B, feat_dim)

            # Domain predictions (gradient reversal is inside DomainClassifier)
            dom_logits_xc = self._domain_clf(features)           # (B, 1) — label=0 (XC)
            dom_logits_sc = self._domain_clf(sc_features)        # (B, 1) — label=1 (soundscape)

            domain_loss = self._domain_loss(dom_logits_xc, dom_logits_sc)
            loss = loss + dann_lambda * domain_loss

            if batch_idx % 100 == 0:
                self.log("train/domain_loss", domain_loss,
                         on_step=True, prog_bar=False)
                self.log("train/dann_lambda", dann_lambda,
                         on_step=True, prog_bar=False)
        # ─────────────────────────────────────────────────────────────────

        # ── NOVEL #4: Prototypical loss for rare species ──────────────────
        if self.use_prototypical and self._proto_head is not None:
            # Get 512-dim embeddings from the head
            if self.use_noise_conditioning and hasattr(self.model.head, "get_embedding"):
                embeddings = self.model.head.get_embedding(features, noise_profile)
            elif hasattr(self.model.head, "get_embedding"):
                embeddings = self.model.head.get_embedding(features)
            else:
                embeddings = None

            if embeddings is not None:
                proto_loss = self._proto_loss_fn(
                    embeddings     = embeddings,
                    labels         = labels,
                    prototypes     = self._proto_head.prototypes,
                    rarity_weights = self._proto_head.rarity_weights,
                    temperature    = self._proto_head.temperature,
                )
                loss = loss + proto_loss

                if batch_idx % 100 == 0:
                    self.log("train/proto_loss", proto_loss,
                             on_step=True, prog_bar=False)
        # ─────────────────────────────────────────────────────────────────

        # ── NOVEL #6: Temporal Consistency Regularization ─────────────────
        if self.use_tcr and self._tcr_loss is not None:
            filenames  = batch.get("filename", [])
            start_secs = batch.get("start_sec", None)
            if filenames and start_secs is not None:
                tcr_loss = self._tcr_loss(logits, filenames, start_secs)
                loss = loss + tcr_loss
                if batch_idx % 100 == 0:
                    self.log("train/tcr_loss", tcr_loss,
                             on_step=True, prog_bar=False)
        # ─────────────────────────────────────────────────────────────────

        # ── NOVEL #7: Taxonomy-Aware Hierarchical Loss ───────────────────
        if (self.use_taxonomy and self._taxonomy_head is not None
                and self._taxonomy_mapper is not None):
            taxon_logits = self._taxonomy_head(features)
            taxon_labels = self._taxonomy_mapper.get_taxon_labels(labels)
            taxonomy_loss = self._taxonomy_loss(
                species_logits = logits,
                taxon_logits   = taxon_logits,
                species_labels = labels,
                taxon_labels   = taxon_labels,
                mapper         = self._taxonomy_mapper,
            )
            loss = loss + taxonomy_loss
            if batch_idx % 100 == 0:
                self.log("train/taxonomy_loss", taxonomy_loss,
                         on_step=True, prog_bar=False)
        # ─────────────────────────────────────────────────────────────────

        # EMA update
        if batch_idx % 5 == 0:
            self._update_ema()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    # ── Validation step ───────────────────────────────────────────────────

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        audio  = batch["audio"]
        labels = batch["label"]

        self.spec_aug.eval()
        with torch.no_grad():
            mel    = self.mel(audio)
            logits = self._inference_model().get_logits(mel)
            loss   = self.loss_fn(logits, labels)
            probs  = torch.sigmoid(logits)

        self._val_probs.append(probs.float().cpu().numpy())
        self._val_targets.append(labels.float().cpu().numpy())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self._val_probs:
            return

        probs   = np.concatenate(self._val_probs,   axis=0)
        targets = np.concatenate(self._val_targets, axis=0)

        # Aggregate to file level using max (paper Appendix B)
        # For simplicity here we use clip-level AUC (use padded_auc_score)
        auc = padded_auc_score(targets, probs)
        self.log("val/auc", auc, prog_bar=True)
        logger.info(f"Epoch {self.current_epoch}: val/auc = {auc:.4f}")

        self._val_probs.clear()
        self._val_targets.clear()

    # ── Optimizers (exact from paper Appendix B) ──────────────────────────

    def configure_optimizers(self):
        """
        EfficientNetV2-S: AdamW, lr=1e-4, ε=1e-8, β=(0.9,0.999)
        NFNet-L0:         RAdam, lr=1e-3
        Both: cosine schedule to 1e-6, NO warmup
        """
        backbone = self.cfg["backbone"]
        lr       = self.cfg.get("lr", None)

        if lr is None:
            lr = 1e-3 if "nfnet" in backbone else 1e-4

        min_lr       = self.cfg.get("min_lr", 1e-6)
        weight_decay = self.cfg.get("weight_decay", 1e-4)

        # Collect parameters: main model + all novel components
        param_groups = [{"params": self.model.parameters(), "lr": lr}]

        if self.use_dann and self._domain_clf is not None:
            # Domain classifier learns faster than backbone
            param_groups.append(
                {"params": self._domain_clf.parameters(), "lr": lr * 3.0}
            )

        if self.use_prototypical and self._proto_head is not None:
            # Prototypes + prototype head at same LR as main model
            param_groups.append(
                {"params": self._proto_head.parameters(), "lr": lr}
            )

        if self.use_noise_conditioning and self._noise_extractor is not None:
            # Noise extractor MLP at standard LR
            param_groups.append(
                {"params": self._noise_extractor.parameters(), "lr": lr}
            )

        if self.use_taxonomy and self._taxonomy_head is not None:
            param_groups.append(
                {"params": self._taxonomy_head.parameters(), "lr": lr * 2.0}
            )

        params = param_groups if len(param_groups) > 1 else list(self.model.parameters())

        if "nfnet" in backbone:
            optimizer = torch.optim.RAdam(
                params,
                lr           = lr,
                weight_decay = weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr           = lr,
                eps          = 1e-8,
                betas        = (0.9, 0.999),
                weight_decay = weight_decay,
            )

        # Cosine schedule, NO warmup (paper: "cosine schedule down to 1e-6 without warm-up")
        total_steps  = self.cfg.get("epochs", 50) * self.cfg.get("steps_per_epoch", 1000)
        scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max    = total_steps,
            eta_min  = min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    # ── Inference predict_step ────────────────────────────────────────────

    def predict_step(
        self, batch: Dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        audio = batch["audio"]
        with torch.no_grad():
            mel   = self.mel(audio)
            probs = self._inference_model().get_probabilities(mel)
        return probs
