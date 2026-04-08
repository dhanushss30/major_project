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
from ..augmentations.nnaudio_mel import MelExtractor
from ..augmentations.audio_aug import AudioMixUp, SpectrogramAugmenter
from ..losses.focal_bce import CombinedLoss
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
            backbone_name   = cfg["backbone"],
            n_classes       = cfg.get("n_classes", 206),
            pretrained      = cfg.get("pretrained", True),
            pretrained_path = cfg.get("pretrained_path", None),
            hidden_dim      = cfg.get("hidden_dim", 512),
            dropout1        = cfg.get("dropout1", 0.25),
            dropout2        = cfg.get("dropout2", 0.5),
            gem_p           = cfg.get("gem_p", 3.0),
            use_sed_head    = cfg.get("use_sed_head", False),
        )

        # Mel extractor (nnAudio, on GPU)
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

        # EMA
        self.use_ema    = cfg.get("use_ema", True)
        self.ema_decay  = cfg.get("ema_decay", 0.999)
        self._ema_model = None

        # Validation cache
        self._val_probs:   List[np.ndarray] = []
        self._val_targets: List[np.ndarray] = []

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

        # MixUp in audio domain (50% probability, applied batch-wise)
        if self.mixup.p > 0 and torch.rand(1).item() < self.mixup.p:
            perm      = torch.randperm(audio.size(0), device=audio.device)
            audio2    = audio[perm]
            labels2   = labels[perm]

            # Mix in numpy (AudioMixUp operates on numpy arrays)
            mixed_audio_list  = []
            mixed_label_list  = []
            for i in range(audio.size(0)):
                a, l = self.mixup(
                    audio[i].cpu().numpy(),  labels[i].cpu().numpy(),
                    audio2[i].cpu().numpy(), labels2[i].cpu().numpy(),
                )
                mixed_audio_list.append(a)
                mixed_label_list.append(l)

            audio  = torch.tensor(np.stack(mixed_audio_list), device=audio.device)
            labels = torch.tensor(np.stack(mixed_label_list), device=labels.device)

        # Mel extraction (GPU, nnAudio)
        mel = self.mel(audio)         # (B, 1, n_mels, T')

        # Spectrogram augmentations (training mode)
        self.spec_aug.train()
        mel = self.spec_aug(mel)

        # Forward
        logits = self.model.get_logits(mel)

        # Loss
        loss = self.loss_fn(logits, labels)

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

        if "nfnet" in backbone:
            optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr           = lr,
                weight_decay = weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
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
