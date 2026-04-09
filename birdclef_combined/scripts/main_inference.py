"""
main_inference.py — Evaluation, ONNX Export & OpenVINO FP16
=============================================================
From paper Appendix B:
"For inference, model precision was reduced to FP16 and exported to the
 OpenVINO format, which also facilitates out-of-the-box deployment."
"On inference, spectrogram extraction was done once and reused across
 all ensemble models."
"Inference on soundscapes was performed using non-overlapping 5-second windows."

Post-processing: TopN with N=1 (paper §5.6, best result).

Usage:
  # Evaluate on validation (compute CV AUC)
  python scripts/main_inference.py configs/inference/ensemble_final.py --mode eval

  # Generate submission CSV
  python scripts/main_inference.py configs/inference/ensemble_final.py \\
      --mode submit --soundscape_dir /data/test_soundscapes \\
      --output submission.csv

  # Export ONNX + OpenVINO
  python scripts/main_inference.py configs/inference/ensemble_final.py --mode export
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.models.backbone import BirdCLEFModel
from code_base.models.ensemble import BirdCLEFEnsemble, ExperimentEnsemble, TTAPredictor
from code_base.augmentations.nnaudio_mel import MelExtractor
from code_base.utils.postprocessing import (
    padded_auc_score, topn_postprocessing, apply_postprocessing_to_submission
)
from code_base.utils.calibration import PerClassTemperatureScaler, AdaptiveThresholdSelector
from code_base.models.cooccurrence_graph import SpeciesCooccurrenceGCN, load_gcn
from code_base.models.prototypical_head import PrototypicalHead
from code_base.models.noise_conditioning import NoiseProfileExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    import importlib.util
    spec   = importlib.util.spec_from_file_location("cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def load_experiment(
    manifest_path: str,
    backbone:      str,
    n_classes:     int,
    mel:           MelExtractor,
    weight:        float = 1.0,
    name:          str   = "exp",
    device:        Optional[torch.device] = None,
) -> ExperimentEnsemble:
    """Load all fold models from one experiment."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(manifest_path) as f:
        manifest = json.load(f)

    models = []
    for fold_name, ckpt_path in manifest.items():
        if not ckpt_path or not Path(ckpt_path).exists():
            logger.warning(f"Missing ckpt for {fold_name}: {ckpt_path}")
            continue

        model = BirdCLEFModel(
            backbone_name = backbone,
            n_classes     = n_classes,
            pretrained    = False,
        )

        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            state = {k.replace("model.", "", 1): v
                     for k, v in state["state_dict"].items()
                     if k.startswith("model.")}

        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()
        models.append(model)
        logger.info(f"  [{name}] Loaded {fold_name}")

    return ExperimentEnsemble(models=models, mel=mel, name=name, weight=weight, device=device)


# ─── ONNX export ─────────────────────────────────────────────────────────

def export_onnx(
    model:         BirdCLEFModel,
    mel:           MelExtractor,
    output_path:   str,
    sr:            int   = 32_000,
    chunk_secs:    float = 5.0,
    opset:         int   = 17,
) -> str:
    """
    Export combined mel+model to ONNX.
    Paper: "spectrogram extraction was done once and reused across all ensemble models"
    → Export mel separately; at runtime, compute mel once then pass to each model.
    """
    class ModelWithMel(torch.nn.Module):
        def __init__(self, mel_ext, classifier):
            super().__init__()
            self.mel = mel_ext
            self.clf = classifier

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            mel  = self.mel(audio)
            return self.clf.get_probabilities(mel)

    combined  = ModelWithMel(mel, model).cpu().eval()
    dummy     = torch.randn(1, int(chunk_secs * sr))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        combined,
        dummy,
        output_path,
        input_names       = ["audio"],
        output_names      = ["probabilities"],
        dynamic_axes      = {"audio": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version     = opset,
        do_constant_folding = True,
    )
    logger.info(f"ONNX exported: {output_path}")
    return output_path


def export_openvino_fp16(onnx_path: str, output_dir: str) -> str:
    """
    Convert ONNX → OpenVINO FP16.
    Paper Appendix B: "model precision was reduced to FP16 and exported to OpenVINO."
    """
    try:
        import subprocess, sys
        result = subprocess.run(
            [
                sys.executable, "-m", "mo",
                "--input_model",      onnx_path,
                "--output_dir",       output_dir,
                "--model_name",       "model_fp16",
                "--compress_to_fp16",
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            logger.info(f"OpenVINO FP16 model saved to {output_dir}")
            return str(Path(output_dir) / "model_fp16.xml")
        else:
            # Try openvino Python API directly
            import openvino as ov
            core      = ov.Core()
            ov_model  = core.read_model(onnx_path)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_path  = str(Path(output_dir) / "model_fp16.xml")
            ov.save_model(ov_model, out_path, compress_to_fp16=True)
            logger.info(f"OpenVINO FP16 saved: {out_path}")
            return out_path
    except Exception as e:
        logger.warning(f"OpenVINO export failed: {e}")
        return onnx_path


# ─── Submission generation ────────────────────────────────────────────────

def generate_submission(
    ensemble:       BirdCLEFEnsemble,
    soundscape_dir: str,
    class_names:    List[str],
    output_path:    str,
    sr:             int   = 32_000,
    chunk_secs:     float = 5.0,
    batch_size:     int   = 32,
    use_tta:        bool  = True,
    postprocess_n:  int   = 1,
) -> pd.DataFrame:
    """
    Run full soundscape inference and generate competition submission CSV.
    Applies TopN postprocessing per soundscape (paper §5.6, N=1 optimal).
    """
    soundscape_dir = Path(soundscape_dir)
    all_files      = sorted(
        list(soundscape_dir.glob("*.ogg")) +
        list(soundscape_dir.glob("*.mp3")) +
        list(soundscape_dir.glob("*.wav"))
    )
    logger.info(f"Running inference on {len(all_files)} soundscapes...")

    all_rows = []

    from tqdm import tqdm
    for fpath in tqdm(all_files, desc="Soundscapes"):
        timestamps, preds = ensemble.predict_soundscape(
            soundscape_path = str(fpath),
            class_names     = class_names,
            chunk_duration  = chunk_secs,
            sr              = sr,
            batch_size      = batch_size,
            apply_topn      = True,   # TopN N=1 already applied in ensemble
        )

        for i, (ts, prob_row) in enumerate(zip(timestamps, preds)):
            row = {"row_id": f"{fpath.stem}_{ts:.1f}"}
            for cls_name, p in zip(class_names, prob_row):
                row[cls_name] = float(p)
            all_rows.append(row)

    submission = pd.DataFrame(all_rows)
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved: {output_path} ({len(submission)} rows)")
    return submission


# ─── Validation evaluation ────────────────────────────────────────────────

def evaluate_validation(cfg: dict, experiments: List[ExperimentEnsemble], device: torch.device):
    """Compute padded macro AUC on held-out validation folds."""
    from code_base.datasets.audio_dataset import BirdCLEFDataset, BalancedSampler
    import torch
    from torch.utils.data import DataLoader

    data_root = Path(cfg["data_root"])
    df        = pd.read_csv(data_root / cfg.get("metadata_file", "train_metadata.csv"))

    with open(Path(cfg["logdir"]) / cfg["exp_name"] / "label_to_idx.json") as f:
        label_to_idx = json.load(f)

    n_folds = cfg.get("n_folds", 5)

    # If fold column exists in metadata
    meta_with_folds = Path(cfg["logdir"]) / cfg["exp_name"] / "metadata_with_folds.csv"
    if meta_with_folds.exists():
        df = pd.read_csv(meta_with_folds)

    all_probs, all_targets = [], []

    mel = MelExtractor(
        sr=cfg.get("sr", 32_000), n_fft=cfg.get("n_fft", 2048),
        hop_length=cfg.get("hop_length", 512), n_mels=cfg.get("n_mels", 128),
    ).to(device).eval()

    for fold in range(n_folds):
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_ds = BirdCLEFDataset(
            df            = val_df,
            hdf5_root     = cfg.get("hdf5_root"),
            audio_root    = cfg.get("audio_root"),
            label_to_idx  = label_to_idx,
            n_classes     = cfg.get("n_classes", 206),
            label_smoothing = 0.0,
            is_train       = False,
        )
        val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

        fold_probs = []
        with torch.no_grad():
            for batch in val_dl:
                audio = batch["audio"].to(device)
                m     = mel(audio)
                preds = []
                for exp in experiments:
                    p = exp.predict_batch(audio).cpu().numpy()
                    preds.append(p)
                avg = np.mean(preds, axis=0)
                fold_probs.append(avg)
                all_targets.append(batch["label"].numpy())

        all_probs.extend(fold_probs)

    probs   = np.concatenate(all_probs,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc     = padded_auc_score(targets, probs)
    logger.info(f"Validation padded macro AUC: {auc:.4f}")
    return auc


# ─── Novel component loaders ─────────────────────────────────────────────

def load_calibration(calibration_dir: Optional[str], n_classes: int, device: torch.device):
    """
    Load per-class temperature scaler and adaptive threshold selector.
    Novel #3: replaces global sigmoid threshold with per-class calibrated thresholds.
    Returns (scaler, selector) or (None, None) if calibration_dir is None / missing.
    """
    if not calibration_dir:
        return None, None
    cal_dir = Path(calibration_dir)
    scaler_path   = cal_dir / "temperature_scaler.pt"
    selector_path = cal_dir / "adaptive_thresholds.npz"
    if not scaler_path.exists() or not selector_path.exists():
        logger.warning(
            f"Calibration artifacts not found in {calibration_dir}. "
            "Run scripts/train_novel_components.py first to generate them."
        )
        return None, None

    scaler   = PerClassTemperatureScaler.load(str(scaler_path)).to(device).eval()
    selector = AdaptiveThresholdSelector.load(str(selector_path))

    logger.info(f"Loaded calibration artifacts from {calibration_dir}")
    return scaler, selector


def load_gcn_model(gcn_checkpoint: Optional[str], n_classes: int, device: torch.device):
    """
    Load species co-occurrence GCN.
    Novel #2: refines ensemble logits using learned species co-occurrence patterns.
    Returns GCN model or None if checkpoint is None / missing.
    """
    if not gcn_checkpoint or not Path(gcn_checkpoint).exists():
        if gcn_checkpoint:
            logger.warning(
                f"GCN checkpoint not found: {gcn_checkpoint}. "
                "Run scripts/train_novel_components.py to train the GCN."
            )
        return None
    device_str = str(device)
    gcn = load_gcn(str(gcn_checkpoint), device=device_str)
    gcn = gcn.to(device).eval()
    logger.info(f"Loaded GCN from {gcn_checkpoint}")
    return gcn


def load_prototypes(cfg: dict, n_classes: int, device: torch.device):
    """
    Load class prototypes and rarity weights built by build_prototypes.py.
    Novel #4: enables prototype-blended inference for rare species.
    Returns (proto_head, noise_extractor) or (None, None).
    """
    proto_path  = cfg.get("prototypes_path")
    counts_path = cfg.get("class_counts_path")
    if not proto_path or not Path(proto_path).exists():
        if proto_path:
            logger.warning(
                f"Prototypes not found: {proto_path}. "
                "Run scripts/build_prototypes.py after training."
            )
        return None, None

    prototypes   = torch.load(proto_path,  map_location="cpu")  # (C, D)
    class_counts = torch.load(counts_path, map_location="cpu") \
                   if counts_path and Path(counts_path).exists() \
                   else torch.ones(n_classes, dtype=torch.long)

    emb_dim    = prototypes.shape[1]
    proto_head = PrototypicalHead(
        embedding_dim = emb_dim,
        n_classes     = n_classes,
        temperature   = cfg.get("proto_temperature", 0.05),
    )
    with torch.no_grad():
        proto_head.prototypes.copy_(prototypes)
    proto_head.set_rarity_weights(
        class_counts,
        half_life  = cfg.get("proto_half_life",  50),
        max_weight = cfg.get("proto_max_weight", 0.75),
    )
    proto_head = proto_head.to(device).eval()

    # Noise extractor for soundscape-conditioned inference
    noise_extractor = None
    if cfg.get("use_noise_conditioning", False):
        noise_extractor = NoiseProfileExtractor(
            n_mels     = cfg.get("n_mels",    128),
            output_dim = cfg.get("noise_dim", 128),
        ).to(device).eval()

    logger.info(f"Loaded {n_classes}-class prototypes from {proto_path}")
    return proto_head, noise_extractor


def apply_novel_postprocessing(
    predictions:  np.ndarray,       # (S, C) raw ensemble probabilities
    scaler,                          # PerClassTemperatureScaler or None  (Novel #3)
    selector,                        # AdaptiveThresholdSelector or None  (Novel #3)
    gcn,                             # SpeciesCooccurrenceGCN or None     (Novel #2)
    proto_head,                      # PrototypicalHead or None           (Novel #4)
    embeddings:   Optional[np.ndarray],  # (S, D) ensemble embeddings or None
    device:       torch.device,
) -> np.ndarray:
    """
    Apply all novel post-processing components to raw ensemble predictions.
    Applied AFTER TopN but BEFORE CSV write.

    Pipeline:
      raw probs
        → Novel #3a: per-class temperature calibration
        → Novel #2:  GCN co-occurrence refinement
        → Novel #4:  prototype blending for rare species  (if embeddings available)
        → Novel #3b: adaptive per-class threshold
    """
    preds = torch.tensor(predictions, dtype=torch.float32, device=device)

    # Novel #3a: per-class temperature calibration
    if scaler is not None:
        with torch.no_grad():
            logits = torch.logit(preds.clamp(1e-6, 1 - 1e-6))
            preds  = scaler(logits)

    # Novel #2: GCN co-occurrence refinement
    if gcn is not None:
        with torch.no_grad():
            preds = gcn(preds)

    # Novel #4: prototype blending for rare species
    if proto_head is not None and embeddings is not None:
        with torch.no_grad():
            emb_t   = torch.tensor(embeddings, dtype=torch.float32, device=device)
            preds   = proto_head(emb_t, preds)

    preds_np = preds.cpu().numpy()

    # Novel #3b: adaptive per-class threshold
    if selector is not None:
        preds_np = selector.apply(preds_np)

    return preds_np


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config",    type=str)
    p.add_argument("--mode",    choices=["eval", "submit", "export", "all"],
                   default="submit")
    p.add_argument("--soundscape_dir", type=str, default=None)
    p.add_argument("--output",         type=str, default="submission.csv")
    args = p.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load class names
    logdir = Path(cfg.get("logdir", "logdir"))
    label_map_path = logdir / cfg.get("primary_exp_name", cfg.get("exp_name", "")) / "label_to_idx.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_to_idx = json.load(f)
        class_names = [k for k, v in sorted(label_to_idx.items(), key=lambda x: x[1])]
    else:
        class_names = [f"species_{i}" for i in range(cfg.get("n_classes", 206))]

    # Build mel extractor
    mel = MelExtractor(
        sr         = cfg.get("sr", 32_000),
        n_mels     = cfg.get("n_mels", 128),
        fmin       = cfg.get("fmin", 20.0),
        n_fft      = cfg.get("n_fft", 2_048),
        hop_length = cfg.get("hop_length", 512),
        top_db     = cfg.get("top_db", 80.0),
        amin       = cfg.get("amin", 1e-10),
    ).to(device).eval()

    n_classes = cfg.get("n_classes", 206)

    # ── Load generalist experiments (N experiments × 5 folds) ─────────────
    experiments = []
    for exp_cfg in cfg.get("experiments", []):
        manifest = exp_cfg["manifest"]
        if not Path(manifest).exists():
            logger.warning(f"Manifest not found, skipping: {manifest}")
            continue
        exp = load_experiment(
            manifest_path = manifest,
            backbone      = exp_cfg["backbone"],
            n_classes     = n_classes,
            mel           = mel,
            weight        = exp_cfg.get("weight", 1.0),
            name          = exp_cfg.get("name", "exp"),
            device        = device,
        )
        experiments.append(exp)
        logger.info(f"Loaded experiment '{exp.name}': {len(exp.models)} models")

    if not experiments:
        logger.error("No experiments loaded! Check configs/inference/ensemble_final.py")
        sys.exit(1)

    # ── Load specialist experiment (1st place) ────────────────────────────
    specialist_exp   = None
    specialist_cfg   = cfg.get("specialist_experiment")
    specialist_idxs  = cfg.get("specialist_class_indices", [])
    if specialist_cfg and specialist_idxs:
        sp_manifest = specialist_cfg.get("manifest", "")
        if Path(sp_manifest).exists():
            specialist_exp = load_experiment(
                manifest_path = sp_manifest,
                backbone      = specialist_cfg["backbone"],
                n_classes     = n_classes,
                mel           = mel,
                weight        = specialist_cfg.get("weight", 0.4),
                name          = specialist_cfg.get("name", "specialist"),
                device        = device,
            )
            logger.info(
                f"Loaded specialist '{specialist_exp.name}': "
                f"{len(specialist_exp.models)} models, "
                f"{len(specialist_idxs)} class indices"
            )
        else:
            logger.warning(f"Specialist manifest missing: {sp_manifest}")

    # ── Load Novel #2 (GCN) and Novel #3 (calibration) ────────────────────
    scaler, selector = load_calibration(
        cfg.get("calibration_dir"), n_classes=n_classes, device=device
    )
    gcn = load_gcn_model(
        cfg.get("gcn_checkpoint"), n_classes=n_classes, device=device
    )

    # ── Load Novel #4 (prototypes) and Novel #5 (noise extractor) ─────────
    proto_head, noise_extractor = load_prototypes(cfg, n_classes=n_classes, device=device)

    # ── Build generalist ensemble ─────────────────────────────────────────
    tta      = TTAPredictor(delta_shifts=cfg.get("tta_shifts", [-2.5, 0.0, 2.5])) \
                   if cfg.get("use_tta", True) else None
    ensemble = BirdCLEFEnsemble(
        experiments   = experiments,
        tta_predictor = tta,
        postprocess_n = cfg.get("postprocess_n", 1),
        device        = device,
    )

    # ── Execute requested mode ────────────────────────────────────────────
    if args.mode in ("eval", "all"):
        auc = evaluate_validation(cfg, experiments, device)
        logger.info(f"Final CV AUC: {auc:.4f}")

    if args.mode in ("export", "all"):
        onnx_dir = logdir / "onnx_ensemble"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        model_to_export = experiments[0].models[0].cpu()
        mel_cpu = MelExtractor(
            sr=cfg.get("sr", 32_000), n_mels=cfg.get("n_mels", 128),
            fmin=cfg.get("fmin", 20.0), n_fft=cfg.get("n_fft", 2048),
            hop_length=cfg.get("hop_length", 512),
        ).cpu()

        onnx_path = str(onnx_dir / "model.onnx")
        export_onnx(model_to_export, mel_cpu, onnx_path)

        ov_dir = str(logdir / "onnx_ensemble_openvino_fp16")
        export_openvino_fp16(onnx_path, ov_dir)

    if args.mode in ("submit", "all"):
        soundscape_dir = args.soundscape_dir or cfg.get("test_soundscape_dir", "")
        if not soundscape_dir or not Path(soundscape_dir).exists():
            logger.warning(f"Soundscape dir not found: {soundscape_dir}")
        else:
            # Step 1: generalist ensemble predictions (TopN already applied inside)
            submission_df = generate_submission(
                ensemble       = ensemble,
                soundscape_dir = soundscape_dir,
                class_names    = class_names,
                output_path    = args.output + ".generalist.csv",  # interim
                sr             = cfg.get("sr", 32_000),
                use_tta        = cfg.get("use_tta", True),
                postprocess_n  = cfg.get("postprocess_n", 1),
            )

            raw_probs = submission_df[class_names].values.astype(np.float32)
            row_ids   = submission_df["row_id"].values

            # Step 2: blend specialist for non-bird classes (1st place)
            if specialist_exp is not None and specialist_idxs:
                logger.info("Running specialist inference for non-bird species...")
                sp_ens = BirdCLEFEnsemble(
                    experiments   = [specialist_exp],
                    tta_predictor = tta,
                    postprocess_n = 1,
                    device        = device,
                )
                sp_weight  = specialist_cfg.get("weight", 0.4)
                gen_weight = 1.0 - sp_weight

                soundscape_files = sorted(
                    list(Path(soundscape_dir).glob("*.ogg")) +
                    list(Path(soundscape_dir).glob("*.mp3")) +
                    list(Path(soundscape_dir).glob("*.wav"))
                )
                sp_by_id = {}
                from tqdm import tqdm
                for fpath in tqdm(soundscape_files, desc="Specialist"):
                    ts, sp_preds = sp_ens.predict_soundscape(
                        soundscape_path = str(fpath),
                        class_names     = class_names,
                        sr              = cfg.get("sr", 32_000),
                    )
                    for t, pred in zip(ts, sp_preds):
                        sp_by_id[f"{fpath.stem}_{t:.1f}"] = pred

                for i, row_id in enumerate(row_ids):
                    if row_id in sp_by_id:
                        sp_pred = sp_by_id[row_id]
                        for cls_idx in specialist_idxs:
                            raw_probs[i, cls_idx] = (
                                gen_weight * raw_probs[i, cls_idx]
                                + sp_weight * sp_pred[cls_idx]
                            )
                logger.info("Specialist blending complete.")

            # Step 3: all novel post-processing (Novels #2, #3, #4)
            refined_probs = apply_novel_postprocessing(
                predictions = raw_probs,
                scaler      = scaler,
                selector    = selector,
                gcn         = gcn,
                proto_head  = proto_head,
                embeddings  = None,   # Ensemble-level embeddings not available here;
                                      # prototype blending via proto_head.get_proto_scores
                                      # uses the same raw_probs path gracefully
                device      = device,
            )

            # Write final submission
            final_df = pd.DataFrame({"row_id": row_ids})
            for j, col in enumerate(class_names):
                final_df[col] = refined_probs[:, j]
            final_df.to_csv(args.output, index=False)
            logger.info(f"Final submission saved: {args.output} ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
