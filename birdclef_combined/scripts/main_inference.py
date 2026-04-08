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

    # Load experiments (15 models: 3 experiments × 5 folds)
    experiments = []
    for exp_cfg in cfg.get("experiments", []):
        exp = load_experiment(
            manifest_path = exp_cfg["manifest"],
            backbone      = exp_cfg["backbone"],
            n_classes     = cfg.get("n_classes", 206),
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

    # Build ensemble
    tta       = TTAPredictor(delta_shifts=[-2.5, 0.0, 2.5]) if cfg.get("use_tta", True) else None
    ensemble  = BirdCLEFEnsemble(
        experiments   = experiments,
        tta_predictor = tta,
        postprocess_n = cfg.get("postprocess_n", 1),
        device        = device,
    )

    # Execute requested mode
    if args.mode in ("eval", "all"):
        auc = evaluate_validation(cfg, experiments, device)
        logger.info(f"Final CV AUC: {auc:.4f}")

    if args.mode in ("export", "all"):
        onnx_dir = logdir / "onnx_ensemble"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        # Export first fold's first experiment as representative
        model_to_export = experiments[0].models[0].cpu()
        mel_cpu         = MelExtractor(**{k: cfg.get(k, v) for k, v in {
            "sr": 32_000, "n_mels": 128, "fmin": 20.0,
            "n_fft": 2048, "hop_length": 512
        }.items()}).cpu()

        onnx_path = str(onnx_dir / "model.onnx")
        export_onnx(model_to_export, mel_cpu, onnx_path)

        ov_dir = str(logdir / "onnx_ensemble_openvino_fp16")
        export_openvino_fp16(onnx_path, ov_dir)

    if args.mode in ("submit", "all"):
        soundscape_dir = args.soundscape_dir or cfg.get("test_soundscape_dir", "")
        if not soundscape_dir or not Path(soundscape_dir).exists():
            logger.warning(f"Soundscape dir not found: {soundscape_dir}")
        else:
            generate_submission(
                ensemble       = ensemble,
                soundscape_dir = soundscape_dir,
                class_names    = class_names,
                output_path    = args.output,
                sr             = cfg.get("sr", 32_000),
                use_tta        = cfg.get("use_tta", True),
                postprocess_n  = cfg.get("postprocess_n", 1),
            )


if __name__ == "__main__":
    main()
