"""
build_prototypes.py — Build Class Prototypes from Trained Model
================================================================
Run AFTER supervised training to compute the mean embedding per class.

Why this matters:
  During training, PrototypicalHead prototypes are initialised randomly
  and updated only via PrototypicalLoss. After training, we can compute
  MUCH better prototypes by running all training clips through the trained
  backbone and taking the mean embedding per class.

  This is analogous to the "support set" in prototypical networks —
  the prototype becomes the centroid of all training embeddings for that class.

Usage:
    python scripts/build_prototypes.py \\
        --manifest logdir/eca_nfnet_l0_.../checkpoint_manifest.json \\
        --data_root D:/birdclef_data/birdclef-2025 \\
        --output_dir logdir/eca_nfnet_l0_... \\
        --backbone eca_nfnet_l0 \\
        --fold 0

Output files in output_dir/:
    class_prototypes.pt      — (C, hidden_dim) float32 prototype tensor
    class_counts.pt          — (C,) int64 training sample counts per class
    prototype_manifest.json  — metadata about the prototype extraction run

After running, set in config:
    "prototypes_path": "logdir/<exp>/class_prototypes.pt"
    "class_counts_path": "logdir/<exp>/class_counts.pt"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.models.backbone import BirdCLEFModel
from code_base.augmentations.nnaudio_mel import MelExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(
    manifest_path: str,
    backbone:      str,
    n_classes:     int,
    fold:          int,
    device:        torch.device,
    use_noise_conditioning: bool = False,
) -> BirdCLEFModel:
    """Load a single fold's model from the checkpoint manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    fold_key  = f"fold_{fold}"
    ckpt_path = manifest.get(fold_key)
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint for {fold_key} not found in manifest: {manifest_path}"
        )

    model = BirdCLEFModel(
        backbone_name          = backbone,
        n_classes              = n_classes,
        pretrained             = False,
        use_noise_conditioning = use_noise_conditioning,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = {
            k.replace("model.", "", 1): v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        }
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    logger.info(f"Loaded {backbone} fold {fold} from {ckpt_path}")
    return model


@torch.no_grad()
def extract_embeddings(
    model:       BirdCLEFModel,
    mel:         MelExtractor,
    audio_paths: List[str],
    labels:      List[int],       # primary label index per clip
    n_classes:   int,
    device:      torch.device,
    sr:          int   = 32_000,
    chunk_secs:  float = 5.0,
    batch_size:  int   = 32,
) -> tuple:
    """
    Run all training clips through the model and collect embeddings.

    Returns:
        class_sums   : (C, D) — sum of embeddings per class
        class_counts : (C,)   — number of clips per class
    """
    import soundfile as sf
    import librosa
    from torch.utils.data import DataLoader, Dataset

    chunk_samples = int(chunk_secs * sr)

    class AudioList(Dataset):
        def __init__(self, paths, lbls):
            self.paths = paths
            self.lbls  = lbls
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            try:
                audio, file_sr = sf.read(self.paths[i], dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if file_sr != sr:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
                # Take first chunk
                if len(audio) >= chunk_samples:
                    audio = audio[:chunk_samples]
                else:
                    audio = np.pad(audio, (0, chunk_samples - len(audio)))
            except Exception:
                audio = np.zeros(chunk_samples, dtype=np.float32)
            return torch.tensor(audio, dtype=torch.float32), self.lbls[i]

    dataset = AudioList(audio_paths, labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Detect embedding dim from a dummy forward
    dummy_mel  = mel(torch.zeros(1, chunk_samples).to(device))
    dummy_feat = model.extract_features(dummy_mel)
    if hasattr(model.head, "get_embedding"):
        if hasattr(model.head, "noise_dim"):
            dummy_noise = torch.zeros(1, model.head.noise_dim, device=device)
            emb_dim     = model.head.get_embedding(dummy_feat, dummy_noise).shape[-1]
        else:
            emb_dim     = model.head.get_embedding(dummy_feat).shape[-1]
    else:
        emb_dim = dummy_feat.shape[-1]

    class_sums   = torch.zeros(n_classes, emb_dim, device=device)
    class_counts = torch.zeros(n_classes, dtype=torch.long, device=device)

    from tqdm import tqdm
    for audio_batch, label_batch in tqdm(loader, desc="Extracting embeddings"):
        audio_batch = audio_batch.to(device)
        label_batch = label_batch.to(device)

        mel_batch  = mel(audio_batch)                          # (B, 1, n_mels, T')
        feat_batch = model.extract_features(mel_batch)         # (B, feat_dim)

        if hasattr(model.head, "get_embedding"):
            if hasattr(model.head, "noise_dim"):
                noise = torch.zeros(
                    feat_batch.size(0), model.head.noise_dim, device=device
                )
                embs = model.head.get_embedding(feat_batch, noise)
            else:
                embs = model.head.get_embedding(feat_batch)
        else:
            embs = feat_batch                                  # fallback

        # L2-normalise before averaging (unit sphere mean = well-defined prototype)
        embs = F.normalize(embs, dim=1)                       # (B, D)

        for b in range(embs.size(0)):
            c = label_batch[b].item()
            class_sums[c]   += embs[b]
            class_counts[c] += 1

    return class_sums, class_counts


def build_prototypes_from_manifest(
    manifest_path:         str,
    data_root:             str,
    output_dir:            str,
    backbone:              str,
    fold:                  int   = 0,
    n_classes:             int   = 206,
    use_noise_conditioning: bool = False,
    sr:                    int   = 32_000,
):
    """
    Main routine: load model, extract embeddings, save prototypes.

    Uses fold 0's model on all training data (not just fold 0's validation set)
    to maximise coverage — especially important for rare species.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    import pandas as pd
    data_root = Path(data_root)
    meta_path = data_root / "train.csv"
    if not meta_path.exists():
        meta_path = data_root / "train_metadata.csv"
    df = pd.read_csv(meta_path)

    # Build label → index map from the experiment's saved label map
    manifest_dir   = Path(manifest_path).parent
    label_map_path = manifest_dir / "label_to_idx.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_to_idx = json.load(f)
    else:
        # Fallback: build from metadata
        classes      = sorted(df["primary_label"].unique())
        label_to_idx = {c: i for i, c in enumerate(classes)}
        logger.warning(f"label_to_idx.json not found; built from metadata ({len(classes)} classes)")

    # Build audio paths and primary label indices
    audio_root  = data_root / "train_audio"
    audio_paths = []
    label_idxs  = []
    for _, row in df.iterrows():
        fname = row.get("filename", row.get("primary_label", ""))
        # filename column may already include subdirectory
        fpath = audio_root / str(fname)
        if not fpath.exists():
            fpath = audio_root / row["primary_label"] / str(fname)
        if not fpath.exists():
            continue
        lbl = row["primary_label"]
        if lbl not in label_to_idx:
            continue
        audio_paths.append(str(fpath))
        label_idxs.append(label_to_idx[lbl])

    logger.info(f"Found {len(audio_paths)} audio files for prototype extraction")

    # Load mel extractor
    mel = MelExtractor(sr=sr, n_mels=128, fmin=20.0, n_fft=2048, hop_length=512).to(device).eval()

    # Load model
    model = load_model(
        manifest_path          = manifest_path,
        backbone               = backbone,
        n_classes              = n_classes,
        fold                   = fold,
        device                 = device,
        use_noise_conditioning = use_noise_conditioning,
    )

    # Extract
    class_sums, class_counts = extract_embeddings(
        model       = model,
        mel         = mel,
        audio_paths = audio_paths,
        labels      = label_idxs,
        n_classes   = n_classes,
        device      = device,
        sr          = sr,
    )

    # Compute mean (= prototype for each class)
    # For classes with 0 clips, prototype = zero vector (fallback to classifier)
    counts_safe = class_counts.float().clamp(min=1).unsqueeze(1)  # (C, 1)
    prototypes  = class_sums / counts_safe                         # (C, D) mean embeddings
    prototypes  = F.normalize(prototypes, dim=1)                   # unit-normalise

    # Save
    proto_path  = str(output_dir / "class_prototypes.pt")
    counts_path = str(output_dir / "class_counts.pt")
    torch.save(prototypes.cpu(), proto_path)
    torch.save(class_counts.cpu(), counts_path)

    # Manifest
    manifest_out = {
        "backbone":   backbone,
        "fold_used":  fold,
        "n_classes":  n_classes,
        "emb_dim":    int(prototypes.shape[1]),
        "prototypes": proto_path,
        "counts":     counts_path,
    }
    with open(output_dir / "prototype_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)

    # Summary
    present   = (class_counts > 0).sum().item()
    rare      = (class_counts < 10).sum().item()
    logger.info(f"Prototypes built: {present}/{n_classes} classes have clips")
    logger.info(f"  {rare} classes have <10 clips (prototype will dominate blending)")
    logger.info(f"Saved prototypes  → {proto_path}")
    logger.info(f"Saved class counts → {counts_path}")
    return proto_path, counts_path


def main():
    p = argparse.ArgumentParser(description="Build class prototypes from trained model")
    p.add_argument("--manifest",    required=True,
                   help="Path to checkpoint_manifest.json from main_train.py")
    p.add_argument("--data_root",   required=True,
                   help="Root of BirdCLEF data (contains train.csv, train_audio/)")
    p.add_argument("--output_dir",  required=True,
                   help="Where to save class_prototypes.pt and class_counts.pt")
    p.add_argument("--backbone",    default="eca_nfnet_l0",
                   help="Backbone name (must match the trained model)")
    p.add_argument("--fold",        type=int, default=0,
                   help="Which fold's weights to use for prototype extraction")
    p.add_argument("--n_classes",   type=int, default=206)
    p.add_argument("--use_noise_conditioning", action="store_true",
                   help="Set if the model was trained with use_noise_conditioning=True")
    args = p.parse_args()

    build_prototypes_from_manifest(
        manifest_path          = args.manifest,
        data_root              = args.data_root,
        output_dir             = args.output_dir,
        backbone               = args.backbone,
        fold                   = args.fold,
        n_classes              = args.n_classes,
        use_noise_conditioning = args.use_noise_conditioning,
    )


if __name__ == "__main__":
    main()
