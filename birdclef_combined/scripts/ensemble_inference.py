"""ensemble_inference.py — Multi-checkpoint TTA ensemble inference.

Loads N Lightning .ckpt files (handles both clean and FiLM-enabled variants
by inspecting state_dict keys), runs each over an audio file with TTA
(overlapping 5-second windows), averages predictions across ckpts and TTA
positions, and returns per-chunk sigmoid probability matrices.

Used by:
  - evaluate_ensemble.py  (val/auc measurement)
  - predict_cli.py        (5-sec demo predictions on a single file)
  - streamlit_app.py      (web UI inference)

CLI usage (basic sanity check on one file):
    python scripts/ensemble_inference.py \
        --ckpts ckpt_a.ckpt ckpt_b.ckpt ckpt_c.ckpt \
        --audio /path/to/audio.wav \
        --output preds.npy

Returns: numpy array of shape (n_chunks, n_classes) with sigmoid probabilities.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchaudio


# ============================================================================
# Constants matching training config
# ============================================================================
SR              = 32_000
CHUNK_SEC       = 5.0
CHUNK_SAMPLES   = int(SR * CHUNK_SEC)
N_CLASSES       = 206
DEFAULT_TTA_HOP = 2.5   # 50% overlap → 2 inference passes per chunk
N_MELS          = 128
N_FFT           = 2_048
HOP_LENGTH      = 512
FMIN            = 20.0
TOP_DB          = 80.0


# ============================================================================
# Model loader — auto-detects FiLM presence in ckpt
# ============================================================================
@dataclass
class LoadedCkpt:
    model:       torch.nn.Module
    has_film:    bool
    path:        Path
    epoch:       int
    val_auc:     Optional[float]
    use_ema:     bool


def _detect_film(state_dict: dict) -> bool:
    """Returns True if the ckpt's head includes FiLM (gamma_net / beta_net)."""
    for k in state_dict.keys():
        if "film" in k.lower() and ("gamma" in k.lower() or "beta" in k.lower()):
            return True
    return False


def _parse_val_auc_from_filename(name: str) -> Optional[float]:
    """Extract val_auc from a filename like 'best-epoch07-auc0.7756.ckpt'."""
    import re
    m = re.search(r"auc([\d\.]+)", name)
    return float(m.group(1).rstrip(".")) if m else None


def load_ckpt(ckpt_path: str | Path, device: str = "cuda") -> LoadedCkpt:
    """Load a Lightning .ckpt, auto-detecting whether FiLM was enabled."""
    ckpt_path = Path(ckpt_path)
    sys.path.insert(0, "/workspace/major_project/birdclef_combined")
    from code_base.models.backbone import BirdCLEFModel

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["state_dict"] if "state_dict" in raw else raw
    has_film = _detect_film(state)

    model = BirdCLEFModel(
        backbone_name          = "eca_nfnet_l0",
        n_classes              = N_CLASSES,
        pretrained             = False,                      # we load weights below
        hidden_dim             = 512,
        dropout1               = 0.25,
        dropout2               = 0.50,
        gem_p                  = 3.0,
        use_sed_head           = False,
        use_noise_conditioning = has_film,
        noise_dim              = 128 if has_film else None,
        use_multi_res_mel      = False,
    )

    # Map Lightning keys ("model.<key>") → bare module keys ("<key>")
    # while filtering out EMA weights (we use main weights for ensembling
    # to avoid double-averaging across ckpts that already used EMA).
    main_keys = {}
    for k, v in state.items():
        if k.startswith("_ema_model.") or k.startswith("ema_model."):
            continue
        if k.startswith("model."):
            main_keys[k[len("model."):]] = v
        else:
            main_keys[k] = v

    # Some ckpts have a top-level noise extractor (FiLM teacher); we ignore
    # it at inference (FiLM operates on whatever noise_features the model
    # internally computes, falling back to identity if absent).
    main_keys = {k: v for k, v in main_keys.items() if not k.startswith("_noise_extractor")}

    missing, unexpected = model.load_state_dict(main_keys, strict=False)
    if len(unexpected) > 5:
        print(f"[WARN] {len(unexpected)} unexpected keys in {ckpt_path.name} "
              f"(first 3: {unexpected[:3]})")
    if len(missing) > 5:
        print(f"[WARN] {len(missing)} missing keys in {ckpt_path.name} "
              f"(first 3: {missing[:3]})")

    model = model.to(device).eval()

    return LoadedCkpt(
        model    = model,
        has_film = has_film,
        path     = ckpt_path,
        epoch    = raw.get("epoch", -1) if isinstance(raw, dict) else -1,
        val_auc  = _parse_val_auc_from_filename(ckpt_path.name),
        use_ema  = False,
    )


# ============================================================================
# Mel-spectrogram extractor (matches training)
# ============================================================================
class MelExtractor(torch.nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate = SR,
            n_fft       = N_FFT,
            hop_length  = HOP_LENGTH,
            f_min       = FMIN,
            f_max       = SR // 2,
            n_mels      = N_MELS,
            power       = 2.0,
        ).to(device)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=TOP_DB
        ).to(device)
        self.device = device

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        mel = self.spec(waveform)             # (B, n_mels, T')
        mel = self.amp_to_db(mel)             # log scale, clamped
        # Normalize per-sample to [0, 1] for backbone
        mn = mel.amin(dim=(-2, -1), keepdim=True)
        mx = mel.amax(dim=(-2, -1), keepdim=True)
        mel = (mel - mn) / (mx - mn + 1e-6)
        return mel.unsqueeze(1)               # (B, 1, n_mels, T')


# ============================================================================
# Audio loading + chunking with TTA overlap
# ============================================================================
def load_audio(path: str | Path) -> torch.Tensor:
    """Load audio, mono, resample to 32 kHz. Returns 1-D tensor."""
    wav, sr_in = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr_in != SR:
        wav = torchaudio.functional.resample(wav, sr_in, SR)
    return wav.squeeze(0)


def chunk_with_tta(waveform: torch.Tensor, tta_hop_sec: float = DEFAULT_TTA_HOP) -> list:
    """
    Split waveform into overlapping 5-sec chunks for TTA.

    Each "primary chunk" (aligned with non-overlapping 5-sec grid) collects
    all TTA windows whose centre falls within it. Returns a list of dicts:
        [{"primary_start": float, "tta_windows": [(start_sample, end_sample), ...]}, ...]

    Short files get padded to one chunk. Long files get N primary chunks with
    TTA windows from positions [0, hop, 2*hop, ...].
    """
    n_samples = waveform.shape[0]
    if n_samples < CHUNK_SAMPLES:
        # Pad short audio to one chunk
        pad = CHUNK_SAMPLES - n_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad))
        return [{"primary_start": 0.0, "tta_windows": [(0, CHUNK_SAMPLES)]}], waveform

    hop_samples = int(tta_hop_sec * SR)
    n_primary   = max(1, (n_samples - CHUNK_SAMPLES) // CHUNK_SAMPLES + 1)

    primary_chunks = []
    for i in range(n_primary):
        primary_start = i * CHUNK_SAMPLES
        windows = []
        # All TTA windows whose centre falls within [primary_start, primary_start + CHUNK_SAMPLES)
        offset = 0
        while True:
            ws = primary_start + offset
            we = ws + CHUNK_SAMPLES
            if we > n_samples or ws >= n_samples:
                break
            centre = ws + CHUNK_SAMPLES // 2
            if centre >= primary_start and centre < primary_start + CHUNK_SAMPLES:
                windows.append((ws, we))
            offset += hop_samples
            if offset >= CHUNK_SAMPLES:
                break
        # Backwards TTA windows (negative offset)
        offset = -hop_samples
        while True:
            ws = primary_start + offset
            we = ws + CHUNK_SAMPLES
            if ws < 0:
                break
            centre = ws + CHUNK_SAMPLES // 2
            if centre >= primary_start and centre < primary_start + CHUNK_SAMPLES:
                windows.append((ws, we))
            offset -= hop_samples
        if not windows:
            windows = [(primary_start, primary_start + CHUNK_SAMPLES)]
        primary_chunks.append({
            "primary_start": primary_start / SR,
            "tta_windows":   sorted(set(windows)),
        })

    return primary_chunks, waveform


# ============================================================================
# Ensemble inference — runs N ckpts on one audio file
# ============================================================================
@torch.no_grad()
def ensemble_predict(
    audio_path:        str | Path,
    ckpts:             List[LoadedCkpt],
    device:            str  = "cuda",
    tta_hop_sec:       float = DEFAULT_TTA_HOP,
    batch_size:        int  = 8,
    noise_preprocess:  bool = False,
    consistency_vote_boost: float = 0.0,
) -> np.ndarray:
    """
    Returns: array of shape (n_primary_chunks, n_classes) with sigmoid probs
             averaged across all ckpts and TTA windows.

    Args:
        noise_preprocess: If True, apply NoiseRobustProcessor (spectral
            gating + bandpass + pre-emphasis) before mel extraction.
            Improves real-world demo performance on noisy audio.
        consistency_vote_boost: If > 0, boost confidence on classes that
            appear in adjacent chunks' top-3. 0.05-0.1 typically works.
    """
    wav = load_audio(audio_path)

    if noise_preprocess:
        from noise_robust_preprocessing import NoiseRobustProcessor
        processor = NoiseRobustProcessor()
        wav = processor.process(wav, sr=SR)

    chunks, wav = chunk_with_tta(wav, tta_hop_sec=tta_hop_sec)
    wav = wav.to(device)

    mel_extractor = MelExtractor(device=device)

    # Collect all TTA window samples for vectorised forward
    all_windows = []
    chunk_ranges = []
    for ci, c in enumerate(chunks):
        start_idx = len(all_windows)
        for (ws, we) in c["tta_windows"]:
            all_windows.append(wav[ws:we])
        chunk_ranges.append((start_idx, len(all_windows)))

    win_tensor = torch.stack(all_windows, dim=0).to(device)   # (N_win, CHUNK_SAMPLES)
    n_win = win_tensor.shape[0]

    # Per-ckpt probabilities, averaged across batches
    ckpt_probs = []   # list of (n_win, n_classes) tensors
    for ckpt in ckpts:
        win_probs = []
        for i in range(0, n_win, batch_size):
            batch_wav = win_tensor[i:i + batch_size]
            mel = mel_extractor(batch_wav)               # (B, 1, n_mels, T')
            # Model.forward(mel, noise_profile=None) works for both FiLM
            # (identity FiLM when profile=None) and clean variants.
            logits = ckpt.model(mel, None)
            if isinstance(logits, dict):
                logits = logits.get("clip_logits", logits.get("logits"))
            probs = torch.sigmoid(logits)                 # (B, n_classes)
            win_probs.append(probs.cpu())
        ckpt_probs.append(torch.cat(win_probs, dim=0))   # (N_win, n_classes)

    # Average across ckpts
    win_probs = torch.stack(ckpt_probs, dim=0).mean(dim=0)   # (N_win, n_classes)

    # Average TTA windows within each primary chunk
    primary_probs = []
    for (a, b) in chunk_ranges:
        primary_probs.append(win_probs[a:b].mean(dim=0))
    probs = torch.stack(primary_probs, dim=0).numpy()

    if consistency_vote_boost > 0 and probs.shape[0] >= 2:
        from noise_robust_preprocessing import consistency_vote
        probs = consistency_vote(
            probs,
            consistency_boost = consistency_vote_boost,
            agreement_window  = 3,
        )
    return probs


# ============================================================================
# CLI entry
# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts",    nargs="+", required=True,
                   help="One or more .ckpt paths to ensemble.")
    p.add_argument("--audio",    required=True, help="Audio file to predict on.")
    p.add_argument("--output",   default=None,
                   help="Save predictions to .npy (default: print top-5 per chunk).")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--tta_hop",  type=float, default=DEFAULT_TTA_HOP)
    p.add_argument("--top_k",    type=int,   default=5)
    args = p.parse_args()

    print(f"loading {len(args.ckpts)} ckpts:")
    loaded = []
    for c in args.ckpts:
        lc = load_ckpt(c, device=args.device)
        loaded.append(lc)
        film_tag = "FiLM" if lc.has_film else "clean"
        print(f"  - {Path(c).name}  [{film_tag}]  val_auc={lc.val_auc}")
    print()

    print(f"predicting on: {args.audio}")
    probs = ensemble_predict(
        audio_path  = args.audio,
        ckpts       = loaded,
        device      = args.device,
        tta_hop_sec = args.tta_hop,
    )
    print(f"output shape: {probs.shape}  (n_chunks, n_classes)")

    if args.output:
        np.save(args.output, probs)
        print(f"saved: {args.output}")
    else:
        # Print top-K per chunk
        print()
        print(f"=== top-{args.top_k} per 5-second chunk ===")
        for i in range(probs.shape[0]):
            order = np.argsort(-probs[i])[:args.top_k]
            top = [f"cls{c:>3}={probs[i, c]:.3f}" for c in order]
            print(f"  chunk {i:>3} ({i*5:>3}-{(i+1)*5:>3}s):  {'  '.join(top)}")


if __name__ == "__main__":
    sys.exit(main())
