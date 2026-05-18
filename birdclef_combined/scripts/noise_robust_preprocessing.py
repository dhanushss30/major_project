"""noise_robust_preprocessing.py — Audio preprocessing for real-world robustness.

Inference-time tricks that compensate for the focal→soundscape domain gap:
  - Spectral noise gating (subtract estimated noise floor)
  - Bandpass filter (focus on bird call frequencies, 1-8 kHz)
  - Pre-emphasis filter (boost high frequencies where bird calls live)
  - Per-chunk consistency voting (if adjacent chunks agree, boost confidence)

These run on CPU, are applied OUTSIDE the model, and add no training cost.
They directly improve real-world demo performance on noisy audio.

Usage (programmatic):
    from noise_robust_preprocessing import NoiseRobustProcessor

    processor = NoiseRobustProcessor(
        enable_spectral_gating  = True,
        enable_bandpass         = True,
        enable_pre_emphasis     = True,
        noise_gate_threshold_db = -40,
    )
    audio_clean = processor.process(audio_raw, sr=32_000)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ============================================================================
# Bandpass + pre-emphasis (lightweight, no FFT needed)
# ============================================================================
def pre_emphasis(audio: torch.Tensor, coef: float = 0.97) -> torch.Tensor:
    """y[n] = x[n] - coef * x[n-1].  Boosts high frequencies."""
    if audio.dim() == 1:
        out = torch.empty_like(audio)
        out[0] = audio[0]
        out[1:] = audio[1:] - coef * audio[:-1]
        return out
    out = torch.empty_like(audio)
    out[..., 0] = audio[..., 0]
    out[..., 1:] = audio[..., 1:] - coef * audio[..., :-1]
    return out


def butter_bandpass_torchaudio(
    audio:    torch.Tensor,
    sr:       int   = 32_000,
    low_hz:   float = 1_000,
    high_hz:  float = 8_000,
    order:    int   = 4,
) -> torch.Tensor:
    """Simple bandpass via biquad cascade. Works in PyTorch end-to-end.

    For broader robustness we just use a smooth approximation: high-pass at
    low_hz, low-pass at high_hz, in sequence.
    """
    try:
        import torchaudio.functional as taF
        # 2-pass band-pass biquad
        y = taF.highpass_biquad(audio.float(), sample_rate=sr, cutoff_freq=low_hz)
        y = taF.lowpass_biquad(y, sample_rate=sr, cutoff_freq=high_hz)
        return y
    except Exception:
        return audio   # graceful fallback if torchaudio missing


# ============================================================================
# Spectral noise gating
# ============================================================================
def spectral_noise_gate(
    audio:                torch.Tensor,
    sr:                   int   = 32_000,
    n_fft:                int   = 2_048,
    hop_length:           int   = 512,
    noise_quantile:       float = 0.2,
    over_subtraction:     float = 1.5,
    spectral_floor_ratio: float = 0.1,
) -> torch.Tensor:
    """Estimate noise floor from quietest frames, subtract from spectrogram.

    Standard spectral subtraction with oversubtraction factor for cleaner
    speech/audio. Tuned for bird-call settings:
      - noise_quantile=0.2 means "the quietest 20% of frames are noise"
      - over_subtraction=1.5 removes a bit MORE than estimated noise
      - spectral_floor_ratio=0.1 prevents negative spectrum (musical noise)
    """
    if audio.dim() == 1:
        audio_b = audio.unsqueeze(0)
        squeeze_back = True
    else:
        audio_b = audio
        squeeze_back = False

    # STFT
    win = torch.hann_window(n_fft, device=audio_b.device, dtype=audio_b.dtype)
    spec = torch.stft(
        audio_b.float(),
        n_fft        = n_fft,
        hop_length   = hop_length,
        window       = win,
        return_complex = True,
    )                                     # (B, F, T)
    mag = spec.abs()
    phase = spec / (mag + 1e-10)

    # Estimate noise: per-bin quantile across time
    noise_estimate = torch.quantile(mag, noise_quantile, dim=-1, keepdim=True)

    # Subtract with oversubtraction + spectral floor
    cleaned_mag = mag - over_subtraction * noise_estimate
    spectral_floor = spectral_floor_ratio * mag
    cleaned_mag = torch.maximum(cleaned_mag, spectral_floor)

    cleaned_spec = cleaned_mag * phase
    out = torch.istft(
        cleaned_spec,
        n_fft       = n_fft,
        hop_length  = hop_length,
        window      = win,
        length      = audio_b.shape[-1],
    )
    if squeeze_back:
        out = out.squeeze(0)
    return out


# ============================================================================
# Main processor class
# ============================================================================
@dataclass
class NoiseRobustProcessor:
    """Composite preprocessor applied to raw audio before mel-spec extraction."""
    enable_spectral_gating:  bool  = True
    enable_bandpass:         bool  = True
    enable_pre_emphasis:     bool  = True

    # Spectral gating params
    noise_quantile:          float = 0.2
    over_subtraction:        float = 1.5
    spectral_floor_ratio:    float = 0.1

    # Bandpass params (bird call range)
    bandpass_low_hz:         float = 1_000
    bandpass_high_hz:        float = 8_000

    # Pre-emphasis coefficient
    pre_emphasis_coef:       float = 0.97

    def process(self, audio: torch.Tensor, sr: int = 32_000) -> torch.Tensor:
        """Apply enabled steps in order. Returns same-shape tensor."""
        x = audio
        if self.enable_spectral_gating:
            x = spectral_noise_gate(
                x, sr=sr,
                noise_quantile       = self.noise_quantile,
                over_subtraction     = self.over_subtraction,
                spectral_floor_ratio = self.spectral_floor_ratio,
            )
        if self.enable_bandpass:
            x = butter_bandpass_torchaudio(
                x, sr=sr,
                low_hz  = self.bandpass_low_hz,
                high_hz = self.bandpass_high_hz,
            )
        if self.enable_pre_emphasis:
            x = pre_emphasis(x, coef=self.pre_emphasis_coef)
        # Re-normalize amplitude to avoid clipping after processing
        peak = x.abs().max()
        if peak > 1e-6:
            x = x * (0.95 / peak)
        return x


# ============================================================================
# Per-chunk consistency voting
# ============================================================================
def consistency_vote(
    chunk_probs:        np.ndarray,     # (n_chunks, n_classes)
    consistency_boost:  float = 0.1,
    agreement_window:   int   = 3,
) -> np.ndarray:
    """Boost confidence on classes that recur in adjacent chunks.

    Logic: if chunks i, i+1, i+2 all have class X in their top-3,
    that's a stronger signal than X appearing in one isolated chunk.
    Apply a small additive boost to consistently-predicted classes.
    """
    if chunk_probs.shape[0] < 2:
        return chunk_probs   # not enough chunks for consistency check

    out = chunk_probs.copy()
    n_chunks, n_classes = chunk_probs.shape

    # For each chunk, which top-3 classes appear?
    top3 = np.argsort(-chunk_probs, axis=1)[:, :3]   # (n_chunks, 3)

    for i in range(n_chunks):
        lo = max(0, i - agreement_window // 2)
        hi = min(n_chunks, i + agreement_window // 2 + 1)
        if hi - lo < 2:
            continue
        # Count how many neighbouring chunks include each class in top-3
        neighbours_top3 = top3[lo:hi].flatten()
        unique, counts = np.unique(neighbours_top3, return_counts=True)
        # Boost classes that appear in ≥ 2 neighbouring chunks
        for cls, cnt in zip(unique, counts):
            if cnt >= 2:
                out[i, cls] = min(1.0, out[i, cls] + consistency_boost * (cnt - 1))
    return out


# ============================================================================
# CLI sanity test
# ============================================================================
def main():
    import argparse
    import torchaudio

    p = argparse.ArgumentParser(description="Sanity-test noise preprocessing.")
    p.add_argument("--audio",  required=True)
    p.add_argument("--output", default=None,
                   help="If set, write the processed audio to this path.")
    args = p.parse_args()

    wav, sr_in = torchaudio.load(args.audio)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr_in != 32_000:
        wav = torchaudio.functional.resample(wav, sr_in, 32_000)
    wav = wav.squeeze(0)

    print(f"input:  shape={tuple(wav.shape)}  duration={len(wav)/32_000:.2f}s")
    print(f"        amp_peak={wav.abs().max():.4f}  amp_rms={wav.pow(2).mean().sqrt():.4f}")

    processor = NoiseRobustProcessor()
    out = processor.process(wav, sr=32_000)
    print(f"output: shape={tuple(out.shape)}")
    print(f"        amp_peak={out.abs().max():.4f}  amp_rms={out.pow(2).mean().sqrt():.4f}")

    if args.output:
        torchaudio.save(args.output, out.unsqueeze(0), 32_000)
        print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
