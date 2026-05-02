"""
visualize_mel_spectrograms.py — Generate mel spectrogram figures for the paper
================================================================================
Reproduces the mel spectrogram visualizations shown in 1st/2nd place writeups.

What is a mel spectrogram?
  The raw audio waveform (1D signal) is transformed into a 2D image:
    X-axis = time (seconds)
    Y-axis = frequency (mel-scaled Hz bands)
    Color  = amplitude (dB) — brighter = louder

  This is what the CNN backbone actually "sees" as input.

Usage:
  python scripts/visualize_mel_spectrograms.py --data_root D:/birdclef_data/birdclef-2025
  python scripts/visualize_mel_spectrograms.py --data_root D:/birdclef_data/birdclef-2025 --species grekis trokin
  python scripts/visualize_mel_spectrograms.py --data_root D:/birdclef_data/birdclef-2025 --multi_res
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import librosa
    import librosa.display
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False


SR = 32_000
DURATION = 5.0
N_SAMPLES = int(SR * DURATION)


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    if SF_OK:
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr and LIBROSA_OK:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    elif LIBROSA_OK:
        audio, _ = librosa.load(path, sr=sr, mono=True)
    else:
        raise ImportError("Need soundfile or librosa")
    return audio


def random_chunk(audio: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    if len(audio) <= n:
        return np.pad(audio, (0, max(0, n - len(audio))))[:n]
    start = random.randint(0, len(audio) - n)
    return audio[start:start + n]


def compute_mel(audio: np.ndarray, sr=SR, n_fft=2048, hop=512, n_mels=128):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, fmin=20, fmax=sr // 2,
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    return S_db


def plot_single_species(audio_root, df, species, output_dir, n_examples=3):
    sp_df = df[df["primary_label"] == species]
    if len(sp_df) == 0:
        print(f"  Species '{species}' not found, skipping")
        return

    tax_info = ""
    row0 = sp_df.iloc[0]
    sci_name = row0.get("scientific_name", species)
    common = row0.get("common_name", "")

    files = sp_df["filename"].tolist()
    random.shuffle(files)
    files = files[:n_examples]

    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 3.5))
    if len(files) == 1:
        axes = [axes]

    for i, fname in enumerate(files):
        fpath = audio_root / fname
        if not fpath.exists():
            axes[i].text(0.5, 0.5, "File not found", ha="center", va="center")
            continue

        audio = load_audio(str(fpath))
        chunk = random_chunk(audio)
        mel = compute_mel(chunk)

        librosa.display.specshow(
            mel, sr=SR, hop_length=512, x_axis="time", y_axis="mel",
            ax=axes[i], cmap="magma",
        )
        axes[i].set_title(Path(fname).stem, fontsize=8)
        axes[i].set_xlabel("")
        if i > 0:
            axes[i].set_ylabel("")

    fig.suptitle(f"{species} — {sci_name} ({common})\n{len(sp_df)} recordings",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = output_dir / f"mel_{species}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_multi_resolution(audio_root, df, species, output_dir):
    sp_df = df[df["primary_label"] == species]
    if len(sp_df) == 0:
        return

    fname = sp_df.iloc[0]["filename"]
    fpath = audio_root / fname
    if not fpath.exists():
        return

    audio = load_audio(str(fpath))
    chunk = random_chunk(audio)

    resolutions = [
        ("Fine (n_fft=1024, hop=256)\nInsects: transient stridulation", 1024, 256),
        ("Medium (n_fft=2048, hop=512)\nBirds: structured calls", 2048, 512),
        ("Coarse (n_fft=4096, hop=1024)\nAmphibians: tonal calls", 4096, 1024),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (title, n_fft, hop) in enumerate(resolutions):
        mel = compute_mel(chunk, n_fft=n_fft, hop=hop)
        librosa.display.specshow(
            mel, sr=SR, hop_length=hop, x_axis="time", y_axis="mel",
            ax=axes[i], cmap="magma",
        )
        axes[i].set_title(title, fontsize=9)
        if i > 0:
            axes[i].set_ylabel("")

    sci = sp_df.iloc[0].get("scientific_name", species)
    fig.suptitle(f"Multi-Resolution Mel: {species} ({sci})\nNovel #8 — 3 resolutions fused as 3-channel input",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = output_dir / f"multi_res_mel_{species}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_taxonomy_comparison(audio_root, df, tax_df, output_dir):
    taxa = {"Aves": None, "Amphibia": None, "Insecta": None, "Mammalia": None}
    for taxon in taxa:
        sp_list = tax_df[tax_df["class_name"] == taxon]["primary_label"].tolist()
        sp_counts = df[df["primary_label"].isin(sp_list)].groupby("primary_label").size()
        if len(sp_counts) > 0:
            taxa[taxon] = sp_counts.idxmax()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, (taxon, sp) in enumerate(taxa.items()):
        if sp is None:
            axes[i].text(0.5, 0.5, f"No {taxon}", ha="center")
            continue

        sp_df = df[df["primary_label"] == sp]
        fname = sp_df.iloc[0]["filename"]
        fpath = audio_root / fname
        if not fpath.exists():
            continue

        audio = load_audio(str(fpath))
        chunk = random_chunk(audio)
        mel = compute_mel(chunk)

        librosa.display.specshow(
            mel, sr=SR, hop_length=512, x_axis="time", y_axis="mel",
            ax=axes[i], cmap="magma",
        )
        sci = sp_df.iloc[0].get("scientific_name", sp)
        n = len(sp_df)
        axes[i].set_title(f"{taxon}\n{sp} ({n} clips)\n{sci}", fontsize=8)
        if i > 0:
            axes[i].set_ylabel("")

    fig.suptitle("Mel Spectrograms Across Taxonomic Groups — BirdCLEF+ 2025",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "taxonomy_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_class_distribution(df, tax_df, output_dir):
    merged = df.merge(tax_df[["primary_label", "class_name"]], on="primary_label", how="left")
    merged["class_name"] = merged["class_name"].fillna("Unknown")

    counts = df.groupby("primary_label").size().sort_values()
    tax_map = dict(zip(tax_df["primary_label"].astype(str), tax_df["class_name"]))
    colors = {"Aves": "#4CAF50", "Amphibia": "#2196F3", "Insecta": "#FF9800", "Mammalia": "#9C27B0", "Unknown": "#999"}

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = [colors.get(tax_map.get(str(sp), "Unknown"), "#999") for sp in counts.index]
    ax.bar(range(len(counts)), counts.values, color=bar_colors, width=1.0)
    ax.set_xlabel("Species (sorted by count)")
    ax.set_ylabel("Number of recordings")
    ax.set_title("BirdCLEF+ 2025: Class Imbalance Across 206 Species", fontweight="bold")
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="min_oversample=100")
    ax.axhline(y=10, color="orange", linestyle="--", alpha=0.5, label="<10 clips (39 species)")

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=f"{n} ({sum(1 for v in tax_map.values() if v==n)})")
              for n, c in colors.items() if n != "Unknown"]
    legend.append(plt.Line2D([0], [0], color="red", linestyle="--", label="min_oversample=100"))
    ax.legend(handles=legend, fontsize=8)
    ax.set_yscale("log")

    plt.tight_layout()
    out = output_dir / "class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="D:/birdclef_data/birdclef-2025")
    parser.add_argument("--output_dir", type=str, default="paper_figures")
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--multi_res", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    assert LIBROSA_OK, "pip install librosa"

    root = Path(args.data_root)
    df = pd.read_csv(root / "train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    audio_root = root / "train_audio"

    tax_path = root / "taxonomy.csv"
    tax_df = pd.read_csv(tax_path) if tax_path.exists() else None
    if tax_df is not None:
        tax_df["primary_label"] = tax_df["primary_label"].astype(str)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution plot
    if tax_df is not None:
        print("Generating class distribution plot...")
        plot_class_distribution(df, tax_df, output_dir)

    # 2. Taxonomy comparison (one example per taxon)
    if tax_df is not None:
        print("Generating taxonomy comparison...")
        plot_taxonomy_comparison(audio_root, df, tax_df, output_dir)

    # 3. Per-species mel spectrograms
    if args.species:
        species_list = args.species
    elif args.all:
        species_list = df["primary_label"].unique().tolist()
    else:
        counts = df["primary_label"].value_counts()
        most_common = counts.head(3).index.tolist()
        rarest = counts.tail(3).index.tolist()
        species_list = most_common + rarest

    print(f"Generating mel spectrograms for {len(species_list)} species...")
    for sp in species_list:
        plot_single_species(audio_root, df, sp, output_dir)

    # 4. Multi-resolution comparison (Novel #8)
    if args.multi_res or args.all:
        if tax_df is not None:
            for taxon in ["Aves", "Insecta", "Amphibia"]:
                sp_list = tax_df[tax_df["class_name"] == taxon]["primary_label"].tolist()
                sp_counts = df[df["primary_label"].isin(sp_list)].groupby("primary_label").size()
                if len(sp_counts) > 0:
                    best = sp_counts.idxmax()
                    print(f"Multi-res mel for {taxon} ({best})...")
                    plot_multi_resolution(audio_root, df, best, output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
