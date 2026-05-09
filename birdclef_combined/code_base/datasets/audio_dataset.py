"""
audio_dataset.py — Training & Soundscape Datasets
====================================================
Implements exact 2nd place data pipeline:

1. HDF5 byte-wise random 5-sec chunk access (h5py, Appendix B)
2. Sampling strategy: w_i = (c_i / Σc_j)^γ  with γ=-0.5 (NFNet) or γ=-1 (EffNetV2)
3. Secondary labels from Xeno-Canto treated as equal-weight positive labels
4. Random 5-sec segment selection with "first/last 7s" strategy for common species
5. Label smoothing: ỹ_i = y_i(1-α) + α/K  with α=0.05
6. MixUp in audio domain (AudioMixUp class called from collate)
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────

SAMPLE_RATE    = 32_000
CHUNK_DURATION = 5.0
CHUNK_SAMPLES  = int(CHUNK_DURATION * SAMPLE_RATE)  # 160,000


# ─── HDF5 audio reader ────────────────────────────────────────────────────

class HDF5AudioReader:
    """
    Byte-wise random 5-second chunk reader from HDF5 files.
    From paper Appendix B: "audio files were converted to h5py format,
    enabling byte-wise access to random 5-second chunks."
    """

    def __init__(self, hdf5_path: str, cache: bool = True):
        self.path  = hdf5_path
        self.cache = cache
        self._fd: Optional[h5py.File] = None

        with h5py.File(hdf5_path, "r") as f:
            self.sr        = int(f.attrs.get("sample_rate", SAMPLE_RATE))
            self.n_samples = int(f.attrs.get("n_samples",   len(f["audio"])))
            self.duration  = self.n_samples / self.sr

    def _open(self) -> h5py.File:
        if self.cache:
            if self._fd is None or not self._fd.id.valid:
                self._fd = h5py.File(self.path, "r")
            return self._fd
        return h5py.File(self.path, "r")

    def read_random_chunk(
        self,
        duration: float    = CHUNK_DURATION,
        strategy: str      = "random",   # "random", "first7", "firstlast7", "full"
    ) -> np.ndarray:
        """
        Select a 5-second chunk using the paper's strategy:
        "random 5 seconds from the first or last 7 seconds"

        Strategies:
          "random"      : uniform random over entire file
          "first7"      : random within first 7 seconds (≈ animal starts vocalising)
          "firstlast7"  : random within first OR last 7 seconds
          "full"        : entire file (pad/trim to exact duration)
        """
        n_needed = int(duration * self.sr)

        if self.n_samples <= n_needed:
            fd    = self._open()
            audio = fd["audio"][:]
            if not self.cache:
                fd.close()
            return self._pad(audio, n_needed)

        window_samples = int(7.0 * self.sr)   # 7-second boundary

        if strategy == "first7":
            max_start = min(window_samples, self.n_samples - n_needed)
            start     = random.randint(0, max(0, max_start))

        elif strategy == "firstlast7":
            use_last = random.random() < 0.5
            if use_last:
                first_start = max(0, self.n_samples - window_samples)
            else:
                first_start = 0
            max_start = min(first_start + window_samples, self.n_samples - n_needed)
            start     = random.randint(first_start, max(first_start, max_start))

        else:  # "random"
            start = random.randint(0, self.n_samples - n_needed)

        fd    = self._open()
        audio = fd["audio"][start: start + n_needed]
        if not self.cache:
            fd.close()

        return self._pad(audio.astype(np.float32), n_needed)

    @staticmethod
    def _pad(audio: np.ndarray, n: int) -> np.ndarray:
        if len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)), mode="constant")
        return audio[:n].astype(np.float32)

    def close(self):
        if self._fd is not None and self._fd.id.valid:
            self._fd.close()


# ─── Main training dataset ────────────────────────────────────────────────

class BirdCLEFDataset(Dataset):
    """
    Main training dataset with exact 2nd-place implementation.

    Key features:
      - HDF5 byte-wise loading with fallback to raw audio
      - Secondary label support (equal weight to primary, Xeno-Canto)
      - Label smoothing α=0.05
      - "firstlast7" chunk strategy for common species
      - Stores audio as numpy (MixUp applied in collate_fn)
    """

    def __init__(
        self,
        df:               pd.DataFrame,
        hdf5_root:        Optional[str] = None,
        audio_root:       Optional[str] = None,
        label_to_idx:     Optional[Dict[str, int]] = None,
        n_classes:        int           = 206,
        sr:               int           = SAMPLE_RATE,
        chunk_duration:   float         = CHUNK_DURATION,
        label_smoothing:  float         = 0.05,      # α from paper Eq. 2
        chunk_strategy:   str           = "firstlast7",  # paper §5.5
        use_secondary:    bool          = True,       # XC secondary labels
        augment:          bool          = True,
        is_train:         bool          = True,
    ):
        self.df              = df.reset_index(drop=True)
        self.hdf5_root       = Path(hdf5_root) if hdf5_root else None
        self.audio_root      = Path(audio_root) if audio_root else None
        self.n_classes       = n_classes
        self.sr              = sr
        self.chunk_samples   = int(chunk_duration * sr)
        self.label_smoothing = label_smoothing
        self.chunk_strategy  = chunk_strategy if is_train else "random"
        self.use_secondary   = use_secondary
        self.augment         = augment
        self.is_train        = is_train

        # Build label→index mapping
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            all_labels = sorted(df["primary_label"].unique())
            self.label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # HDF5 reader cache
        self._hdf5_readers: Dict[str, HDF5AudioReader] = {}

    # ── Audio loading ──────────────────────────────────────────────────────

    def _get_hdf5_reader(self, hdf5_path: str) -> HDF5AudioReader:
        if hdf5_path not in self._hdf5_readers:
            self._hdf5_readers[hdf5_path] = HDF5AudioReader(hdf5_path)
        return self._hdf5_readers[hdf5_path]

    def _load_audio(self, row: pd.Series) -> np.ndarray:
        filename = row.get("filename", row.get("filepath", ""))

        # Try HDF5 first
        if self.hdf5_root is not None:
            hdf5_path = str(self.hdf5_root / Path(filename).with_suffix(".hdf5"))
            if Path(hdf5_path).exists():
                reader = self._get_hdf5_reader(hdf5_path)
                return reader.read_random_chunk(
                    duration = self.chunk_samples / self.sr,
                    strategy = self.chunk_strategy if self.is_train else "random",
                )

        # Multi-year support: each row may have its own audio_root (prior year data)
        row_audio_root = row.get("audio_root", None)
        if row_audio_root and isinstance(row_audio_root, str):
            audio_path = Path(row_audio_root) / filename
            if audio_path.exists():
                return self._load_raw_audio(str(audio_path))

        # Fallback: global audio_root from config
        if self.audio_root is not None:
            audio_path = self.audio_root / filename
            if audio_path.exists():
                return self._load_raw_audio(str(audio_path))

        return np.zeros(self.chunk_samples, dtype=np.float32)

    def _load_raw_audio(self, path: str) -> np.ndarray:
        try:
            if SOUNDFILE_AVAILABLE:
                audio, sr = sf.read(path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
            elif LIBROSA_AVAILABLE:
                audio, sr = librosa.load(path, sr=None, mono=True)
            else:
                return np.zeros(self.chunk_samples, dtype=np.float32)

            if sr != self.sr and LIBROSA_AVAILABLE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

            if len(audio) > self.chunk_samples:
                start = random.randint(0, len(audio) - self.chunk_samples)
                audio = audio[start: start + self.chunk_samples]
            elif len(audio) < self.chunk_samples:
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))

            return audio.astype(np.float32)
        except Exception:
            return np.zeros(self.chunk_samples, dtype=np.float32)

    # ── Label construction ─────────────────────────────────────────────────

    def _make_label(self, row: pd.Series) -> np.ndarray:
        """
        Multi-hot label with label smoothing (Eq. 2 from paper):
            ỹ_i = y_i * (1-α) + α/K
        where α=0.05, K=206.

        Secondary labels (Xeno-Canto) are treated with equal weight.
        """
        y = np.zeros(self.n_classes, dtype=np.float32)

        # Primary label
        prim = row.get("primary_label", "")
        if prim in self.label_to_idx:
            y[self.label_to_idx[prim]] = 1.0

        # Secondary labels (XC background species)
        if self.use_secondary:
            sec_raw = row.get("secondary_labels", "")
            if isinstance(sec_raw, str) and sec_raw.strip():
                for lbl in sec_raw.strip("[]").replace("'", "").split():
                    lbl = lbl.strip(",")
                    if lbl and lbl in self.label_to_idx:
                        y[self.label_to_idx[lbl]] = 1.0

        # Label smoothing
        if self.label_smoothing > 0:
            y = y * (1 - self.label_smoothing) + self.label_smoothing / self.n_classes

        return y

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row   = self.df.iloc[idx]
        audio = self._load_audio(row)
        label = self._make_label(row)

        return {
            "audio":    torch.tensor(audio, dtype=torch.float32),
            "label":    torch.tensor(label, dtype=torch.float32),
            "idx":      idx,
            "filename": str(row.get("filename", "")),
        }


# ─── Balanced sampler (exact 2nd-place Eq. 1) ────────────────────────────

class BalancedSampler(Sampler):
    """
    From paper Eq. 1:  w_i = (c_i / Σc_j)^γ

    γ = -0.5  → SqrtBalancing  (NFNet default)
    γ = -1.0  → Equal/Balanced (EffNetV2 default)

    Also supports: duplicate rare classes to min_count threshold.
    From paper §5.2:
    "For NFNet: all classes with fewer than 100 samples were duplicated
     until they reached this threshold."

    oversampling_map: Dict[str, int] — species → target count. Species with
    fewer samples than the target are upweighted to reach it. From 2nd place
    GitHub: 64 rare species get minor oversampling (10–96 extra draws).
    """

    def __init__(
        self,
        df:                pd.DataFrame,
        gamma:             float         = -0.5,
        n_samples:         Optional[int] = None,
        replacement:       bool          = True,
        oversampling_map:  Optional[Dict[str, int]] = None,
        min_count:         int           = 0,
        use_rating_weight: bool          = False,
    ):
        self.gamma       = gamma
        self.n_samples   = n_samples or len(df)
        self.replacement = replacement

        label_col = "primary_label" if "primary_label" in df.columns else "label"
        counts    = df[label_col].value_counts().to_dict()

        has_rating = use_rating_weight and "rating" in df.columns

        total = sum(counts.values())
        weights = []
        for i in range(len(df)):
            row   = df.iloc[i]
            lbl   = row[label_col]
            c_i   = counts.get(lbl, 1)
            w_i   = (c_i / total) ** gamma

            if oversampling_map and lbl in oversampling_map:
                target = oversampling_map[lbl]
                if c_i < target:
                    w_i *= target / c_i
            elif min_count > 0 and c_i < min_count:
                w_i *= min_count / c_i

            if has_rating:
                r = float(row.get("rating", 0))
                w_i *= 0.5 + 0.1 * min(r, 5.0)

            weights.append(w_i)

        w = torch.tensor(weights, dtype=torch.float64)
        self.weights = w / w.sum()

    def __iter__(self):
        return iter(
            torch.multinomial(
                self.weights,
                num_samples = self.n_samples,
                replacement = self.replacement,
            ).tolist()
        )

    def __len__(self) -> int:
        return self.n_samples


# ─── Soundscape dataset for pseudo-label generation ───────────────────────

class SoundscapeDataset(Dataset):
    """
    Dataset for unlabeled soundscape files.
    Non-overlapping 5-second windows (Appendix B):
    "Inference on soundscapes was performed using non-overlapping 5-second windows."

    Supports OOF (out-of-fold) pseudo-labeling strategy from §5.4:
    fold_id is assigned to each soundscape file for fold-based filtering.
    """

    def __init__(
        self,
        soundscape_dir:   str,
        sr:               int   = SAMPLE_RATE,
        chunk_duration:   float = CHUNK_DURATION,
        fold_id:          Optional[int] = None,   # For OOF filtering
        n_folds:          int   = 5,
        extensions:       Tuple = (".ogg", ".mp3", ".wav", ".flac"),
    ):
        self.sr            = sr
        self.chunk_samples = int(chunk_duration * sr)
        self.fold_id       = fold_id

        # Discover files
        root   = Path(soundscape_dir)
        files  = []
        for ext in extensions:
            files.extend(sorted(root.rglob(f"*{ext}")))

        # Assign deterministic fold IDs to files
        self.index: List[Tuple[Path, int, float]] = []  # (file, file_fold, start_sec)

        for file_idx, fpath in enumerate(files):
            file_fold = file_idx % n_folds
            if fold_id is not None and file_fold != fold_id:
                continue

            try:
                if SOUNDFILE_AVAILABLE:
                    info     = sf.info(str(fpath))
                    n_samps  = int(info.duration * sr)
                else:
                    n_samps  = self.chunk_samples * 12  # Assume 1min

                pos = 0
                while pos + self.chunk_samples <= n_samps:
                    self.index.append((fpath, file_fold, pos / sr))
                    pos += self.chunk_samples
            except Exception:
                pass

    def _load_chunk(self, fpath: Path, start_sec: float) -> np.ndarray:
        start_sample = int(start_sec * self.sr)
        try:
            if SOUNDFILE_AVAILABLE:
                audio, sr = sf.read(
                    str(fpath),
                    start    = start_sample,
                    frames   = self.chunk_samples,
                    dtype    = "float32",
                    always_2d = False,
                )
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != self.sr and LIBROSA_AVAILABLE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            else:
                audio = np.zeros(self.chunk_samples, dtype=np.float32)

            if len(audio) < self.chunk_samples:
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
            return audio[:self.chunk_samples].astype(np.float32)
        except Exception:
            return np.zeros(self.chunk_samples, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        fpath, file_fold, start_sec = self.index[idx]
        audio = self._load_chunk(fpath, start_sec)

        return {
            "audio":     torch.tensor(audio, dtype=torch.float32),
            "filename":  fpath.name,
            "start_sec": start_sec,
            "file_fold": file_fold,
            "chunk_idx": idx,
        }


# ─── Pseudo-label dataset (50/50 labeled + pseudo) ────────────────────────

class PseudoLabelDataset(Dataset):
    """
    Combined labeled + pseudo-labeled dataset for noisy student training.

    Sampling strategy from paper Figure 4:
    "For each training sample, if the corresponding class exists in the
     pseudo dataset, it is selected with 40% probability; otherwise, the
     original training sample is used."

    Implementation: interleave via weighted sampling from both pools.
    50% from labeled, 50% from pseudo-labeled (configurable).
    """

    def __init__(
        self,
        labeled_dataset:  BirdCLEFDataset,
        pseudo_df:        pd.DataFrame,
        soundscape_root:  str,
        pseudo_col:       str           = "soft_labels",   # Column with (C,) arrays
        labeled_frac:     float         = 0.5,
        sr:               int           = SAMPLE_RATE,
        chunk_duration:   float         = CHUNK_DURATION,
        label_smoothing:  float         = 0.05,
    ):
        self.labeled_ds     = labeled_dataset
        self.pseudo_df      = pseudo_df.reset_index(drop=True)
        self.soundscape_root = Path(soundscape_root)
        self.pseudo_col     = pseudo_col
        self.chunk_samples  = int(chunk_duration * sr)
        self.sr             = sr
        self.label_smoothing = label_smoothing

        n_pseudo  = len(pseudo_df)
        n_labeled = int(n_pseudo * labeled_frac / max(1 - labeled_frac, 1e-6))

        self.n_labeled = n_labeled
        self._total    = n_labeled + n_pseudo

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Dict:
        if idx < self.n_labeled:
            labeled_idx = idx % len(self.labeled_ds)
            item = self.labeled_ds[labeled_idx]
            item["start_sec"] = torch.tensor(0.0)
            # BUG FIX: DataLoader collation requires all items in a batch to have
            # the same keys. Labeled items need this key so batches with pseudo
            # items don't fail. Weight=1.0 means "fully trust this sample".
            item["uncertainty_weight"] = torch.tensor(1.0, dtype=torch.float32)
            return item

        pseudo_idx = idx - self.n_labeled
        row        = self.pseudo_df.iloc[pseudo_idx]

        audio_path  = self.soundscape_root / row["filename"]
        start_sec   = float(row.get("start_sec", 0.0))
        start_samp  = int(start_sec * self.sr)

        audio = np.zeros(self.chunk_samples, dtype=np.float32)
        try:
            if SOUNDFILE_AVAILABLE and audio_path.exists():
                data, sr = sf.read(
                    str(audio_path),
                    start     = start_samp,
                    frames    = self.chunk_samples,
                    dtype     = "float32",
                    always_2d = False,
                )
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if len(data) < self.chunk_samples:
                    data = np.pad(data, (0, self.chunk_samples - len(data)))
                audio = data[:self.chunk_samples].astype(np.float32)
        except Exception:
            pass

        raw_label = row.get(self.pseudo_col, None)
        if raw_label is not None:
            label = np.array(raw_label, dtype=np.float32)
        else:
            label = np.zeros(self.labeled_ds.n_classes, dtype=np.float32)

        # Novel #9: uncertainty weight (from MC Dropout filtering)
        unc_weight = float(row.get("uncertainty_weight", 1.0))

        return {
            "audio":              torch.tensor(audio, dtype=torch.float32),
            "label":              torch.tensor(label, dtype=torch.float32),
            "idx":                self._total - len(self.pseudo_df) + pseudo_idx,
            "filename":           str(row.get("filename", "")),
            "start_sec":          torch.tensor(start_sec),
            "uncertainty_weight": torch.tensor(unc_weight, dtype=torch.float32),
        }
