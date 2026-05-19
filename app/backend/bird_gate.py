"""bird_gate.py - pretrained audio classifier used as a 'is this a bird?' gate.

Uses MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer
fine-tuned on AudioSet, 527 classes). For each 5-second chunk, computes:
  - bird_score:           sum of softmax probability over AudioSet bird classes
  - top_non_bird_label:   highest-scoring non-bird AudioSet class (Speech, Music, ...)
  - top_non_bird_score:   its probability

routes/predict.py uses these to decide whether to surface the 206-class species
prediction or label the chunk as a non-bird sound. The point is to fix the
open-set problem: our 206-class softmax classifier has no concept of 'not a
bird', so on human voice / music / traffic it still picks some species with
moderate confidence. The AST gate is purpose-trained on a 2M-clip corpus that
includes both birds and these distractor categories, so it can distinguish.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio

import config


# AudioSet ontology labels classified as "bird". Curated from the official
# AudioSet ontology bird subtree: https://github.com/audioset/ontology
# Strings must match the model's id2label entries EXACTLY.
# AudioSet labels that are definitively NON-bird — used to give the predict
# route a high-confidence rejection signal. If AST's top non-bird class is in
# this set with reasonable confidence, we mark the chunk as "non-bird (Label)"
# instead of fishing through the 206-class species head.
HARD_NON_BIRD_LABELS = frozenset({
    # Speech / voice
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue", "Babbling", "Whispering",
    "Shout", "Yell", "Children shouting", "Screaming",
    "Laughter", "Crying, sobbing", "Sigh", "Cough", "Sneeze", "Snoring",
    # Music
    "Music", "Musical instrument", "Singing", "Background music",
    "Drum", "Drum kit", "Guitar", "Piano", "Violin, fiddle",
    "Rock music", "Pop music", "Hip hop music", "Classical music",
    # Vehicles / engines
    "Vehicle", "Engine", "Engine starting", "Idling",
    "Motor vehicle (road)", "Car", "Truck", "Bus", "Motorcycle", "Train",
    "Aircraft", "Helicopter", "Boat, Water vehicle",
    # Tones / synthetic
    "Sine wave", "Chirp tone", "Static", "White noise", "Pink noise",
    "Noise", "Hum", "Tone", "Hiss",
    # Silence / room
    "Silence", "Inside, small room", "Inside, public space",
    # Beeps / alarms
    "Beep, bleep", "Buzzer", "Alarm", "Siren", "Telephone bell ringing",
    "Ringtone", "Bell", "Church bell",
    # Tools / domestic
    "Hammer", "Saw", "Power tool", "Vacuum cleaner", "Blender",
    "Microwave oven", "Washing machine",
    # Domestic / common mammals (none of these are in any wildlife species set)
    "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)",
    "Canidae, dogs, wolves", "Domestic animals, pets",
    "Cat", "Caterwaul", "Meow", "Hiss", "Purr",
    "Cattle, bovinae", "Moo",
    "Horse", "Neigh, whinny", "Clip-clop",
    "Pig", "Oink",
    "Goat", "Bleat", "Sheep",
    "Livestock, farm animals, working animals",
    "Rodents, rats, mice", "Mouse",
})


BIRD_AUDIOSET_LABELS = frozenset({
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Chirp, tweet",
    "Squawk",
    "Pigeon, dove",
    "Coo",
    "Crow",
    "Caw",
    "Owl",
    "Hoot",
    "Bird flight, flapping wings",
    "Chicken, rooster",
    "Cluck",
    "Crowing, cock-a-doodle-doo",
    "Turkey",
    "Gobble",
    "Duck",
    "Quack",
    "Goose",
    "Honk",
    "Fowl",
    "Wild animals",  # often co-fires with bird in field recordings
})

AST_MODEL_ID  = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR        = 16_000           # AST input sample rate
CHUNK_SEC     = 5.0
CHUNK_SAMPLES = int(AST_SR * CHUNK_SEC)


class BirdGate:
    """Singleton-style AST wrapper. Loads model once at startup."""

    def __init__(self):
        self._model = None
        self._fe    = None
        self._labels: Dict[int, str] = {}
        self._bird_mask: np.ndarray | None = None   # shape (527,)
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        print(f"[bird_gate] loading AST ({AST_MODEL_ID})...")
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
        self._fe    = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID)
        self._model = ASTForAudioClassification.from_pretrained(AST_MODEL_ID)
        self._model.to(config.DEVICE).eval()

        self._labels = dict(self._model.config.id2label)
        n_classes = len(self._labels)
        mask = np.zeros(n_classes, dtype=bool)
        found_labels: set[str] = set()
        for i, label in self._labels.items():
            if label in BIRD_AUDIOSET_LABELS:
                mask[int(i)] = True
                found_labels.add(label)
        self._bird_mask = mask
        # Hard-non-bird mask (Dog, Speech, Music, etc.) — used to pick a
        # SPECIFIC display label when the gate fires on bird_score alone.
        hard_mask = np.zeros(n_classes, dtype=bool)
        for i, label in self._labels.items():
            if label in HARD_NON_BIRD_LABELS:
                hard_mask[int(i)] = True
        self._hard_mask = hard_mask
        missing = BIRD_AUDIOSET_LABELS - found_labels
        if missing:
            print(f"[bird_gate] WARN: {len(missing)} expected bird labels not found in model: {missing}")
        print(f"[bird_gate] ready - {n_classes} AudioSet classes, "
              f"{int(mask.sum())} flagged as bird, "
              f"{int(hard_mask.sum())} flagged hard-non-bird")
        self._initialized = True

    @property
    def n_bird_classes(self) -> int:
        return int(self._bird_mask.sum()) if self._bird_mask is not None else 0

    # ─── Audio loading ──────────────────────────────────────────────────
    @staticmethod
    def _load_audio_16k(audio_path: str | Path) -> torch.Tensor:
        """Read a file as mono float32 @ 16 kHz (AST's required input rate)."""
        data, sr_in = sf.read(str(audio_path), always_2d=False, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data)
        if sr_in != AST_SR:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr_in, AST_SR
            ).squeeze(0)
        return wav

    @staticmethod
    def _chunk_to_5s(wav16k: torch.Tensor, n_chunks: int) -> np.ndarray:
        """Split a 16 kHz waveform into exactly `n_chunks` non-overlapping 5-sec
        windows, zero-padding the final chunk if the audio is shorter than
        n_chunks * 5 sec. Returns shape (n_chunks, CHUNK_SAMPLES) float32.
        """
        total_needed = n_chunks * CHUNK_SAMPLES
        if wav16k.numel() < total_needed:
            pad = total_needed - wav16k.numel()
            wav16k = torch.nn.functional.pad(wav16k, (0, pad))
        else:
            wav16k = wav16k[:total_needed]
        return wav16k.view(n_chunks, CHUNK_SAMPLES).cpu().numpy()

    # ─── Scoring ────────────────────────────────────────────────────────
    @torch.no_grad()
    def score_file(
        self,
        audio_path: str | Path,
        n_chunks:   int,
    ) -> List[Dict]:
        """Score `n_chunks` consecutive 5-second windows of the audio file.

        Returns a list of n_chunks dicts:
            {
                bird_score:          float in [0, 1],
                top_non_bird_label:  str,
                top_non_bird_score:  float in [0, 1],
                top_label:           str,
                top_score:           float in [0, 1],
            }
        """
        if not self._initialized:
            self.initialize()

        wav16   = self._load_audio_16k(audio_path)
        chunks  = self._chunk_to_5s(wav16, n_chunks)
        # chunks: list of np arrays for the feature extractor
        chunk_list = [chunks[i] for i in range(chunks.shape[0])]

        inputs = self._fe(chunk_list, sampling_rate=AST_SR, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
        logits = self._model(**inputs).logits         # (n_chunks, 527)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        bird_mask    = self._bird_mask
        nonbird_mask = ~bird_mask
        hard_mask    = self._hard_mask
        for p in probs:
            bird_score = float(p[bird_mask].sum())
            # Top non-bird class
            nb_probs = p * nonbird_mask
            top_nb_idx   = int(np.argmax(nb_probs))
            top_nb_score = float(nb_probs[top_nb_idx])
            # Best HARD non-bird class (Dog, Speech, Music, ...) — may be
            # different from the top non-bird (which can be the parent
            # "Animal" while a more specific child like "Dog" sits below).
            hard_probs = p * hard_mask
            best_hard_idx   = int(np.argmax(hard_probs))
            best_hard_score = float(hard_probs[best_hard_idx])
            # Top overall
            top_idx   = int(np.argmax(p))
            top_score = float(p[top_idx])
            results.append({
                "bird_score":          bird_score,
                "top_non_bird_label":  self._labels[top_nb_idx],
                "top_non_bird_score":  top_nb_score,
                "best_hard_label":     self._labels[best_hard_idx],
                "best_hard_score":     best_hard_score,
                "top_label":           self._labels[top_idx],
                "top_score":           top_score,
            })
        return results


bird_gate = BirdGate()
