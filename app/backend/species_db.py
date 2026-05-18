"""species_db.py — In-memory metadata database for the 206 WildEar classes.

Auto-builds from train.csv + ensemble_per_class_auc.csv. Augments with
hand-curated info for top common species (and on-demand AI-generated info
via the RAG service for the rest).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

import config


# ── Hand-curated info for the top demo species ───────────────────────────
# These are the well-trained common species we know work in demos.
CURATED_INFO: Dict[str, dict] = {
    "trokin": {
        "common_name":     "Tropical Kingbird",
        "scientific_name": "Tyrannus melancholicus",
        "family":          "Tyrannidae",
        "order":           "Passeriformes",
        "habitat":         "Open and semi-open habitats, edges, gardens",
        "size_cm":         22,
        "conservation":    "Least Concern",
        "description":     "A medium-sized, vocal flycatcher with grey head, yellow underparts and dark mask. One of the most common Neotropical flycatchers. Aggressive defender of its territory.",
    },
    "grekis": {
        "common_name":     "Great Kiskadee",
        "scientific_name": "Pitangus sulphuratus",
        "family":          "Tyrannidae",
        "order":           "Passeriformes",
        "habitat":         "Open woodland, agricultural areas, urban parks",
        "size_cm":         23,
        "conservation":    "Least Concern",
        "description":     "Large flycatcher with bold black-and-white head pattern and bright yellow underparts. Famously named after its distinctive 'kis-ka-dee' call. Highly adaptable across the Neotropics.",
    },
    "compau": {
        "common_name":     "Common Pauraque",
        "scientific_name": "Nyctidromus albicollis",
        "family":          "Caprimulgidae",
        "order":           "Caprimulgiformes",
        "habitat":         "Forest edges, scrubland, often near roads at night",
        "size_cm":         28,
        "conservation":    "Least Concern",
        "description":     "Nocturnal ground-dwelling bird with cryptic plumage. Detected almost exclusively by its long whistled call at dusk and night. Often heard calling from forest edges.",
    },
    "yeofly1": {
        "common_name":     "Yellow-bellied Elaenia",
        "scientific_name": "Elaenia flavogaster",
        "family":          "Tyrannidae",
        "order":           "Passeriformes",
        "habitat":         "Shrubby second growth, gardens, woodland edges",
        "size_cm":         16,
        "conservation":    "Least Concern",
        "description":     "Small olive-and-yellow flycatcher with raised crown feathers. Common in disturbed habitats across the Neotropics. Often seen perched in pairs.",
    },
    "banana": {
        "common_name":     "Bananaquit",
        "scientific_name": "Coereba flaveola",
        "family":          "Thraupidae",
        "order":           "Passeriformes",
        "habitat":         "Forest, gardens, urban areas with flowers",
        "size_cm":         11,
        "conservation":    "Least Concern",
        "description":     "Tiny, active nectar-feeding bird with a curved black bill and bright yellow underparts. Very common across the Neotropics, often visits feeders.",
    },
    "smbani": {
        "common_name":     "Smooth-billed Ani",
        "scientific_name": "Crotophaga ani",
        "family":          "Cuculidae",
        "order":           "Cuculiformes",
        "habitat":         "Open grassy areas, agricultural land, scrubland",
        "size_cm":         34,
        "conservation":    "Least Concern",
        "description":     "Highly social black cuckoo with a heavy curved bill. Forages in flocks. Communal breeding — multiple females lay eggs in one nest.",
    },
    "bugtan": {
        "common_name":     "Blue-grey Tanager",
        "scientific_name": "Thraupis episcopus",
        "family":          "Thraupidae",
        "order":           "Passeriformes",
        "habitat":         "Forest edges, gardens, plantations, towns",
        "size_cm":         17,
        "conservation":    "Least Concern",
        "description":     "Small pale-blue tanager with darker wings. Extremely common in human-modified habitats across the Neotropics. Often in pairs or small groups visiting fruiting trees.",
    },
    "whtdov": {
        "common_name":     "White-tipped Dove",
        "scientific_name": "Leptotila verreauxi",
        "family":          "Columbidae",
        "order":           "Columbiformes",
        "habitat":         "Forest understory, scrubland, gardens",
        "size_cm":         28,
        "conservation":    "Least Concern",
        "description":     "Medium-sized pale dove with white outer tail feathers visible in flight. Often forages on the ground. Distinct mournful cooing call.",
    },
    "saffin": {
        "common_name":     "Saffron Finch",
        "scientific_name": "Sicalis flaveola",
        "family":          "Thraupidae",
        "order":           "Passeriformes",
        "habitat":         "Open country, parks, agricultural fields",
        "size_cm":         14,
        "conservation":    "Least Concern",
        "description":     "Bright yellow tanager (despite the name 'finch') with orange wash on the head. Very common in open landscapes. Often nests in cavities, including building walls.",
    },
    "roahaw": {
        "common_name":     "Roadside Hawk",
        "scientific_name": "Rupornis magnirostris",
        "family":          "Accipitridae",
        "order":           "Accipitriformes",
        "habitat":         "Forest edges, agricultural land, roadsides",
        "size_cm":         36,
        "conservation":    "Least Concern",
        "description":     "Small adaptable hawk commonly seen perched on roadside trees. Mostly hunts insects, reptiles, and small rodents from a perch.",
    },
    "socfly1": {
        "common_name":     "Social Flycatcher",
        "scientific_name": "Myiozetetes similis",
        "family":          "Tyrannidae",
        "order":           "Passeriformes",
        "habitat":         "Forest edges, gardens, near water",
        "size_cm":         17,
        "conservation":    "Least Concern",
        "description":     "Yellow-bellied flycatcher resembling a small kiskadee but with shorter, thinner bill. Highly social, often calls in groups.",
    },
}


@dataclass
class SpeciesInfo:
    code:               str
    common_name:        Optional[str]      = None
    scientific_name:    Optional[str]      = None
    family:             Optional[str]      = None
    order:              Optional[str]      = None
    habitat:            Optional[str]      = None
    size_cm:            Optional[float]    = None
    conservation:       Optional[str]      = None
    description:        Optional[str]      = None
    n_train_samples:    int                = 0
    val_auc:            Optional[float]    = None
    is_well_trained:    bool               = False
    is_bird:            bool               = True       # False for iNat numeric IDs (insects/amphibians)
    curated:            bool               = False


class SpeciesDB:
    def __init__(self):
        self._species: Dict[str, SpeciesInfo] = {}
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        train_csv = Path(config.TRAIN_CSV)
        auc_csv   = Path(config.PER_CLASS_AUC_CSV)

        # Strategy A: Use train.csv if available (preferred — exact sample counts)
        counts: Dict[str, int] = {}
        labels: List[str] = []
        train_csv_available = train_csv.exists()

        if train_csv_available:
            df = pd.read_csv(train_csv)
            counts_series = df["primary_label"].value_counts()
            counts = {str(k): int(v) for k, v in counts_series.items()}
            labels = sorted(counts.keys())
            print(f"[species_db] loading {len(labels)} species from train.csv…")
        elif auc_csv.exists():
            # Strategy B: Derive labels from per_class_auc.csv (no exact train counts)
            df = pd.read_csv(auc_csv)
            labels = sorted(df["label"].astype(str).unique())
            # Estimate n_train as ~5x n_val_pos (5-fold split assumption)
            for _, row in df.iterrows():
                code = str(row["label"])
                n_pos = int(row.get("n_pos", 0)) if pd.notna(row.get("n_pos")) else 0
                counts[code] = n_pos * 5   # approximation: 4-folds-train + 1-fold-val
            print(f"[species_db] train.csv not found — loading {len(labels)} species "
                  f"from {auc_csv.name} (n_train estimated as 5x n_pos)")
        else:
            print(f"[species_db] WARN: neither train.csv nor per_class_auc.csv found")
            return

        # Load per-class AUC
        auc_map: Dict[str, float] = {}
        if auc_csv.exists():
            auc_df = pd.read_csv(auc_csv)
            for _, row in auc_df.iterrows():
                if pd.notna(row.get("auc")):
                    auc_map[str(row["label"])] = float(row["auc"])

        # Build entries
        for code in labels:
            info = SpeciesInfo(code=code)
            info.n_train_samples = counts.get(code, 0)
            info.val_auc = auc_map.get(code)
            info.is_well_trained = info.n_train_samples >= 50

            # iNat numeric IDs are non-birds (filtered rare classes)
            info.is_bird = not code.isdigit()

            # Curated metadata for top species
            if code in CURATED_INFO:
                cur = CURATED_INFO[code]
                info.common_name     = cur["common_name"]
                info.scientific_name = cur["scientific_name"]
                info.family          = cur["family"]
                info.order           = cur["order"]
                info.habitat         = cur["habitat"]
                info.size_cm         = cur["size_cm"]
                info.conservation    = cur["conservation"]
                info.description     = cur["description"]
                info.curated         = True

            self._species[code] = info

        self._initialized = True
        n_curated = sum(1 for s in self._species.values() if s.curated)
        n_well    = sum(1 for s in self._species.values() if s.is_well_trained)
        n_birds   = sum(1 for s in self._species.values() if s.is_bird)
        print(f"[species_db] ready — {len(self._species)} species "
              f"({n_birds} birds, {n_curated} curated, {n_well} well-trained)")

    def get(self, code: str) -> Optional[SpeciesInfo]:
        if not self._initialized:
            self.initialize()
        return self._species.get(code)

    def all_codes(self) -> List[str]:
        if not self._initialized:
            self.initialize()
        return list(self._species.keys())

    def to_dict(self, code: str) -> Optional[dict]:
        info = self.get(code)
        return asdict(info) if info else None

    def list_all(self) -> List[dict]:
        if not self._initialized:
            self.initialize()
        return [asdict(s) for s in self._species.values()]

    def list_well_trained_birds(self) -> List[dict]:
        if not self._initialized:
            self.initialize()
        return [
            asdict(s) for s in self._species.values()
            if s.is_bird and s.is_well_trained
        ]


species_db = SpeciesDB()
