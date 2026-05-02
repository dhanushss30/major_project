"""
build_taxonomy_map.py — Build species-to-taxon mapping from metadata
=====================================================================
Reads train.csv and assigns each species to its taxonomic group:
  0=Aves, 1=Amphibia, 2=Insecta, 3=Mammalia

Output: JSON file mapping species_code → taxon_index.

Usage:
  python scripts/build_taxonomy_map.py \
      --data_root D:/birdclef_data/birdclef-2025 \
      --output configs/species_to_taxon.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TAXON_MAP = {
    "Aves":     0,
    "Amphibia": 1,
    "Insecta":  2,
    "Mammalia": 3,
}

KNOWN_INSECTS = [
    "y00678", "y00590", "y00498", "y00475", "y00474",
    "y00293", "y00285", "y00266", "y00234", "y00213",
    "y00193", "y00179", "y00175", "y00140", "y00099",
    "y00076", "y00017",
]

KNOWN_AMPHIBIA = [
    "64862", "65448", "65547", "55514", "52884",
    "41663", "22333", "22976", "blarai1",
]

KNOWN_MAMMALIA = [
    "y01234", "y01235", "y01236", "y01237",
    "y01238", "y01239", "y01240", "y01241", "y01242",
]


def build_taxonomy_map(data_root: str) -> dict:
    df = pd.read_csv(Path(data_root) / "train.csv")

    species_to_taxon = {}

    if "class_name" in df.columns or "taxonomic_class" in df.columns:
        tax_col = "class_name" if "class_name" in df.columns else "taxonomic_class"
        for _, row in df.drop_duplicates("primary_label").iterrows():
            species = row["primary_label"]
            taxon_str = str(row.get(tax_col, "Aves"))
            taxon_idx = TAXON_MAP.get(taxon_str, 0)
            species_to_taxon[species] = taxon_idx
    else:
        all_species = df["primary_label"].unique()
        for sp in all_species:
            if sp in KNOWN_INSECTS:
                species_to_taxon[sp] = 2
            elif sp in KNOWN_AMPHIBIA:
                species_to_taxon[sp] = 1
            elif sp in KNOWN_MAMMALIA:
                species_to_taxon[sp] = 3
            else:
                species_to_taxon[sp] = 0

    counts = {}
    for v in species_to_taxon.values():
        label = [k for k, val in TAXON_MAP.items() if val == v][0]
        counts[label] = counts.get(label, 0) + 1

    print(f"Taxonomy mapping built: {counts}")
    return species_to_taxon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, default="configs/species_to_taxon.json")
    args = parser.parse_args()

    mapping = build_taxonomy_map(args.data_root)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
