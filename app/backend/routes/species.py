"""routes/species.py — Species catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from species_db import species_db
from rag_service import rag_service


router = APIRouter(prefix="/api/species", tags=["species"])


@router.get("/")
def list_species(
    only_birds:        bool = Query(False, description="Filter out non-bird iNat IDs"),
    only_well_trained: bool = Query(False, description="Filter to species with >=50 training samples"),
    min_train:         int  = Query(0,     description="Minimum training-sample count"),
    family:            str | None = Query(None),
    search:            str | None = Query(None, description="Search by code/common name"),
):
    species_list = species_db.list_all()
    if only_birds:
        species_list = [s for s in species_list if s.get("is_bird")]
    if only_well_trained:
        species_list = [s for s in species_list if s.get("is_well_trained")]
    if min_train > 0:
        species_list = [s for s in species_list if s.get("n_train_samples", 0) >= min_train]
    if family:
        species_list = [s for s in species_list if s.get("family") == family]
    if search:
        q = search.lower()
        species_list = [
            s for s in species_list
            if q in (s.get("code") or "").lower()
            or q in (s.get("common_name") or "").lower()
            or q in (s.get("scientific_name") or "").lower()
        ]
    # Sort by training samples (descending)
    species_list.sort(key=lambda s: -s.get("n_train_samples", 0))
    return {
        "n_total": len(species_list),
        "species": species_list,
    }


@router.get("/{code}")
def get_species(code: str, ai_description: bool = Query(False)):
    info = species_db.to_dict(code)
    if not info:
        raise HTTPException(status_code=404, detail=f"Species code '{code}' not found.")

    # Optionally request an AI-generated description for non-curated species
    if ai_description and not info.get("curated") and rag_service.enabled:
        ai_desc = rag_service.generate_species_description(info)
        info = dict(info)
        info["ai_description"] = ai_desc

    return info


@router.get("/{code}/info")
def get_species_with_ai(code: str):
    """Always include AI description if Groq is available."""
    info = species_db.to_dict(code)
    if not info:
        raise HTTPException(status_code=404, detail=f"Species code '{code}' not found.")
    if rag_service.enabled:
        info["ai_description"] = rag_service.generate_species_description(info)
    return info
