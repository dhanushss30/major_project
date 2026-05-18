"""rag_service.py — RAG over species + methodology info using Groq.

Acts as both:
  1. AI-generated species info (when curated info is missing)
  2. General chatbot for the app (asks about model, methodology, species)

Uses Groq's free API. Get your key at https://console.groq.com/keys
"""

from __future__ import annotations

import os
from typing import List, Optional

import config

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


PROJECT_CONTEXT = """\
You are an expert assistant for WildEar — a Neotropical bird species classification system.

ABOUT THE PROJECT:
- 206-class multi-label audio classifier built on ECA-NFNet-L0 backbone
- Achieves macro-averaged ROC AUC of 0.8246 on the validation set
- 4-checkpoint ensemble: v3 clean (val/auc 0.7756), v3 FiLM (0.7841), ESC-50 BG noise (0.6814), new fold 1 (0.6976)
- Inference: 5-second mel-spectrogram chunks at 32 kHz, 128 mel bins
- Test-Time Augmentation: 50% overlapping windows
- Noise robustness: spectral noise gating + 1-8 kHz bandpass + pre-emphasis filter
- Open-set rejection: max-prob threshold for "not a bird" detection
- Rare-class filter: 78 of 206 classes (<50 samples) excluded from training

NOVEL RESEARCH FINDINGS:
1. Rare-class pseudo-pollution: when teacher-student pseudo-labeling is applied to long-tail
   datasets, rare classes (iNaturalist numeric taxa) dominate pseudo-label generation due
   to memorization, corrupting the student. Mitigation: post-hoc rare-class masking.
2. BG-augmentation labeling contradiction: mixing same-region soundscape audio as background
   noise during training causes collapse because the BG contains unlabeled target species,
   creating contradictory labels. Mitigation: use out-of-distribution noise (ESC-50).
3. Open-set rejection via max-prob thresholding for real-world deployment.

KNOWN LIMITATIONS:
- Trained on focal (single-bird, clean) recordings — performance degrades on multi-species soundscape audio
- Rare classes (<50 training samples) are unreliable
- Conservative probability calibration (max ~0.20 even for correct predictions; BCE + label-smoothing artifact)
- Per-recording top-1 accuracy is lower than average ranking AUC

RESPONSE STYLE:
- Be concise, friendly, knowledgeable
- Cite specific numbers when relevant
- Acknowledge limitations honestly
- For species questions, provide: scientific name, family, distinctive vocalizations, habitat
- For methodology questions, explain the technical detail clearly
"""


class RAGService:
    def __init__(self):
        self._client = None
        self._enabled = False

    def initialize(self):
        if not HAS_GROQ:
            print("[rag_service] groq package not installed; chatbot disabled")
            return
        if not config.GROQ_API_KEY:
            print("[rag_service] GROQ_API_KEY not set; chatbot disabled "
                  "(get one at https://console.groq.com/keys)")
            return
        self._client = Groq(api_key=config.GROQ_API_KEY)
        self._enabled = True
        print(f"[rag_service] ready — using {config.GROQ_MODEL}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def chat(
        self,
        user_message: str,
        context_species: Optional[dict] = None,
        history: Optional[List[dict]] = None,
    ) -> str:
        """Single-turn chat. Optional species context for in-app species queries."""
        if not self._enabled:
            return ("⚠️ AI chat is currently disabled. Set GROQ_API_KEY in the "
                    "backend .env file (free key at https://console.groq.com/keys) "
                    "and restart.")

        system = PROJECT_CONTEXT
        if context_species:
            system += f"\n\nCURRENT SPECIES CONTEXT:\n{context_species}\n"
            system += ("\nAnswer the user's question with this species in mind. "
                       "If the species is in our 206-class list, prefer to reference "
                       "the technical metadata (n_train_samples, val_auc, etc.).")

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model       = config.GROQ_MODEL,
                messages    = messages,
                temperature = 0.4,
                max_tokens  = 800,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ AI chat error: {e}"

    def generate_species_description(self, species_info: dict) -> str:
        """Generate a rich description for a species lacking curated info."""
        if not self._enabled:
            return ""
        prompt = (
            f"Provide a concise (3-4 sentences) description of the species with "
            f"label code '{species_info.get('code')}'. If this is a numeric iNaturalist "
            f"taxon ID, mention it may be a non-bird species (insect, amphibian). "
            f"Include scientific name, common name, family, and a distinctive "
            f"vocalization or habitat detail if known."
        )
        return self.chat(prompt)


rag_service = RAGService()
