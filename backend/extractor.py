"""
NER Extractor - loads trained spaCy model from models/ner/model-best
Entity types: CAMPUS_LOCATION, FACILITY_TYPE, TIME_REFERENCE, EVENT_TYPE
"""

import spacy
import os
from pathlib import Path


class NERExtractor:
    def __init__(self):
        model_path = Path(__file__).resolve().parents[3] / "models" / "ner" / "model-best"

        if not model_path.exists():
            print(f"[NER] WARNING: Model not found at {model_path}. Run train_ner.py first.")
            self.nlp = None
        else:
            self.nlp = spacy.load(str(model_path))
            print(f"[NER] Model loaded from {model_path}")

    def extract(self, text: str) -> dict:
        """
        Extract entities from text.
        Returns dict with entity type keys and lists of matched strings.
        """
        result = {
            "CAMPUS_LOCATION": [],
            "FACILITY_TYPE": [],
            "TIME_REFERENCE": [],
            "EVENT_TYPE": [],
        }

        if self.nlp is None:
            return result

        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in result:
                result[ent.label_].append(ent.text)

        return result

    def extract_flat(self, text: str) -> list:
        """
        Returns flat list of (text, label) tuples — useful for debugging.
        """
        if self.nlp is None:
            return []
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
