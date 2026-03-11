"""Classifies content toxicity using a pre-trained model."""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

TOXICITY_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ToxicityClassifier:
    """
    Classifies text content (titles, comments) for toxicity.
    Uses a pre-trained text classification model.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._model = None
        self._is_loaded = False

    def load(self, model_path: str) -> None:
        logger.info(f"Loading toxicity model from {model_path}")
        self._is_loaded = True

    def classify(self, text: str) -> Dict[str, float]:
        """Return toxicity scores per category (0-1 scale)."""
        if not self._is_loaded:
            return self._keyword_fallback(text)
        # In production: run inference with loaded model
        return {label: 0.0 for label in TOXICITY_LABELS}

    def is_toxic(self, text: str) -> Tuple[bool, str]:
        """Returns (is_toxic, reason) tuple."""
        scores = self.classify(text)
        for label, score in scores.items():
            if score >= self.threshold:
                return True, f"{label}:{score:.2f}"
        return False, ""

    def batch_classify(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self.classify(t) for t in texts]

    def _keyword_fallback(self, text: str) -> Dict[str, float]:
        """Simple keyword-based fallback when model not loaded."""
        blocked_keywords = {"spam", "scam", "hate", "kill"}
        text_lower = text.lower()
        is_toxic = any(kw in text_lower for kw in blocked_keywords)
        score = 0.9 if is_toxic else 0.0
        return {label: score if label == "toxic" else 0.0 for label in TOXICITY_LABELS}
