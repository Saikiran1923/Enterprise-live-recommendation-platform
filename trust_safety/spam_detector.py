"""Spam detection for videos and user accounts."""

import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class SpamDetector:
    """Detects spam content and fraudulent engagement patterns."""

    SPAM_PATTERNS = [
        r"(click here|subscribe now|free money|100% guaranteed)",
        r"(buy followers|cheap views|instant subscribers)",
        r"(\b\d{10}\b.*whatsapp|telegram.*@\w+)",
    ]

    def __init__(self, spam_threshold: float = 0.8):
        self.spam_threshold = spam_threshold
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.SPAM_PATTERNS]

    def is_spam_content(self, title: str, description: str = "") -> Tuple[bool, float]:
        """Detect if video content is spam based on title/description signals."""
        text = f"{title} {description}"
        pattern_matches = sum(1 for pat in self._compiled if pat.search(text))
        score = min(1.0, pattern_matches / max(len(self._compiled), 1) * 2)
        return score >= self.spam_threshold, score

    def is_fraud_engagement(self, video_signals: Dict[str, Any]) -> Tuple[bool, str]:
        """Detect artificially inflated engagement metrics."""
        views = video_signals.get("views", 0)
        likes = video_signals.get("likes", 0)
        watch_sec = video_signals.get("watch_sec", 0)
        avg_watch = watch_sec / max(views, 1)

        # Suspiciously high like rate with very low watch time
        like_rate = likes / max(views, 1)
        if like_rate > 0.8 and avg_watch < 5:
            return True, "like_rate_anomaly"

        # Views with zero watch time
        if views > 100 and avg_watch < 1:
            return True, "zero_watch_anomaly"

        return False, ""

    def get_spam_score(self, title: str, description: str,
                       video_signals: Dict[str, Any]) -> float:
        """Composite spam score combining content and engagement signals."""
        _, content_score = self.is_spam_content(title, description)
        is_fraud, _ = self.is_fraud_engagement(video_signals)
        fraud_score = 0.9 if is_fraud else 0.0
        return max(content_score, fraud_score)
