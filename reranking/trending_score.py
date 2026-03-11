"""Computes and applies trending scores based on velocity signals."""

import math
import time
from typing import Dict, Any, List


class TrendingScore:
    """
    Computes trending scores based on view velocity,
    like velocity, and share velocity over configurable time windows.
    """

    def __init__(self, short_window_hours: float = 1.0,
                 long_window_hours: float = 24.0):
        self.short_window = short_window_hours * 3600
        self.long_window = long_window_hours * 3600

    def compute(self, video_signals: Dict[str, Any]) -> float:
        """Compute trending score for a single video."""
        now = time.time()
        last_viewed = video_signals.get("last_viewed", now)
        time_since = now - last_viewed

        views = video_signals.get("views", 0)
        likes = video_signals.get("likes", 0)
        shares = video_signals.get("shares", 0)

        # Velocity: engagement per hour in the recent window
        velocity = (views + likes * 2 + shares * 5) / max(time_since / 3600, 0.1)
        trending = math.log1p(velocity)

        # Recency multiplier
        recency_mul = math.exp(-time_since / self.long_window)
        return trending * recency_mul

    def apply_batch(self, candidates: List[Dict[str, Any]],
                    video_signals_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply trending scores to a list of candidates."""
        for cand in candidates:
            vid = cand.get("video_id", "")
            signals = video_signals_map.get(vid, {})
            cand["video_trending_score"] = self.compute(signals)
        return candidates
