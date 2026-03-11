"""Applies freshness boosting to promote recent content."""

import math
import time
from typing import List, Dict, Any


class FreshnessBoost:
    """Boosts ranking scores for recently uploaded videos."""

    def __init__(self, decay_hours: float = 24.0, boost_factor: float = 0.15):
        self.decay_hours = decay_hours
        self.boost_factor = boost_factor

    def apply(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply freshness boost to candidate scores in-place."""
        for cand in candidates:
            age_hours = cand.get("video_age_hours", 0)
            freshness = math.exp(-age_hours / self.decay_hours)
            cand["ranking_score"] = (
                cand.get("ranking_score", 0) * (1 + self.boost_factor * freshness)
            )
            cand["freshness_boost"] = freshness * self.boost_factor
        return sorted(candidates, key=lambda x: x["ranking_score"], reverse=True)
