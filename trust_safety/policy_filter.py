"""Content policy filter for enforcing platform rules."""

import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)


class PolicyFilter:
    """Filters recommendations against content policy rules."""

    def __init__(self, blocked_categories: List[str] = None,
                 min_creator_trust_score: float = 0.5):
        self._blocked_categories: Set[str] = set(blocked_categories or [])
        self.min_creator_trust_score = min_creator_trust_score

    def filter_candidates(self, candidates: List[Dict[str, Any]],
                           video_metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove policy-violating videos from candidates."""
        filtered = []
        removed_count = 0
        for cand in candidates:
            vid = cand.get("video_id", "")
            meta = video_metadata.get(vid, {})
            if self._passes_policy(cand, meta):
                filtered.append(cand)
            else:
                removed_count += 1
        if removed_count > 0:
            logger.info(f"Policy filter removed {removed_count} candidates")
        return filtered

    def _passes_policy(self, candidate: Dict[str, Any],
                       meta: Dict[str, Any]) -> bool:
        # Check blocked categories
        if meta.get("category") in self._blocked_categories:
            return False
        # Check creator trust score
        if meta.get("creator_trust_score", 1.0) < self.min_creator_trust_score:
            return False
        # Check if video is age-restricted and user is underage
        if meta.get("age_restricted") and candidate.get("user_age", 99) < 18:
            return False
        # Check if video is active (not deleted/suspended)
        if not meta.get("is_active", True):
            return False
        return True

    def add_blocked_category(self, category: str) -> None:
        self._blocked_categories.add(category)

    def remove_blocked_category(self, category: str) -> None:
        self._blocked_categories.discard(category)
