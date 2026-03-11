"""Item-based and user-based collaborative filtering for candidate generation."""

import numpy as np
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollaborativeFiltering:
    """
    Collaborative filtering for generating video candidates.
    Implements item-item CF using co-watch matrices and
    user-user CF using interaction similarity.
    """

    def __init__(self, num_candidates: int = 200):
        self.num_candidates = num_candidates
        self._co_watch: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._user_history: Dict[str, List[str]] = defaultdict(list)
        self._item_popularity: Dict[str, float] = defaultdict(float)
        self._is_fitted = False

    def fit(self, interactions: List[Dict]) -> None:
        """Build co-watch matrix and user history from interactions."""
        user_videos: Dict[str, List[str]] = defaultdict(list)
        for event in interactions:
            uid = event.get("user_id")
            vid = event.get("video_id")
            if uid and vid:
                user_videos[uid].append(vid)
                self._item_popularity[vid] += 1

        for uid, videos in user_videos.items():
            self._user_history[uid] = videos[-50:]  # keep last 50
            unique = list(dict.fromkeys(videos))
            for i, v1 in enumerate(unique):
                for v2 in unique[i+1:i+6]:  # window of 5
                    self._co_watch[v1][v2] = self._co_watch[v1].get(v2, 0) + 1
                    self._co_watch[v2][v1] = self._co_watch[v2].get(v1, 0) + 1

        self._is_fitted = True
        logger.info(f"CF fitted: {len(self._user_history)} users, "
                    f"{len(self._co_watch)} items in co-watch matrix")

    def recommend(self, user_id: str,
                  exclude_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """Generate candidates using item-item CF from user's watch history."""
        if not self._is_fitted:
            return []

        history = self._user_history.get(user_id, [])
        if not history:
            return self._popularity_fallback(exclude_ids)

        scores: Dict[str, float] = defaultdict(float)
        exclude = exclude_ids or set(history)

        for i, video_id in enumerate(history[-10:]):  # use recent 10
            weight = 1.0 / (i + 1)  # recency weight
            for neighbor, co_count in self._co_watch.get(video_id, {}).items():
                if neighbor not in exclude:
                    scores[neighbor] += co_count * weight

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:self.num_candidates]

    def _popularity_fallback(self, exclude_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """Fallback to popularity-based candidates for cold-start users."""
        exclude = exclude_ids or set()
        sorted_items = sorted(self._item_popularity.items(),
                              key=lambda x: x[1], reverse=True)
        return [(vid, score) for vid, score in sorted_items
                if vid not in exclude][:self.num_candidates]
