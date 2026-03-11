"""Ranking inference pipeline: score candidates and return ranked list."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from ranking.ranking_model import RankingModel

logger = logging.getLogger(__name__)


class RankingInference:
    """
    Orchestrates the ranking step: fetches video features,
    builds the feature matrix, scores with the model, and returns
    the top-k ranked candidates.
    """

    def __init__(self, ranking_model: RankingModel,
                 feature_store=None, top_k: int = 50):
        self._model = ranking_model
        self._feature_store = feature_store
        self.top_k = top_k

    async def rank(self, candidates: List[Dict[str, Any]],
                   user_features: Dict[str, Any],
                   session_features: Dict[str, Any],
                   top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Score and rank candidate videos for a user.
        Returns list of ranked candidates with scores.
        """
        k = top_k or self.top_k
        if not candidates:
            return []

        enriched = await self._enrich_candidates(candidates)
        feature_matrix, video_ids = self._model.build_feature_matrix(
            enriched, user_features, session_features
        )

        scores = self._model.predict(feature_matrix)

        ranked = sorted(
            zip(video_ids, scores, enriched),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {**cand, "ranking_score": float(score), "rank": i + 1}
            for i, (vid, score, cand) in enumerate(ranked[:k])
        ]

    async def _enrich_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch video features for all candidates."""
        if not self._feature_store:
            return candidates

        video_ids = [c["video_id"] for c in candidates]
        feature_names = [
            "video_view_count_log", "video_like_rate", "video_avg_watch_sec",
            "video_freshness_score", "video_trending_score",
            "video_duration_log", "video_category",
        ]
        enriched = []
        for i, (cand, vid) in enumerate(zip(candidates, video_ids)):
            try:
                vf = await self._feature_store.get_video_features(vid, feature_names)
                enriched.append({**cand, "video_features": vf})
            except Exception:
                enriched.append({**cand, "video_features": {}})

        return enriched
