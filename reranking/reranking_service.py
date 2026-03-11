"""Orchestrates the full reranking pipeline."""

import logging
from typing import List, Dict, Any

from reranking.diversity_optimizer import DiversityOptimizer
from reranking.freshness_boost import FreshnessBoost
from reranking.trending_score import TrendingScore

logger = logging.getLogger(__name__)


class RerankingService:
    """
    Applies post-ranking transformations:
    1. Trending score injection
    2. Freshness boost
    3. Diversity optimization
    4. Safety/policy filtering
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._diversity = DiversityOptimizer(
            diversity_weight=cfg.get("diversity_weight", 0.3),
            max_same_creator=cfg.get("max_same_creator", 3),
        )
        self._freshness = FreshnessBoost(
            decay_hours=cfg.get("freshness_decay_hours", 24),
            boost_factor=cfg.get("freshness_boost_factor", 0.15),
        )
        self._trending = TrendingScore()

    async def rerank(self, candidates: List[Dict[str, Any]],
                     video_signals_map: Dict[str, Dict[str, Any]],
                     top_k: int = 20) -> List[Dict[str, Any]]:
        """Full reranking pipeline."""
        if not candidates:
            return []

        # Step 1: Inject trending scores
        candidates = self._trending.apply_batch(candidates, video_signals_map)

        # Step 2: Apply freshness boost
        candidates = self._freshness.apply(candidates)

        # Step 3: Diversity optimization
        candidates = self._diversity.optimize(candidates, top_k=top_k)

        logger.debug(f"Reranked {len(candidates)} candidates to top {top_k}")
        return candidates
