"""High-level recommendation service coordinating all subsystems."""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RecommendationService:
    """Top-level service facade for the recommendation system."""

    def __init__(self, engine, ab_router, experiment_logger, metrics_collector):
        self._engine = engine
        self._ab_router = ab_router
        self._exp_logger = experiment_logger
        self._metrics = metrics_collector

    async def recommend(self, user_id: str, session_id: str,
                        context: Dict[str, Any], top_k: int = 20) -> Dict[str, Any]:
        assignments = self._ab_router.assign_all(user_id)
        context["experiment_assignments"] = assignments

        result = await self._engine.recommend(
            user_id=user_id, session_id=session_id,
            context=context, top_k=top_k,
        )

        for exp_id, variant in assignments.items():
            rec_ids = [r["video_id"] for r in result.get("recommendations", [])]
            await self._exp_logger.log_impression(user_id, exp_id, variant, rec_ids)

        if self._metrics:
            self._metrics.increment("recommendations_served")
            latency = result.get("metadata", {}).get("latency_ms", 0)
            self._metrics.histogram("recommendation_latency_ms", latency)

        return result
