"""Main engine for generating real-time recommendations within live sessions."""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LiveRecommendationEngine:
    """
    Orchestrates the full recommendation pipeline for live sessions.
    Combines long-term user preferences with real-time session signals.
    """

    def __init__(self, candidate_service, ranking_inference,
                 reranking_service, discovery_service,
                 session_tracker, session_interest_model,
                 embedding_service, feature_store,
                 config: Dict[str, Any] = None):
        self._candidates = candidate_service
        self._ranking = ranking_inference
        self._reranking = reranking_service
        self._discovery = discovery_service
        self._session_tracker = session_tracker
        self._interest_model = session_interest_model
        self._embeddings = embedding_service
        self._feature_store = feature_store
        self._config = config or {}
        self._request_count = 0
        self._total_latency = 0.0

    async def recommend(self, user_id: str, session_id: str,
                        context: Dict[str, Any],
                        top_k: int = 20) -> Dict[str, Any]:
        """
        Generate top-k recommendations for a user in their current session.
        """
        start = time.time()
        self._request_count += 1

        try:
            # 1. Get session state
            session = self._session_tracker.get_or_create(session_id, user_id)
            session_features = session.to_features()

            # 2. Get user embedding (merged with session interest)
            user_emb = await self._embeddings.get_user_embedding(user_id)
            session_emb = self._interest_model.get(session_id)
            if session_emb is not None:
                import numpy as np
                merged_emb = 0.6 * user_emb + 0.4 * session_emb
                merged_emb = merged_emb / (np.linalg.norm(merged_emb) + 1e-8)
            else:
                merged_emb = user_emb

            # 3. Get user features
            user_features = await self._feature_store.get_user_features(
                user_id, ["user_age_days_log", "user_total_views_log",
                          "user_avg_watch_sec", "user_like_rate",
                          "user_skip_rate", "user_engagement_score"]
            )

            # 4. Candidate generation
            exclude = set(session.recent_videos)
            candidates = await self._candidates.get_candidates(
                user_id, merged_emb, exclude_ids=exclude, context=context
            )

            # 5. Ranking
            ranked = await self._ranking.rank(
                candidates, user_features, session_features, top_k=top_k * 3
            )

            # 6. Exploration injection
            exploration_pool = ranked[top_k:]
            explore_items = await self._discovery.get_discovery_candidates(
                user_id, exploration_pool, context
            )
            final = self._discovery.inject_exploration(ranked[:top_k], explore_items, top_k)

            # 7. Reranking
            video_signals = {}  # Would fetch from stream processor in production
            final = await self._reranking.rerank(final, video_signals, top_k=top_k)

            latency_ms = (time.time() - start) * 1000
            self._total_latency += latency_ms

            return {
                "user_id": user_id,
                "session_id": session_id,
                "recommendations": final,
                "metadata": {
                    "latency_ms": latency_ms,
                    "candidate_count": len(candidates),
                    "model_version": "live_v1",
                    "timestamp": time.time(),
                }
            }

        except Exception as e:
            logger.error(f"Recommendation error for user {user_id}: {e}", exc_info=True)
            return {"user_id": user_id, "recommendations": [], "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self._request_count,
            "avg_latency_ms": self._total_latency / max(self._request_count, 1),
        }
