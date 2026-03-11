"Core recommendation engine orchestrating retrieval → prerank → rank → rerank."

import logging
import time
import numpy as np
from typing import Dict, Any

from candidate_generation.candidate_service import CandidateService
from services.feature_service import FeatureService
from services.ranking_service import RankingService
from services.preranking_service import PreRankingService
from reranking.reranking_service import RerankingService

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 128


class RecommendationEngine:

    def __init__(
        self,
        feature_service: FeatureService,
        candidate_service: CandidateService,
        ranking_service: RankingService,
        reranking_service: RerankingService,
        preranking_service: PreRankingService,
    ):
        self._feature_service = feature_service
        self._candidate_service = candidate_service
        self._ranking_service = ranking_service
        self._reranking_service = reranking_service
        self._preranking = preranking_service

    async def recommend(
        self,
        user_id: str,
        session_id: str,
        context: Dict[str, Any],
        top_k: int = 20
    ) -> Dict[str, Any]:

        start = time.time()

        user_features = await self._feature_service.get_user_features(user_id)

        # Use provided embedding or generate a deterministic fallback from user_id
        user_embedding = context.get("user_embedding")
        if user_embedding is None:
            rng = np.random.default_rng(abs(hash(user_id)) % (2**32))
            user_embedding = rng.random(EMBEDDING_DIM).astype(np.float32)
            logger.debug(f"Generated fallback embedding for user {user_id}")

        user_embedding = np.array(user_embedding, dtype=np.float32)

        candidates = await self._candidate_service.get_candidates(
            user_id=user_id,
            user_embedding=user_embedding,
            context=context,
        )

        if not candidates:
            logger.warning(f"No candidates found for user {user_id}")
            return {"recommendations": [], "metadata": {"latency_ms": 0}}

        preranked = await self._preranking.prerank(
            candidates,
            user_features,
            context,
        )

        video_ids = [c["video_id"] for c in preranked]

        video_features = await self._feature_service.get_batch_video_features(video_ids)

        ranked = self._ranking_service.rank(
            user_features,
            preranked,
        )

        final_results = await self._reranking_service.rerank(
            ranked,
            video_signals_map={},
            top_k=top_k,
        )

        latency = int((time.time() - start) * 1000)

        return {
            "recommendations": final_results,
            "metadata": {"latency_ms": latency},
        }