"""Orchestrates multiple candidate generators and merges results."""

import logging
import asyncio
from typing import List, Dict, Tuple, Any, Optional, Set

logger = logging.getLogger(__name__)


class CandidateService:
    """
    Merges candidates from multiple sources (Two-Tower, CF, MF, trending)
    with deduplication and score normalization.
    """

    def __init__(self, two_tower=None, collab_filter=None,
                 matrix_fact=None, top_k: int = 500):
        self._two_tower = two_tower
        self._cf = collab_filter
        self._mf = matrix_fact
        self.top_k = top_k

    async def get_candidates(self, user_id: str,
                             user_embedding,
                             exclude_ids: Optional[Set[str]] = None,
                             context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch and merge candidates from all sources.
        Returns list of candidate dicts with video_id and retrieval_score.
        """
        tasks = []
        if self._two_tower is not None:
            tasks.append(self._get_two_tower_candidates(user_embedding, exclude_ids))
        if self._cf is not None:
            tasks.append(self._get_cf_candidates(user_id, exclude_ids))
        if self._mf is not None:
            tasks.append(self._get_mf_candidates(user_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged = self._merge_candidates(results)
        return merged[:self.top_k]

    async def _get_two_tower_candidates(self, user_embedding,
                                         exclude_ids) -> List[Tuple[str, float]]:
        try:
            return self._two_tower.retrieve(user_embedding, exclude_ids=list(exclude_ids or []))
        except Exception as e:
            logger.error(f"Two-tower retrieval error: {e}")
            return []

    async def _get_cf_candidates(self, user_id: str,
                                  exclude_ids) -> List[Tuple[str, float]]:
        try:
            return self._cf.recommend(user_id, exclude_ids=exclude_ids)
        except Exception as e:
            logger.error(f"CF retrieval error: {e}")
            return []

    async def _get_mf_candidates(self, user_id: str) -> List[Tuple[str, float]]:
        try:
            return self._mf.recommend(user_id)
        except Exception as e:
            logger.error(f"MF retrieval error: {e}")
            return []

    def _merge_candidates(self, results) -> List[Dict[str, Any]]:
        """Merge and deduplicate candidates, normalizing scores per source."""
        seen: Set[str] = set()
        merged: List[Dict[str, Any]] = []

        for source_idx, result in enumerate(results):
            if isinstance(result, Exception) or not result:
                continue
            source_name = ["two_tower", "cf", "mf"][min(source_idx, 2)]
            max_score = max(s for _, s in result) if result else 1.0
            for video_id, score in result:
                if video_id not in seen:
                    seen.add(video_id)
                    merged.append({
                        "video_id": video_id,
                        "retrieval_score": score / max(max_score, 1e-8),
                        "source": source_name,
                    })

        merged.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return merged
