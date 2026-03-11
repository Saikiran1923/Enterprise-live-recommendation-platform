"""Service layer for serving embeddings with caching and batch support."""

import asyncio
import logging
from typing import Dict, List, Optional
import numpy as np

from embeddings.user_embedding_model import UserEmbeddingModel
from embeddings.video_embedding_model import VideoEmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Unified service for fetching user and video embeddings.
    Provides in-memory caching and async batch retrieval.
    """

    def __init__(self, user_model: UserEmbeddingModel,
                 video_model: VideoEmbeddingModel,
                 cache_size: int = 50000):
        self._user_model = user_model
        self._video_model = video_model
        self._user_cache: Dict[str, np.ndarray] = {}
        self._video_cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size

    async def get_user_embedding(self, user_id: str) -> np.ndarray:
        if user_id not in self._user_cache:
            emb = self._user_model.get_embedding(user_id)
            self._maybe_cache(self._user_cache, user_id, emb)
        return self._user_cache[user_id]

    async def get_video_embedding(self, video_id: str) -> np.ndarray:
        if video_id not in self._video_cache:
            emb = self._video_model.get_embedding(video_id)
            self._maybe_cache(self._video_cache, video_id, emb)
        return self._video_cache[video_id]

    async def get_batch_user_embeddings(self, user_ids: List[str]) -> np.ndarray:
        embs = await asyncio.gather(*[self.get_user_embedding(uid) for uid in user_ids])
        return np.vstack(embs)

    async def get_batch_video_embeddings(self, video_ids: List[str]) -> np.ndarray:
        embs = await asyncio.gather(*[self.get_video_embedding(vid) for vid in video_ids])
        return np.vstack(embs)

    def invalidate_user(self, user_id: str) -> None:
        self._user_cache.pop(user_id, None)

    def invalidate_video(self, video_id: str) -> None:
        self._video_cache.pop(video_id, None)

    def _maybe_cache(self, cache: Dict, key: str, value: np.ndarray) -> None:
        if len(cache) >= self._cache_size:
            oldest = next(iter(cache))
            del cache[oldest]
        cache[key] = value

    def get_stats(self) -> Dict:
        return {
            "user_cache_size": len(self._user_cache),
            "video_cache_size": len(self._video_cache),
            "max_cache_size": self._cache_size,
        }
