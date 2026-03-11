"""Online feature store backed by Redis for low-latency feature serving."""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """
    Redis-backed online feature store.
    Supports batch get/set, TTL-based expiry, and feature versioning.
    """

    def __init__(self, redis_client=None, key_prefix: str = "fs"):
        self._redis = redis_client
        self._prefix = key_prefix
        self._local_cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # local cache TTL in seconds
        self._cache_timestamps: Dict[str, float] = {}

    def _make_key(self, entity_type: str, entity_id: str, feature_name: str) -> str:
        return f"{self._prefix}:{entity_type}:{entity_id}:{feature_name}"

    async def get_user_features(self, user_id: str, feature_names: List[str]) -> Dict[str, Any]:
        """Fetch multiple user features in a single round trip."""
        return await self._batch_get("user", user_id, feature_names)

    async def get_video_features(self, video_id: str, feature_names: List[str]) -> Dict[str, Any]:
        """Fetch multiple video features in a single round trip."""
        return await self._batch_get("video", video_id, feature_names)

    async def _batch_get(self, entity_type: str, entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
        results = {}
        missing = []

        for name in feature_names:
            key = self._make_key(entity_type, entity_id, name)
            cached = self._get_local(key)
            if cached is not None:
                results[name] = cached
            else:
                missing.append((name, key))

        if missing and self._redis:
            try:
                keys = [k for _, k in missing]
                values = await self._redis.mget(*keys)
                for (name, key), val in zip(missing, values):
                    if val is not None:
                        parsed = json.loads(val)
                        results[name] = parsed
                        self._set_local(key, parsed)
            except Exception as e:
                logger.warning(f"Redis batch get failed: {e}")

        return results

    async def set_user_features(self, user_id: str, features: Dict[str, Any], ttl: int = 300) -> None:
        await self._batch_set("user", user_id, features, ttl)

    async def set_video_features(self, video_id: str, features: Dict[str, Any], ttl: int = 600) -> None:
        await self._batch_set("video", video_id, features, ttl)

    async def _batch_set(self, entity_type: str, entity_id: str,
                         features: Dict[str, Any], ttl: int) -> None:
        if not self._redis:
            for name, val in features.items():
                key = self._make_key(entity_type, entity_id, name)
                self._set_local(key, val)
            return
        try:
            pipe = self._redis.pipeline()
            for name, val in features.items():
                key = self._make_key(entity_type, entity_id, name)
                pipe.setex(key, ttl, json.dumps(val))
                self._set_local(key, val)
            await pipe.execute()
        except Exception as e:
            logger.error(f"Redis batch set failed: {e}")

    def _get_local(self, key: str) -> Optional[Any]:
        ts = self._cache_timestamps.get(key, 0)
        if time.time() - ts > self._cache_ttl:
            return None
        return self._local_cache.get(key)

    def _set_local(self, key: str, value: Any) -> None:
        self._local_cache[key] = value
        self._cache_timestamps[key] = time.time()

    async def delete_user_features(self, user_id: str) -> None:
        pattern = f"{self._prefix}:user:{user_id}:*"
        if self._redis:
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
