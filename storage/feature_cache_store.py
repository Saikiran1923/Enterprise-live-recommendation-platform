"Redis-backed feature cache store."

import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class FeatureCacheStore:
    "Wraps Redis for feature caching with automatic serialization."

    def __init__(self, redis_client=None, prefix: str = "fc"):
        self._redis = redis_client
        self._prefix = prefix
        self._local: Dict[str, Any] = {}

    def _key(self, entity_type: str, entity_id: str) -> str:
        return f"{self._prefix}:{entity_type}:{entity_id}"

    async def get(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        key = self._key(entity_type, entity_id)

        if key in self._local:
            return self._local[key]

        if self._redis:
            val = await self._redis.get(key)
            if val:
                return json.loads(val)

        return None

    async def set(self, entity_type: str, entity_id: str,
                  features: Dict[str, Any], ttl: int = 300) -> None:

        key = self._key(entity_type, entity_id)

        self._local[key] = features

        if self._redis:
            await self._redis.setex(key, ttl, json.dumps(features))

    async def delete(self, entity_type: str, entity_id: str) -> None:
        key = self._key(entity_type, entity_id)

        self._local.pop(key, None)

        if self._redis:
            await self._redis.delete(key)

    async def get_user_features(self, user_id: str, feature_names=None) -> Dict[str, Any]:
        """Fetch user features."""
        features = await self.get("user", user_id)

        if not features:
            return {}

        if feature_names:
            return {k: features.get(k) for k in feature_names}

        return features

    async def get_video_features(self, video_ids: List[str], feature_names=None) -> Dict[str, Dict[str, Any]]:
        """Fetch video features."""
        results: Dict[str, Dict[str, Any]] = {}

        for vid in video_ids:
            features = await self.get("video", vid)

            if not features:
                results[vid] = {}
                continue

            if feature_names:
                results[vid] = {k: features.get(k) for k in feature_names}
            else:
                results[vid] = features

        return results