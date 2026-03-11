"""Service for fetching and caching features for recommendations."""

import logging
from typing import Dict, Any, List

from feature_store.user_feature_builder import UserFeatureBuilder
from feature_store.video_feature_builder import VideoFeatureBuilder
from feature_store.session_feature_builder import SessionFeatureBuilder

logger = logging.getLogger(__name__)


class FeatureService:
    def __init__(self, feature_store):
        self._store = feature_store
        self._user_builder = UserFeatureBuilder(feature_store)
        self._video_builder = VideoFeatureBuilder()
        self._session_builder = SessionFeatureBuilder()

    async def get_user_features(self, user_id: str) -> Dict[str, Any]:
        feature_names = [
            "user_age_days_log", "user_total_views_log", "user_avg_watch_sec",
            "user_like_rate", "user_skip_rate", "user_engagement_score",
        ]
        return await self._store.get_user_features(user_id, feature_names)

    async def get_video_features(self, video_id: str) -> Dict[str, Any]:
        feature_names = [
            "video_view_count_log", "video_like_rate", "video_avg_watch_sec",
            "video_freshness_score", "video_trending_score", "video_duration_log",
            "video_category",
        ]
        return await self._store.get_video_features(video_id, feature_names)

    async def get_batch_video_features(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        result = {}
        for vid in video_ids:
            result[vid] = await self.get_video_features(vid)
        return result
