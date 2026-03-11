"""Builds session-level features capturing real-time user context."""

import time
import math
from typing import Dict, Any, List, Deque
from collections import deque


class SessionFeatureBuilder:
    """
    Builds features from the current user session to capture
    real-time intent and context for recommendations.
    """

    def __init__(self, session_window: int = 10):
        self._session_window = session_window

    def build(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        features.update(self._build_recency_features(session_data))
        features.update(self._build_session_engagement(session_data))
        features.update(self._build_session_intent(session_data))
        return features

    def _build_recency_features(self, session: Dict[str, Any]) -> Dict[str, Any]:
        start_ts = session.get("start_timestamp", time.time())
        session_len = (time.time() - start_ts) / 60
        return {
            "session_length_min": session_len,
            "session_length_log": math.log1p(session_len),
            "session_videos_watched": session.get("videos_watched", 0),
            "session_hour_of_day": time.localtime().tm_hour,
            "session_day_of_week": time.localtime().tm_wday,
        }

    def _build_session_engagement(self, session: Dict[str, Any]) -> Dict[str, Any]:
        videos = session.get("videos_watched", 0)
        watch_sec = session.get("total_watch_sec", 0)
        likes = session.get("session_likes", 0)
        skips = session.get("session_skips", 0)
        return {
            "session_avg_watch_sec": watch_sec / max(videos, 1),
            "session_like_rate": likes / max(videos, 1),
            "session_skip_rate": skips / max(videos, 1),
            "session_engagement_level": self._engagement_level(likes, skips, videos),
        }

    def _build_session_intent(self, session: Dict[str, Any]) -> Dict[str, Any]:
        recent_cats = session.get("recent_categories", [])
        if not recent_cats:
            return {"session_primary_category": "unknown", "session_category_diversity": 0.0}
        from collections import Counter
        counts = Counter(recent_cats[-self._session_window:])
        top_cat = counts.most_common(1)[0][0]
        diversity = len(counts) / max(len(recent_cats), 1)
        return {
            "session_primary_category": top_cat,
            "session_category_diversity": diversity,
        }

    @staticmethod
    def _engagement_level(likes: int, skips: int, views: int) -> float:
        if views == 0:
            return 0.5
        score = (likes * 2 - skips + views) / (views * 3 + 1)
        return max(0.0, min(1.0, score))
