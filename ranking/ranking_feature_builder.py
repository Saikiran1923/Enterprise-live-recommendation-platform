"""Builds cross features for the ranking model."""

import math
from typing import Dict, Any


class RankingFeatureBuilder:
    """Constructs cross features combining user, video, and context signals."""

    def build(self, user_features: Dict[str, Any],
              video_features: Dict[str, Any],
              session_features: Dict[str, Any],
              retrieval_score: float = 0.0) -> Dict[str, Any]:
        features = {}
        features.update(self._extract_user(user_features))
        features.update(self._extract_video(video_features))
        features.update(self._extract_session(session_features))
        features.update(self._cross_features(user_features, video_features))
        features["retrieval_score"] = retrieval_score
        return features

    def _extract_user(self, uf: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_age_days_log": uf.get("user_age_days_log", 0),
            "user_total_views_log": uf.get("user_total_views_log", 0),
            "user_avg_watch_sec": uf.get("user_avg_watch_sec", 0),
            "user_like_rate": uf.get("user_like_rate", 0),
            "user_skip_rate": uf.get("user_skip_rate", 0),
            "user_engagement_score": uf.get("user_engagement_score", 0),
        }

    def _extract_video(self, vf: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "video_view_count_log": vf.get("video_view_count_log", 0),
            "video_like_rate": vf.get("video_like_rate", 0),
            "video_avg_watch_sec": vf.get("video_avg_watch_sec", 0),
            "video_freshness_score": vf.get("video_freshness_score", 0),
            "video_trending_score": vf.get("video_trending_score", 0),
            "video_duration_log": vf.get("video_duration_log", 0),
        }

    def _extract_session(self, sf: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "session_length_min": sf.get("session_length_min", 0),
            "session_skip_rate": sf.get("session_skip_rate", 0),
        }

    def _cross_features(self, uf: Dict[str, Any], vf: Dict[str, Any]) -> Dict[str, Any]:
        video_cat = vf.get("video_category", "")
        affinity = uf.get(f"user_cat_{video_cat}_affinity", 0.0)
        user_avg = uf.get("user_avg_watch_sec", 0)
        video_dur = math.exp(vf.get("video_duration_log", 0)) - 1
        return {
            "user_video_category_match": affinity,
            "watch_duration_ratio": user_avg / max(video_dur, 1),
        }
