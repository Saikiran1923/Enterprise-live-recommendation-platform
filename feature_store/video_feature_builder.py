"""Builds video feature vectors from metadata and engagement signals."""

import math
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


class VideoFeatureBuilder:
    """
    Constructs feature vectors for videos using metadata,
    engagement statistics, and temporal features.
    """

    def __init__(self):
        pass

    def build(self, video_id: str, video_metadata: Dict[str, Any],
              video_signals: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        features.update(self._build_metadata_features(video_metadata))
        features.update(self._build_engagement_features(video_signals))
        features.update(self._build_freshness_features(video_metadata))
        return features

    def _build_metadata_features(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        duration = meta.get("duration_sec", 0)
        return {
            "video_duration_sec": duration,
            "video_duration_log": math.log1p(duration),
            "video_duration_bucket": self._duration_bucket(duration),
            "video_category": meta.get("category", "unknown"),
            "video_language": meta.get("language", "en"),
            "video_is_live": int(meta.get("is_live", False)),
            "video_has_subtitles": int(meta.get("has_subtitles", False)),
            "video_creator_follower_count_log": math.log1p(
                meta.get("creator_follower_count", 0)
            ),
            "video_tag_count": len(meta.get("tags", [])),
        }

    def _build_engagement_features(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        views = signals.get("views", 0)
        likes = signals.get("likes", 0)
        shares = signals.get("shares", 0)
        watch_sec = signals.get("watch_sec", 0)
        return {
            "video_view_count": views,
            "video_view_count_log": math.log1p(views),
            "video_like_rate": likes / max(views, 1),
            "video_share_rate": shares / max(views, 1),
            "video_avg_watch_sec": watch_sec / max(views, 1),
            "video_engagement_score": (likes * 3 + shares * 5 + views) / max(views + 1, 1),
        }

    def _build_freshness_features(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        upload_ts = meta.get("upload_timestamp", time.time())
        age_hours = (time.time() - upload_ts) / 3600
        freshness = math.exp(-age_hours / 48)  # exponential decay over 48h
        return {
            "video_age_hours": age_hours,
            "video_age_hours_log": math.log1p(age_hours),
            "video_freshness_score": freshness,
        }

    @staticmethod
    def _duration_bucket(duration_sec: float) -> int:
        if duration_sec < 60: return 0       # shorts
        if duration_sec < 600: return 1      # medium
        if duration_sec < 1800: return 2     # long
        return 3                              # very long
