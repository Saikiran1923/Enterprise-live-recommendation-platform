"""Builds user feature vectors from raw signals and history."""

import logging
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class UserFeatureBuilder:
    """
    Constructs feature vectors for users from raw event data,
    aggregated signals, and profile information.
    """

    CATEGORY_VOCAB = [
        "gaming", "music", "sports", "news", "education",
        "comedy", "lifestyle", "tech", "cooking", "travel",
    ]

    def __init__(self, feature_store=None):
        self._feature_store = feature_store

    def build(self, user_id: str, user_profile: Dict[str, Any],
              user_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete user feature dict."""
        features = {}
        features.update(self._build_profile_features(user_profile))
        features.update(self._build_engagement_features(user_signals))
        features.update(self._build_preference_features(user_signals))
        features.update(self._build_session_features(user_signals))
        return features

    def _build_profile_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).timestamp()
        created_at = profile.get("created_at", now)
        age_days = max(0, (now - created_at) / 86400)
        return {
            "user_age_days": age_days,
            "user_age_days_log": math.log1p(age_days),
            "user_country": profile.get("country", "unknown"),
            "user_language": profile.get("language", "en"),
            "user_is_creator": int(profile.get("is_creator", False)),
            "user_subscription_tier": profile.get("subscription_tier", 0),
        }

    def _build_engagement_features(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        views = signals.get("view_count", 0)
        watch_sec = signals.get("total_watch_sec", 0)
        likes = signals.get("like_count", 0)
        shares = signals.get("share_count", 0)
        skips = signals.get("skip_count", 0)
        total_actions = views + likes + shares + skips + 1
        return {
            "user_total_views": views,
            "user_total_views_log": math.log1p(views),
            "user_total_watch_hours": watch_sec / 3600,
            "user_avg_watch_sec": watch_sec / max(views, 1),
            "user_like_rate": likes / max(views, 1),
            "user_share_rate": shares / max(views, 1),
            "user_skip_rate": skips / max(views, 1),
            "user_engagement_score": (likes * 3 + shares * 5 + views) / total_actions,
        }

    def _build_preference_features(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        recent_cats = list(signals.get("recent_categories", []))
        cat_counts = {c: 0 for c in self.CATEGORY_VOCAB}
        for c in recent_cats:
            if c in cat_counts:
                cat_counts[c] += 1
        total = sum(cat_counts.values()) + 1
        return {
            f"user_cat_{c}_affinity": count / total
            for c, count in cat_counts.items()
        }

    def _build_session_features(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        import time
        last_active = signals.get("last_active", 0)
        hours_since = (time.time() - last_active) / 3600
        return {
            "user_hours_since_active": hours_since,
            "user_is_active": int(hours_since < 1),
        }
