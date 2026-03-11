"""Central registry for all feature definitions in the platform."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    SEQUENCE = "sequence"
    BOOLEAN = "boolean"


@dataclass
class FeatureDefinition:
    name: str
    feature_type: FeatureType
    description: str
    default_value: Any = None
    ttl_seconds: int = 300
    source: str = "online"
    tags: List[str] = field(default_factory=list)


class FeatureRegistry:
    """Singleton registry for all feature definitions."""

    _instance = None
    _features: Dict[str, FeatureDefinition] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_defaults()
        return cls._instance

    def _initialize_defaults(self) -> None:
        defaults = [
            FeatureDefinition("user_age_days", FeatureType.NUMERICAL, "Days since user joined", 0),
            FeatureDefinition("user_total_views", FeatureType.NUMERICAL, "Total videos viewed", 0),
            FeatureDefinition("user_avg_watch_rate", FeatureType.NUMERICAL, "Avg completion rate", 0.0),
            FeatureDefinition("user_preferred_categories", FeatureType.SEQUENCE, "Top categories", []),
            FeatureDefinition("user_embedding", FeatureType.EMBEDDING, "128-dim user embedding", []),
            FeatureDefinition("video_view_count", FeatureType.NUMERICAL, "Total video views", 0),
            FeatureDefinition("video_like_rate", FeatureType.NUMERICAL, "Like/view ratio", 0.0),
            FeatureDefinition("video_avg_watch_rate", FeatureType.NUMERICAL, "Avg watch completion", 0.0),
            FeatureDefinition("video_embedding", FeatureType.EMBEDDING, "128-dim video embedding", []),
            FeatureDefinition("video_age_hours", FeatureType.NUMERICAL, "Hours since upload", 0),
            FeatureDefinition("video_trending_score", FeatureType.NUMERICAL, "Real-time trending", 0.0),
        ]
        for f in defaults:
            self._features[f.name] = f

    def register(self, feature: FeatureDefinition) -> None:
        self._features[feature.name] = feature

    def get(self, name: str) -> Optional[FeatureDefinition]:
        return self._features.get(name)

    def list_all(self) -> List[FeatureDefinition]:
        return list(self._features.values())

    def list_by_type(self, ftype: FeatureType) -> List[FeatureDefinition]:
        return [f for f in self._features.values() if f.feature_type == ftype]
