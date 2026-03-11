"""Event schemas for the recommendation platform ingestion pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class EventType(str, Enum):
    VIDEO_VIEW = "video_view"
    VIDEO_LIKE = "video_like"
    VIDEO_SHARE = "video_share"
    VIDEO_COMMENT = "video_comment"
    VIDEO_SKIP = "video_skip"
    VIDEO_COMPLETE = "video_complete"
    SEARCH_QUERY = "search_query"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    RECOMMENDATION_IMPRESSION = "recommendation_impression"
    RECOMMENDATION_CLICK = "recommendation_click"


@dataclass
class BaseEvent:
    event_id: str
    user_id: str
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    platform: str = "web"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "platform": self.platform,
            "metadata": self.metadata,
        }


@dataclass
class VideoEvent(BaseEvent):
    video_id: Optional[str] = None
    watch_duration_sec: float = 0.0
    video_duration_sec: float = 0.0
    completion_rate: float = 0.0
    creator_id: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    position_in_feed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "video_id": self.video_id,
            "watch_duration_sec": self.watch_duration_sec,
            "video_duration_sec": self.video_duration_sec,
            "completion_rate": self.completion_rate,
            "creator_id": self.creator_id,
            "category": self.category,
            "tags": self.tags,
            "position_in_feed": self.position_in_feed,
        })
        return d


@dataclass
class RecommendationEvent(BaseEvent):
    recommendation_id: str = ""
    video_id: Optional[str] = None
    rank_position: int = 0
    model_version: str = ""
    experiment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "recommendation_id": self.recommendation_id,
            "video_id": self.video_id,
            "rank_position": self.rank_position,
            "model_version": self.model_version,
            "experiment_id": self.experiment_id,
        })
        return d


@dataclass
class SessionEvent(BaseEvent):
    device_type: str = "unknown"
    app_version: str = ""
    country: str = ""
    language: str = "en"
    total_watch_time: float = 0.0
    videos_watched: int = 0
