"""Tracks real-time session state for live recommendation updates."""

import time
import logging
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    user_id: str
    start_time: float = field(default_factory=time.time)
    videos_watched: int = 0
    total_watch_sec: float = 0.0
    session_likes: int = 0
    session_shares: int = 0
    session_skips: int = 0
    recent_videos: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    recent_categories: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    last_event_time: float = field(default_factory=time.time)
    is_active: bool = True
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def session_duration_min(self) -> float:
        return (time.time() - self.start_time) / 60

    @property
    def avg_watch_rate(self) -> float:
        if self.videos_watched == 0:
            return 0.5
        return min(1.0, self.total_watch_sec / max(self.videos_watched * 300, 1))

    def to_features(self) -> Dict[str, Any]:
        return {
            "session_length_min": self.session_duration_min,
            "session_videos_watched": self.videos_watched,
            "session_avg_watch_sec": self.total_watch_sec / max(self.videos_watched, 1),
            "session_like_rate": self.session_likes / max(self.videos_watched, 1),
            "session_skip_rate": self.session_skips / max(self.videos_watched, 1),
            "session_hour_of_day": time.localtime().tm_hour,
            "session_day_of_week": time.localtime().tm_wday,
            "recent_categories": list(self.recent_categories),
        }


class SessionStateTracker:
    """Manages in-memory session states with TTL eviction."""

    def __init__(self, session_ttl_minutes: float = 30.0):
        self._sessions: Dict[str, SessionState] = {}
        self._ttl = session_ttl_minutes * 60

    def get_or_create(self, session_id: str, user_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(
                session_id=session_id, user_id=user_id
            )
        return self._sessions[session_id]

    def update(self, session_id: str, event: Dict[str, Any]) -> Optional[SessionState]:
        session = self._sessions.get(session_id)
        if not session:
            return None

        event_type = event.get("event_type")
        session.last_event_time = time.time()

        if event_type == "video_view":
            session.videos_watched += 1
            session.total_watch_sec += event.get("watch_duration_sec", 0)
            if event.get("video_id"):
                session.recent_videos.append(event["video_id"])
            if event.get("category"):
                session.recent_categories.append(event["category"])
        elif event_type == "video_like":
            session.session_likes += 1
        elif event_type == "video_share":
            session.session_shares += 1
        elif event_type == "video_skip":
            session.session_skips += 1
        elif event_type == "session_end":
            session.is_active = False

        return session

    def evict_expired(self) -> int:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.last_event_time > self._ttl]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def get_active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.is_active)
