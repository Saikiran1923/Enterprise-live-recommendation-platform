"""Real-time stream processor for aggregating user and video signals."""

import asyncio
import logging
from typing import Dict, Any, List, Deque
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class UserSignals:
    user_id: str
    view_count: int = 0
    total_watch_sec: float = 0.0
    like_count: int = 0
    share_count: int = 0
    skip_count: int = 0
    last_active: float = field(default_factory=time.time)
    recent_categories: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    recent_videos: Deque[str] = field(default_factory=lambda: deque(maxlen=50))


class StreamProcessor:
    """
    Processes streaming events and maintains real-time aggregated signals.
    Used to update the online feature store with the latest user/video signals.
    """

    def __init__(self, flush_interval: float = 5.0):
        self._user_signals: Dict[str, UserSignals] = {}
        self._video_signals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"views": 0, "likes": 0, "shares": 0, "watch_sec": 0.0, "last_viewed": 0.0}
        )
        self._flush_interval = flush_interval
        self._event_buffer: List[Dict[str, Any]] = []
        self._flush_callbacks: List[Any] = []

    def add_flush_callback(self, fn) -> None:
        self._flush_callbacks.append(fn)

    async def process_event(self, event: Dict[str, Any]) -> None:
        """Process a single event and update in-memory signals."""
        self._event_buffer.append(event)
        event_type = event.get("event_type")
        user_id = event.get("user_id")
        video_id = event.get("video_id")

        if not user_id:
            return

        signals = self._user_signals.setdefault(user_id, UserSignals(user_id=user_id))
        signals.last_active = event.get("timestamp", time.time())

        if event_type == "video_view":
            signals.view_count += 1
            signals.total_watch_sec += event.get("watch_duration_sec", 0)
            if video_id:
                signals.recent_videos.append(video_id)
            if event.get("category"):
                signals.recent_categories.append(event["category"])
            if video_id:
                vs = self._video_signals[video_id]
                vs["views"] += 1
                vs["watch_sec"] += event.get("watch_duration_sec", 0)
                vs["last_viewed"] = signals.last_active

        elif event_type == "video_like":
            signals.like_count += 1
            if video_id:
                self._video_signals[video_id]["likes"] += 1

        elif event_type == "video_share":
            signals.share_count += 1
            if video_id:
                self._video_signals[video_id]["shares"] += 1

        elif event_type == "video_skip":
            signals.skip_count += 1

    async def flush_loop(self) -> None:
        """Periodically flush buffered signals to the feature store."""
        while True:
            await asyncio.sleep(self._flush_interval)
            if self._event_buffer:
                await self._flush()

    async def _flush(self) -> None:
        """Flush current signals to all registered callbacks."""
        snapshot = {
            "user_signals": {k: v.__dict__ for k, v in self._user_signals.items()},
            "video_signals": dict(self._video_signals),
            "event_count": len(self._event_buffer),
        }
        self._event_buffer.clear()
        for cb in self._flush_callbacks:
            try:
                await cb(snapshot)
            except Exception as e:
                logger.error(f"Flush callback error: {e}")

    def get_user_signals(self, user_id: str) -> Dict[str, Any]:
        s = self._user_signals.get(user_id)
        return s.__dict__ if s else {}

    def get_video_signals(self, video_id: str) -> Dict[str, Any]:
        return dict(self._video_signals.get(video_id, {}))
