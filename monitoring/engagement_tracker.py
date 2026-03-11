"""Tracks real-time engagement metrics for recommendations."""

import time
import logging
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class EngagementTracker:
    """Tracks CTR, watch time, and engagement for recommendation slots."""

    def __init__(self):
        self._impressions: Dict[str, int] = defaultdict(int)
        self._clicks: Dict[str, int] = defaultdict(int)
        self._watch_time: Dict[str, float] = defaultdict(float)
        self._position_clicks: Dict[int, int] = defaultdict(int)
        self._position_impressions: Dict[int, int] = defaultdict(int)

    def record_impression(self, recommendation_id: str, video_id: str,
                          position: int, model_version: str) -> None:
        key = f"{model_version}:{video_id}"
        self._impressions[key] += 1
        self._position_impressions[position] += 1

    def record_click(self, recommendation_id: str, video_id: str,
                     position: int, model_version: str) -> None:
        key = f"{model_version}:{video_id}"
        self._clicks[key] += 1
        self._position_clicks[position] += 1

    def record_watch_time(self, video_id: str, watch_sec: float,
                          model_version: str) -> None:
        key = f"{model_version}:{video_id}"
        self._watch_time[key] += watch_sec

    def get_ctr(self, model_version: str = "") -> float:
        total_imp = sum(v for k, v in self._impressions.items()
                        if model_version in k)
        total_clk = sum(v for k, v in self._clicks.items()
                        if model_version in k)
        return total_clk / max(total_imp, 1)

    def get_position_ctr(self) -> Dict[int, float]:
        return {
            pos: self._position_clicks.get(pos, 0) / max(imp, 1)
            for pos, imp in self._position_impressions.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "overall_ctr": self.get_ctr(),
            "position_ctr": self.get_position_ctr(),
            "total_watch_sec": sum(self._watch_time.values()),
        }
