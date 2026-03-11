"""Logs experiment impressions and events for offline analysis."""

import json
import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Logs experiment events to persistent storage for offline analysis."""

    def __init__(self, event_store=None):
        self._event_store = event_store
        self._buffer: List[Dict[str, Any]] = []
        self._flush_size = 100

    async def log_impression(self, user_id: str, experiment_id: str,
                              variant: str, recommendations: List[str]) -> None:
        event = {
            "type": "experiment_impression",
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant,
            "recommendations": recommendations[:10],
            "timestamp": time.time(),
        }
        await self._buffer_event(event)

    async def log_outcome(self, user_id: str, experiment_id: str,
                          variant: str, metric_name: str, value: float) -> None:
        event = {
            "type": "experiment_outcome",
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant,
            "metric_name": metric_name,
            "value": value,
            "timestamp": time.time(),
        }
        await self._buffer_event(event)

    async def _buffer_event(self, event: Dict[str, Any]) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self._flush_size:
            await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        if self._event_store:
            try:
                await self._event_store.write_batch(self._buffer)
            except Exception as e:
                logger.error(f"Failed to flush experiment logs: {e}")
        self._buffer.clear()
