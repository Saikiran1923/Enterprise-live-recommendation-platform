"""Event persistence store for audit trails and offline training."""

import json
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class EventStore:
    """Stores events for training data collection and audit."""

    def __init__(self, db=None):
        self._db = db
        self._buffer: List[Dict[str, Any]] = []

    async def write(self, event: Dict[str, Any]) -> None:
        event.setdefault("stored_at", time.time())
        self._buffer.append(event)
        if len(self._buffer) >= 500:
            await self.flush()

    async def write_batch(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            event.setdefault("stored_at", time.time())
        self._buffer.extend(events)
        if len(self._buffer) >= 500:
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        logger.debug(f"Flushing {len(self._buffer)} events to store")
        # In production: write to Kafka / S3 / PostgreSQL
        self._buffer.clear()

    async def query_user_events(self, user_id: str,
                                 event_type: Optional[str] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        # In production: query from database
        return []
