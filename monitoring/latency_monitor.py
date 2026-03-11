"""Monitors and alerts on API latency."""

import time
import logging
from typing import Dict, Any, Callable
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class LatencyMonitor:
    """Tracks endpoint latency with SLA alerting."""

    def __init__(self, metrics_collector=None,
                 sla_ms: float = 100.0):
        self._metrics = metrics_collector
        self.sla_ms = sla_ms
        self._sla_violations = 0
        self._total_requests = 0

    @asynccontextmanager
    async def track(self, endpoint: str):
        start = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start) * 1000
            self._total_requests += 1
            if self._metrics:
                self._metrics.histogram("recommendation_latency_ms", latency_ms,
                                        {"endpoint": endpoint})
            if latency_ms > self.sla_ms:
                self._sla_violations += 1
                logger.warning(f"SLA violation on {endpoint}: {latency_ms:.1f}ms")

    def get_violation_rate(self) -> float:
        return self._sla_violations / max(self._total_requests, 1)
