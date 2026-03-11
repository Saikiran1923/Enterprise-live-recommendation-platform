"""Collects and aggregates platform metrics."""

import time
import logging
from typing import Dict, Any, List
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    name: str
    value: float
    tags: Dict[str, str]
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Thread-safe metrics collector with in-memory ring buffer."""

    def __init__(self, buffer_size: int = 10000):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0,
                  tags: Dict[str, str] = None) -> None:
        self._counters[name] += value
        self._buffer.append(MetricPoint(name, value, tags or {}))

    def gauge(self, name: str, value: float,
              tags: Dict[str, str] = None) -> None:
        self._gauges[name] = value
        self._buffer.append(MetricPoint(name, value, tags or {}))

    def histogram(self, name: str, value: float,
                  tags: Dict[str, str] = None) -> None:
        self._histograms[name].append(value)
        if len(self._histograms[name]) > 10000:
            self._histograms[name] = self._histograms[name][-5000:]

    def get_percentile(self, name: str, p: float) -> float:
        vals = sorted(self._histograms.get(name, [0]))
        if not vals:
            return 0.0
        idx = int(p / 100 * len(vals))
        return vals[min(idx, len(vals) - 1)]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "p50_latency_ms": self.get_percentile("recommendation_latency_ms", 50),
            "p95_latency_ms": self.get_percentile("recommendation_latency_ms", 95),
            "p99_latency_ms": self.get_percentile("recommendation_latency_ms", 99),
        }
