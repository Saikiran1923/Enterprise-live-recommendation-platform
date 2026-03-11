"""System health monitoring for all platform components."""

import time
import asyncio
import logging
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors health of all platform dependencies and services."""

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, Dict[str, Any]] = {}

    def register_check(self, name: str, check_fn: Callable) -> None:
        self._checks[name] = check_fn

    async def run_checks(self) -> Dict[str, Any]:
        results = {}
        for name, check in self._checks.items():
            start = time.time()
            try:
                result = await check()
                results[name] = {
                    "status": "healthy",
                    "details": result,
                    "latency_ms": (time.time() - start) * 1000,
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "latency_ms": (time.time() - start) * 1000,
                }
        self._last_results = results
        return results

    def is_healthy(self) -> bool:
        if not self._last_results:
            return True
        return all(v.get("status") == "healthy" for v in self._last_results.values())

    def get_status(self) -> Dict[str, Any]:
        return {
            "healthy": self.is_healthy(),
            "checks": self._last_results,
            "timestamp": time.time(),
        }
