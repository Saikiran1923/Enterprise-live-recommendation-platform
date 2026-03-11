"""Automated rollback controller for degraded model deployments."""

import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class RollbackController:
    """
    Monitors model performance and triggers rollbacks when
    metrics drop below configured thresholds.
    """

    def __init__(self, deployment_manager=None, model_registry=None):
        self._deployment_manager = deployment_manager
        self._model_registry = model_registry
        self._thresholds: Dict[str, float] = {
            "ctr_drop_threshold": 0.10,
            "latency_increase_threshold": 0.50,
            "error_rate_threshold": 0.05,
        }
        self._rollback_history: List[Dict[str, Any]] = []

    def should_rollback(self, current_metrics: Dict[str, float],
                        baseline_metrics: Dict[str, float]) -> tuple:
        """Check if rollback is warranted based on metric comparison."""
        reasons = []
        if baseline_metrics.get("ctr", 1) > 0:
            ctr_drop = (baseline_metrics["ctr"] - current_metrics.get("ctr", 0)) / \
                       baseline_metrics["ctr"]
            if ctr_drop > self._thresholds["ctr_drop_threshold"]:
                reasons.append(f"CTR dropped {ctr_drop:.1%}")

        if current_metrics.get("error_rate", 0) > self._thresholds["error_rate_threshold"]:
            reasons.append(f"Error rate {current_metrics['error_rate']:.1%}")

        latency_increase = (
            current_metrics.get("p99_latency_ms", 0) -
            baseline_metrics.get("p99_latency_ms", 0)
        ) / max(baseline_metrics.get("p99_latency_ms", 1), 1)
        if latency_increase > self._thresholds["latency_increase_threshold"]:
            reasons.append(f"Latency increased {latency_increase:.1%}")

        return len(reasons) > 0, reasons

    def rollback(self, deployment_id: str, reason: str) -> None:
        """Execute rollback for a deployment."""
        logger.warning(f"Rolling back deployment {deployment_id}: {reason}")
        if self._deployment_manager:
            dep = self._deployment_manager.get_deployment_status(deployment_id)
            if dep:
                dep["status"] = "rolled_back"
                dep["rollback_reason"] = reason
                dep["rolled_back_at"] = time.time()
        self._rollback_history.append({
            "deployment_id": deployment_id,
            "reason": reason,
            "timestamp": time.time(),
        })

    def get_rollback_history(self) -> List[Dict[str, Any]]:
        return self._rollback_history.copy()
