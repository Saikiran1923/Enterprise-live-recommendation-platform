"""Computes statistical metrics for A/B experiment analysis."""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExperimentMetrics:
    """Computes and stores A/B test metrics with statistical significance."""

    def __init__(self):
        self._metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record(self, experiment_id: str, variant: str,
               metric_name: str, value: float) -> None:
        key = f"{experiment_id}:{variant}"
        self._metrics[key][metric_name].append(value)

    def compute_summary(self, experiment_id: str, metric_name: str) -> Dict[str, Any]:
        """Compute per-variant statistics for a metric."""
        summary = {}
        for variant in self._get_variants(experiment_id):
            key = f"{experiment_id}:{variant}"
            values = self._metrics[key].get(metric_name, [])
            if values:
                summary[variant] = self._stats(values)
        return summary

    def compute_lift(self, experiment_id: str, metric_name: str,
                     control: str = "control",
                     treatment: str = "treatment") -> Dict[str, Any]:
        """Compute lift and statistical significance between variants."""
        ctrl_key = f"{experiment_id}:{control}"
        trt_key = f"{experiment_id}:{treatment}"
        ctrl_vals = self._metrics[ctrl_key].get(metric_name, [])
        trt_vals = self._metrics[trt_key].get(metric_name, [])

        if not ctrl_vals or not trt_vals:
            return {"error": "insufficient data"}

        ctrl_mean = sum(ctrl_vals) / len(ctrl_vals)
        trt_mean = sum(trt_vals) / len(trt_vals)
        lift = (trt_mean - ctrl_mean) / max(abs(ctrl_mean), 1e-10)
        p_value = self._welch_t_test(ctrl_vals, trt_vals)

        return {
            "control_mean": ctrl_mean,
            "treatment_mean": trt_mean,
            "relative_lift": lift,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_control": len(ctrl_vals),
            "n_treatment": len(trt_vals),
        }

    def _stats(self, values: List[float]) -> Dict[str, float]:
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        return {"mean": mean, "std": math.sqrt(variance), "n": n}

    def _welch_t_test(self, a: List[float], b: List[float]) -> float:
        """Simplified Welch's t-test p-value approximation."""
        na, nb = len(a), len(b)
        ma = sum(a) / na
        mb = sum(b) / nb
        va = sum((x - ma) ** 2 for x in a) / max(na - 1, 1)
        vb = sum((x - mb) ** 2 for x in b) / max(nb - 1, 1)
        se = math.sqrt(va / na + vb / nb)
        if se == 0:
            return 1.0
        t = abs(ma - mb) / se
        # Approximate p-value (two-tailed) via logistic transform
        return 2 * (1 / (1 + math.exp(t * 0.8 - 2)))

    def _get_variants(self, experiment_id: str) -> List[str]:
        return [k.split(":", 1)[1] for k in self._metrics
                if k.startswith(f"{experiment_id}:")]
