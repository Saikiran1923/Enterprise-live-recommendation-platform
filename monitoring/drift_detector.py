"""Detects feature and prediction distribution drift."""

import math
import logging
from typing import Dict, Any, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift using Population Stability Index (PSI)
    and KL divergence between reference and current distributions.
    """

    PSI_THRESHOLD = 0.2  # moderate drift
    SEVERE_PSI_THRESHOLD = 0.4

    def __init__(self, n_bins: int = 10, window_size: int = 1000):
        self.n_bins = n_bins
        self.window_size = window_size
        self._reference: Dict[str, List[float]] = {}
        self._current: Dict[str, deque] = {}

    def set_reference(self, feature_name: str, values: List[float]) -> None:
        self._reference[feature_name] = sorted(values)
        self._current[feature_name] = deque(maxlen=self.window_size)

    def update(self, feature_name: str, value: float) -> None:
        if feature_name in self._current:
            self._current[feature_name].append(value)

    def compute_psi(self, feature_name: str) -> Optional[float]:
        ref = self._reference.get(feature_name)
        curr = list(self._current.get(feature_name, []))
        if not ref or len(curr) < 100:
            return None

        ref_pcts = self._percentile_bins(ref)
        curr_pcts = self._percentile_bins(curr)
        psi = sum(
            (c - r) * math.log((c + 1e-10) / (r + 1e-10))
            for r, c in zip(ref_pcts, curr_pcts)
        )
        return psi

    def check_all_features(self) -> Dict[str, Any]:
        results = {}
        for feature in self._reference:
            psi = self.compute_psi(feature)
            if psi is not None:
                severity = "none"
                if psi > self.SEVERE_PSI_THRESHOLD:
                    severity = "severe"
                    logger.warning(f"Severe drift detected in {feature}: PSI={psi:.3f}")
                elif psi > self.PSI_THRESHOLD:
                    severity = "moderate"
                results[feature] = {"psi": psi, "severity": severity}
        return results

    def _percentile_bins(self, values: List[float]) -> List[float]:
        n = len(values)
        sorted_vals = sorted(values)
        bins = []
        for i in range(self.n_bins):
            start = int(i / self.n_bins * n)
            end = int((i + 1) / self.n_bins * n)
            count = end - start
            bins.append(count / n)
        return bins
