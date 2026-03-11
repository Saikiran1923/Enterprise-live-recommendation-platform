"""A/B test traffic router for experiment assignment."""

import hashlib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ABRouter:
    """
    Deterministically assigns users to experiment variants
    using consistent hashing. Ensures stable assignments across requests.
    """

    def __init__(self, experiment_manager=None):
        self._experiment_manager = experiment_manager

    def assign(self, user_id: str, experiment_id: str) -> Optional[str]:
        """
        Assign a user to an experiment variant.
        Returns variant name or None if user is not in experiment.
        """
        experiment = self._experiment_manager.get_experiment(experiment_id) if \
            self._experiment_manager else None
        if not experiment or not experiment.get("active", False):
            return None

        traffic_allocation = experiment.get("traffic_allocation", 1.0)
        hash_val = self._hash(user_id, experiment_id)

        if hash_val >= traffic_allocation:
            return None  # Not in experiment

        variants = experiment.get("variants", [])
        if not variants:
            return None

        # Assign to variant based on second hash bucket
        variant_hash = self._hash(user_id, f"{experiment_id}_variant")
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.get("weight", 1.0 / len(variants))
            if variant_hash < cumulative:
                return variant["name"]

        return variants[-1]["name"]

    def assign_all(self, user_id: str) -> Dict[str, str]:
        """Get variant assignments for all active experiments."""
        if not self._experiment_manager:
            return {}
        assignments = {}
        for exp_id in self._experiment_manager.list_active():
            variant = self.assign(user_id, exp_id)
            if variant:
                assignments[exp_id] = variant
        return assignments

    @staticmethod
    def _hash(user_id: str, salt: str) -> float:
        """Return a stable float in [0, 1) for user + salt."""
        key = f"{user_id}:{salt}".encode()
        digest = hashlib.md5(key).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF
