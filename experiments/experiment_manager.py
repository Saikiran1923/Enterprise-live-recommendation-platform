"""Manages experiment lifecycle and configuration."""

import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Stores and retrieves experiment configurations."""

    def __init__(self, config: Dict[str, Any] = None):
        self._experiments: Dict[str, Dict[str, Any]] = {}
        if config:
            for exp in config.get("active_experiments", []):
                self.register(exp)

    def register(self, experiment: Dict[str, Any]) -> None:
        exp_id = experiment["id"]
        experiment.setdefault("active", True)
        experiment.setdefault("traffic_allocation", 0.5)
        experiment.setdefault("created_at", time.time())
        self._experiments[exp_id] = experiment
        logger.info(f"Registered experiment: {exp_id}")

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._experiments.get(experiment_id)

    def list_active(self) -> List[str]:
        return [eid for eid, e in self._experiments.items() if e.get("active", False)]

    def deactivate(self, experiment_id: str) -> None:
        if experiment_id in self._experiments:
            self._experiments[experiment_id]["active"] = False
            logger.info(f"Deactivated experiment: {experiment_id}")

    def update_traffic(self, experiment_id: str, allocation: float) -> None:
        if experiment_id in self._experiments:
            self._experiments[experiment_id]["traffic_allocation"] = allocation
