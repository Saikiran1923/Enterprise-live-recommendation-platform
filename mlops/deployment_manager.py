"""Manages canary and blue/green model deployments."""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DeploymentManager:
    """
    Orchestrates safe model deployments via canary rollouts.
    Supports gradual traffic shifting and automatic rollback on degradation.
    """

    def __init__(self, model_registry=None, rollback_controller=None):
        self._registry = model_registry
        self._rollback = rollback_controller
        self._deployments: Dict[str, Dict[str, Any]] = {}

    def deploy(self, model_name: str, version: str,
               strategy: str = "canary",
               canary_fraction: float = 0.05) -> str:
        deployment_id = f"{model_name}:{version}:{int(time.time())}"
        self._deployments[deployment_id] = {
            "model_name": model_name,
            "version": version,
            "strategy": strategy,
            "canary_fraction": canary_fraction,
            "status": "deploying",
            "started_at": time.time(),
            "traffic_fraction": canary_fraction if strategy == "canary" else 1.0,
        }
        logger.info(f"Started {strategy} deployment: {deployment_id}")
        return deployment_id

    def increase_traffic(self, deployment_id: str, new_fraction: float) -> None:
        """Gradually increase traffic to a canary deployment."""
        dep = self._deployments.get(deployment_id)
        if not dep:
            raise ValueError(f"Deployment {deployment_id} not found")
        dep["traffic_fraction"] = min(1.0, new_fraction)
        if new_fraction >= 1.0:
            dep["status"] = "fully_deployed"
            model_name = dep["model_name"]
            version = dep["version"]
            if self._registry:
                self._registry.promote_to_production(model_name, version)
        logger.info(f"Deployment {deployment_id} traffic: {new_fraction:.0%}")

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        return self._deployments.get(deployment_id)

    def list_active_deployments(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self._deployments.items()
                if v["status"] not in ("rolled_back", "retired")}
