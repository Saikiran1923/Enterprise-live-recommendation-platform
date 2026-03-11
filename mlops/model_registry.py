"""Central model registry for versioning and deployment tracking."""

import time
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    TRAINING = "training"
    STAGED = "staged"
    PRODUCTION = "production"
    SHADOW = "shadow"
    RETIRED = "retired"


class ModelRegistry:
    """
    Manages model versions, metadata, and deployment status.
    Integrates with the deployment manager for safe rollouts.
    """

    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._production_models: Dict[str, str] = {}

    def register(self, model_name: str, version: str,
                 metadata: Dict[str, Any]) -> str:
        key = f"{model_name}:{version}"
        self._models[key] = {
            "name": model_name,
            "version": version,
            "status": ModelStatus.STAGED,
            "registered_at": time.time(),
            "metrics": metadata.get("metrics", {}),
            "artifact_path": metadata.get("artifact_path", ""),
            "training_data_version": metadata.get("training_data_version", ""),
            "feature_schema_version": metadata.get("feature_schema_version", ""),
        }
        logger.info(f"Registered model: {key}")
        return key

    def promote_to_production(self, model_name: str, version: str) -> None:
        key = f"{model_name}:{version}"
        if key not in self._models:
            raise ValueError(f"Model {key} not found in registry")
        old_prod = self._production_models.get(model_name)
        if old_prod:
            self._models[f"{model_name}:{old_prod}"]["status"] = ModelStatus.RETIRED
        self._models[key]["status"] = ModelStatus.PRODUCTION
        self._models[key]["promoted_at"] = time.time()
        self._production_models[model_name] = version
        logger.info(f"Promoted {key} to production")

    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        version = self._production_models.get(model_name)
        if not version:
            return None
        return self._models.get(f"{model_name}:{version}")

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        return [v for k, v in self._models.items() if k.startswith(f"{model_name}:")]

    def get_model(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        return self._models.get(f"{model_name}:{version}")
