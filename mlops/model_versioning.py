"""Model versioning utilities for reproducible experiments."""

import hashlib
import time
import json
from typing import Dict, Any


class ModelVersioning:
    """Utilities for generating and comparing model version identifiers."""

    @staticmethod
    def generate_version(config: Dict[str, Any],
                         data_version: str,
                         training_timestamp: Optional[float] = None) -> str:
        """Generate a deterministic version string from training config + data."""
        config_str = json.dumps(config, sort_keys=True)
        content = f"{config_str}:{data_version}"
        short_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        ts = int(training_timestamp or time.time())
        return f"v{ts}_{short_hash}"

    @staticmethod
    def parse_version(version: str) -> Dict[str, Any]:
        parts = version.lstrip("v").split("_")
        return {
            "timestamp": int(parts[0]) if parts else 0,
            "hash": parts[1] if len(parts) > 1 else "",
        }

    @staticmethod
    def is_newer(version_a: str, version_b: str) -> bool:
        a = ModelVersioning.parse_version(version_a)
        b = ModelVersioning.parse_version(version_b)
        return a["timestamp"] > b["timestamp"]


from typing import Optional
