"""Training pipeline for the ranking model."""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class RankingTrainer:
    """Trains a LightGBM ranking model on historical engagement data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare_dataset(self, interactions: List[Dict[str, Any]],
                        feature_builder) -> Tuple[np.ndarray, np.ndarray]:
        """Build training features and labels from interaction logs."""
        X, y = [], []
        for record in interactions:
            uf = record.get("user_features", {})
            vf = record.get("video_features", {})
            sf = record.get("session_features", {})
            rs = record.get("retrieval_score", 0.0)
            features = feature_builder.build(uf, vf, sf, rs)
            label = self._compute_label(record)
            X.append(list(features.values()))
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            params = {
                "objective": "binary",
                "metric": ["auc", "binary_logloss"],
                "n_estimators": self.config.get("n_estimators", 500),
                "max_depth": self.config.get("max_depth", 6),
                "learning_rate": self.config.get("learning_rate", 0.05),
                "num_leaves": self.config.get("num_leaves", 63),
                "subsample": self.config.get("subsample", 0.8),
                "verbose": -1,
            }
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
            model = lgb.train(params, dtrain,
                              valid_sets=[dval],
                              callbacks=callbacks)
            return {"model": model, "best_iteration": model.best_iteration}
        except ImportError:
            logger.warning("LightGBM not installed — skipping actual training")
            return {}

    @staticmethod
    def _compute_label(record: Dict[str, Any]) -> float:
        watch_rate = record.get("completion_rate", 0)
        liked = float(record.get("liked", False))
        shared = float(record.get("shared", False))
        return min(1.0, watch_rate * 0.5 + liked * 0.3 + shared * 0.2)
