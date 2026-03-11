"""Gradient boosting ranking model (LightGBM-based)."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RankingModel:
    """
    Pointwise ranking model using LightGBM (or XGBoost).
    Predicts engagement score for (user, video) pairs.
    """

    FEATURE_NAMES = [
        "user_age_days_log", "user_total_views_log", "user_avg_watch_sec",
        "user_like_rate", "user_skip_rate", "user_engagement_score",
        "video_view_count_log", "video_like_rate", "video_avg_watch_sec",
        "video_freshness_score", "video_trending_score", "video_duration_log",
        "retrieval_score", "user_video_category_match",
        "session_length_min", "session_skip_rate",
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._model = None
        self._feature_importance: Dict[str, float] = {}
        self._is_loaded = False
        self._version = "v0"

    def load(self, model_path: str) -> None:
        """Load trained model from disk."""
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=model_path)
            self._is_loaded = True
            self._version = model_path.split("/")[-1]
            logger.info(f"Loaded ranking model from {model_path}")
        except ImportError:
            logger.warning("LightGBM not available, using dummy scoring")
            self._is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Predict engagement scores for a batch of (user, video) feature vectors."""
        if self._is_loaded and self._model is not None:
            return self._model.predict(feature_matrix)
        return self._fallback_score(feature_matrix)

    def predict_single(self, features: Dict[str, Any]) -> float:
        """Score a single (user, video) pair."""
        vec = self._dict_to_vector(features)
        scores = self.predict(vec.reshape(1, -1))
        return float(scores[0])

    def _fallback_score(self, X: np.ndarray) -> np.ndarray:
        """Simple linear combination fallback when model not loaded."""
        weights = np.array([0.05, 0.1, 0.05, 0.1, -0.05, 0.1,
                            0.1, 0.1, 0.05, 0.1, 0.05, -0.02,
                            0.15, 0.1, -0.02, -0.05])
        if X.shape[1] != len(weights):
            weights = np.ones(X.shape[1]) / X.shape[1]
        scores = X @ weights
        return 1 / (1 + np.exp(-scores))  # sigmoid

    def _dict_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        return np.array([
            float(features.get(name, 0)) for name in self.FEATURE_NAMES
        ], dtype=np.float32)

    def build_feature_matrix(self, candidates: List[Dict[str, Any]],
                              user_features: Dict[str, Any],
                              session_features: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Build feature matrix for a list of candidates."""
        rows = []
        video_ids = []
        for c in candidates:
            vf = c.get("video_features", {})
            combined = {**user_features, **vf, **session_features,
                        "retrieval_score": c.get("retrieval_score", 0.0),
                        "user_video_category_match": self._category_match(
                            user_features, vf)}
            rows.append(self._dict_to_vector(combined))
            video_ids.append(c["video_id"])
        return np.vstack(rows), video_ids

    @staticmethod
    def _category_match(user_features: Dict, video_features: Dict) -> float:
        video_cat = video_features.get("video_category", "")
        affinity_key = f"user_cat_{video_cat}_affinity"
        return float(user_features.get(affinity_key, 0.0))
