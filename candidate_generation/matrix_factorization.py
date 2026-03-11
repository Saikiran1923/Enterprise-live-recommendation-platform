"""Alternating Least Squares matrix factorization for recommendations."""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class MatrixFactorization:
    """
    Implicit feedback matrix factorization using ALS.
    Produces user and item factor matrices for recommendation.
    """

    def __init__(self, num_factors: int = 64, regularization: float = 0.01,
                 iterations: int = 20, confidence_weight: float = 40.0):
        self.num_factors = num_factors
        self.regularization = regularization
        self.iterations = iterations
        self.confidence_weight = confidence_weight
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._user_index: Dict[str, int] = {}
        self._item_index: Dict[str, int] = {}
        self._item_ids: List[str] = []
        self._is_fitted = False

    def fit(self, interactions: List[Dict]) -> None:
        """Fit ALS model on user-item interaction data."""
        users = list({i["user_id"] for i in interactions if "user_id" in i})
        items = list({i["video_id"] for i in interactions if "video_id" in i})
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {v: i for i, v in enumerate(items)}
        self._item_ids = items

        n_users, n_items = len(users), len(items)
        confidence_matrix = np.zeros((n_users, n_items), dtype=np.float32)

        for event in interactions:
            u = self._user_index.get(event.get("user_id"))
            v = self._item_index.get(event.get("video_id"))
            if u is not None and v is not None:
                watch_rate = event.get("completion_rate", 0.5)
                confidence_matrix[u, v] += self.confidence_weight * watch_rate

        rng = np.random.RandomState(42)
        self._user_factors = rng.normal(0, 0.1, (n_users, self.num_factors)).astype(np.float32)
        self._item_factors = rng.normal(0, 0.1, (n_items, self.num_factors)).astype(np.float32)

        for it in range(self.iterations):
            self._als_step(confidence_matrix, update_users=True)
            self._als_step(confidence_matrix, update_users=False)
            if it % 5 == 0:
                logger.info(f"ALS iteration {it}/{self.iterations}")

        self._is_fitted = True

    def _als_step(self, C: np.ndarray, update_users: bool) -> None:
        """Single ALS update step."""
        if update_users:
            factors_fixed = self._item_factors
            factors_update = self._user_factors
            C_iter = C
        else:
            factors_fixed = self._user_factors
            factors_update = self._item_factors
            C_iter = C.T

        YtY = factors_fixed.T @ factors_fixed
        reg = self.regularization * np.eye(self.num_factors)

        for u in range(len(factors_update)):
            Cu = np.diag(C_iter[u])
            A = YtY + factors_fixed.T @ Cu @ factors_fixed + reg
            b = factors_fixed.T @ (Cu + np.eye(len(C_iter[u]))) @ C_iter[u]
            try:
                factors_update[u] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                pass

    def recommend(self, user_id: str, top_k: int = 200) -> List[Tuple[str, float]]:
        """Get top-k video recommendations for a user."""
        if not self._is_fitted:
            return []
        u_idx = self._user_index.get(user_id)
        if u_idx is None:
            return []
        u_vec = self._user_factors[u_idx]
        scores = self._item_factors @ u_vec
        top_indices = np.argsort(-scores)[:top_k]
        return [(self._item_ids[i], float(scores[i])) for i in top_indices]
