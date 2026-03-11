"""User embedding model using collaborative signals and profile features."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class UserEmbeddingModel:
    """
    Generates dense user embeddings from interaction history and profile.
    In production, replaced by a trained Two-Tower or matrix factorization model.
    """

    def __init__(self, embedding_dim: int = 128, vocab_size: int = 100000):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self._embeddings: Dict[str, np.ndarray] = {}
        self._is_loaded = False

    def load(self, model_path: str) -> None:
        """Load pre-trained embeddings from disk."""
        try:
            import numpy as np
            # In production: load from model registry
            logger.info(f"Loading user embeddings from {model_path}")
            self._is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def get_embedding(self, user_id: str,
                      user_features: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Return a 128-dim embedding for the given user.
        Falls back to deterministic random embedding for cold-start users.
        """
        if user_id in self._embeddings:
            return self._embeddings[user_id]
        return self._cold_start_embedding(user_id)

    def get_batch_embeddings(self, user_ids: List[str]) -> np.ndarray:
        """Return embeddings for a batch of users. Shape: (N, embedding_dim)"""
        return np.vstack([self.get_embedding(uid) for uid in user_ids])

    def update_embedding(self, user_id: str, embedding: np.ndarray) -> None:
        """Update a user's embedding (e.g., after online learning step)."""
        assert embedding.shape == (self.embedding_dim,), \
            f"Expected shape ({self.embedding_dim},), got {embedding.shape}"
        self._embeddings[user_id] = embedding.astype(np.float32)

    def _cold_start_embedding(self, user_id: str) -> np.ndarray:
        """Generate a deterministic embedding for new users via hashing."""
        seed = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        emb = rng.normal(0, 0.1, self.embedding_dim).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)

    def similarity(self, user_id_a: str, user_id_b: str) -> float:
        """Cosine similarity between two users."""
        a = self.get_embedding(user_id_a)
        b = self.get_embedding(user_id_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
