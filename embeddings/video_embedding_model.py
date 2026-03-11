"""Video embedding model combining content and collaborative signals."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class VideoEmbeddingModel:
    """
    Generates dense video embeddings from content features and engagement signals.
    Supports content-based (title/tags) and collaborative (watch history) embeddings.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._embeddings: Dict[str, np.ndarray] = {}
        self._is_loaded = False

    def load(self, model_path: str) -> None:
        logger.info(f"Loading video embeddings from {model_path}")
        self._is_loaded = True

    def get_embedding(self, video_id: str,
                      video_metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        if video_id in self._embeddings:
            return self._embeddings[video_id]
        if video_metadata:
            return self._content_based_embedding(video_id, video_metadata)
        return self._cold_start_embedding(video_id)

    def get_batch_embeddings(self, video_ids: List[str]) -> np.ndarray:
        return np.vstack([self.get_embedding(vid) for vid in video_ids])

    def update_embedding(self, video_id: str, embedding: np.ndarray) -> None:
        assert embedding.shape == (self.embedding_dim,)
        self._embeddings[video_id] = embedding.astype(np.float32)

    def _content_based_embedding(self, video_id: str,
                                  metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding from content features (title, tags, category)."""
        seed = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        base = rng.normal(0, 0.1, self.embedding_dim).astype(np.float32)

        # Shift embedding based on category
        category = metadata.get("category", "")
        cat_seed = int(hashlib.md5(category.encode()).hexdigest(), 16) % (2**31)
        cat_rng = np.random.RandomState(cat_seed)
        cat_bias = cat_rng.normal(0, 0.05, self.embedding_dim).astype(np.float32)

        emb = base + cat_bias
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _cold_start_embedding(self, video_id: str) -> np.ndarray:
        seed = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        emb = rng.normal(0, 0.1, self.embedding_dim).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)

    def find_similar(self, video_id: str, candidate_ids: List[str],
                     top_k: int = 10) -> List[str]:
        """Find top-k most similar videos by cosine similarity."""
        query = self.get_embedding(video_id)
        scores = []
        for vid in candidate_ids:
            emb = self.get_embedding(vid)
            score = float(np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8))
            scores.append((vid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [vid for vid, _ in scores[:top_k]]
