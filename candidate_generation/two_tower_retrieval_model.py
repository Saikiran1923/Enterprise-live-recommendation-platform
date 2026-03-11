"""Two-Tower retrieval model for fast approximate nearest neighbor search."""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TwoTowerRetrievalModel:
    """
    Two-Tower model that encodes users and videos into a shared embedding space.
    Uses dot-product similarity for fast ANN retrieval.
    Supports both exact search and approximate search via FAISS.
    """

    def __init__(self, embedding_dim: int = 128, top_k: int = 500):
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self._video_index: Optional[Any] = None
        self._video_ids: List[str] = []
        self._video_embeddings: Optional[np.ndarray] = None
        self._use_faiss = False

    def build_index(self, video_ids: List[str], video_embeddings: np.ndarray) -> None:
        """Build the video embedding index for fast retrieval."""
        assert video_embeddings.shape == (len(video_ids), self.embedding_dim)
        self._video_ids = video_ids
        self._video_embeddings = video_embeddings.astype(np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self._video_embeddings, axis=1, keepdims=True)
        self._video_embeddings = self._video_embeddings / (norms + 1e-8)

        try:
            import faiss
            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(self._video_embeddings)
            self._video_index = index
            self._use_faiss = True
            logger.info(f"Built FAISS index for {len(video_ids)} videos")
        except ImportError:
            logger.warning("FAISS not available, using numpy for ANN search")
            self._use_faiss = False

    def retrieve(self, user_embedding: np.ndarray,
                 exclude_ids: Optional[List[str]] = None,
                 top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Retrieve top-k candidate videos for a user embedding.
        Returns list of (video_id, score) tuples sorted by relevance.
        """
        k = top_k or self.top_k
        query = user_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        query = query / (norm + 1e-8)

        if self._use_faiss and self._video_index is not None:
            scores, indices = self._video_index.search(query, k * 2)
            results = [
                (self._video_ids[idx], float(scores[0][i]))
                for i, idx in enumerate(indices[0])
                if idx < len(self._video_ids)
            ]
        else:
            scores = (self._video_embeddings @ query.T).flatten()
            indices = np.argsort(-scores)[:k * 2]
            results = [(self._video_ids[i], float(scores[i])) for i in indices]

        exclude_set = set(exclude_ids or [])
        filtered = [(vid, s) for vid, s in results if vid not in exclude_set]
        return filtered[:k]

    def batch_retrieve(self, user_embeddings: np.ndarray,
                       top_k: Optional[int] = None) -> List[List[Tuple[str, float]]]:
        """Retrieve candidates for a batch of user embeddings."""
        return [self.retrieve(emb, top_k=top_k) for emb in user_embeddings]

    def get_index_stats(self) -> Dict[str, Any]:
        return {
            "num_videos": len(self._video_ids),
            "embedding_dim": self.embedding_dim,
            "use_faiss": self._use_faiss,
        }
