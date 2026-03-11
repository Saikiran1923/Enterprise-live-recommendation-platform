"""Models evolving user interest within a live session."""

import numpy as np
from typing import List, Dict, Any, Optional


class SessionInterestModel:
    """
    Maintains a dynamic interest embedding for the current session
    by aggregating video embeddings of watched content with recency weighting.
    """

    def __init__(self, embedding_dim: int = 128, decay: float = 0.9):
        self.embedding_dim = embedding_dim
        self.decay = decay
        self._session_embeddings: Dict[str, np.ndarray] = {}

    def update(self, session_id: str, video_embedding: np.ndarray,
               watch_rate: float = 1.0) -> np.ndarray:
        """
        Update the session interest embedding with a newly watched video.
        Uses exponential moving average with watch-rate weighting.
        """
        weight = watch_rate
        current = self._session_embeddings.get(session_id)

        if current is None:
            new_emb = video_embedding * weight
        else:
            new_emb = self.decay * current + (1 - self.decay) * video_embedding * weight

        norm = np.linalg.norm(new_emb)
        new_emb = new_emb / (norm + 1e-8)
        self._session_embeddings[session_id] = new_emb
        return new_emb

    def get(self, session_id: str) -> Optional[np.ndarray]:
        return self._session_embeddings.get(session_id)

    def merge_with_user(self, session_id: str,
                        user_embedding: np.ndarray,
                        session_weight: float = 0.4) -> np.ndarray:
        """
        Merge long-term user embedding with short-term session interest.
        """
        session_emb = self._session_embeddings.get(session_id)
        if session_emb is None:
            return user_embedding
        merged = (1 - session_weight) * user_embedding + session_weight * session_emb
        return merged / (np.linalg.norm(merged) + 1e-8)

    def clear(self, session_id: str) -> None:
        self._session_embeddings.pop(session_id, None)
