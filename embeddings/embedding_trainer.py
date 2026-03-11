"""Training loop for Two-Tower embedding model."""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """
    Trains user and video embeddings using contrastive learning
    on positive/negative interaction pairs.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_dim = config.get("user_embedding_dim", 128)
        self.lr = config.get("learning_rate", 0.001)
        self.temperature = config.get("temperature", 0.07)
        self.batch_size = config.get("batch_size", 1024)
        self.epochs = config.get("epochs", 20)
        self._train_losses: List[float] = []

    def prepare_training_data(self, interactions: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """
        Convert raw interactions into (user_id, positive_video_id, negative_video_id) triples.
        Uses in-batch negatives strategy.
        """
        positives = [(i["user_id"], i["video_id"]) for i in interactions
                     if i.get("event_type") in ("video_view", "video_like")]
        users = [u for u, _ in positives]
        pos_videos = [v for _, v in positives]
        # In-batch negatives: shift positive videos by 1
        neg_videos = pos_videos[1:] + [pos_videos[0]]
        return users, pos_videos, neg_videos

    def train(self, interactions: List[Dict[str, Any]],
              user_model, video_model) -> Dict[str, Any]:
        """Run training loop and return training metrics."""
        logger.info(f"Starting embedding training with {len(interactions)} interactions")
        users, pos_vids, neg_vids = self.prepare_training_data(interactions)
        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.epochs):
            epoch_loss = self._run_epoch(users, pos_vids, neg_vids, user_model, video_model)
            self._train_losses.append(epoch_loss)
            total_loss += epoch_loss
            n_batches += 1
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs} loss={epoch_loss:.4f}")

        return {
            "epochs": self.epochs,
            "final_loss": self._train_losses[-1] if self._train_losses else 0.0,
            "avg_loss": total_loss / max(n_batches, 1),
        }

    def _run_epoch(self, users, pos_vids, neg_vids, user_model, video_model) -> float:
        """Single epoch of contrastive learning (simplified numpy implementation)."""
        total_loss = 0.0
        n = len(users)
        indices = np.random.permutation(n)

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            u_embs = np.vstack([user_model.get_embedding(users[i]) for i in batch_idx])
            p_embs = np.vstack([video_model.get_embedding(pos_vids[i]) for i in batch_idx])
            n_embs = np.vstack([video_model.get_embedding(neg_vids[i]) for i in batch_idx])

            # InfoNCE loss approximation
            pos_scores = np.sum(u_embs * p_embs, axis=1) / self.temperature
            neg_scores = np.sum(u_embs * n_embs, axis=1) / self.temperature
            loss = -np.mean(pos_scores - np.logaddexp(pos_scores, neg_scores))
            total_loss += float(loss)

        return total_loss / max(n // self.batch_size, 1)

    def get_training_history(self) -> List[float]:
        return self._train_losses.copy()
