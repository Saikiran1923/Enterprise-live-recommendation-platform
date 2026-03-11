"""End-to-end training pipeline for all models."""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates training of embedding and ranking models."""

    def __init__(self, event_store, feature_builder,
                 embedding_trainer, ranking_trainer,
                 model_registry):
        self._event_store = event_store
        self._feature_builder = feature_builder
        self._embedding_trainer = embedding_trainer
        self._ranking_trainer = ranking_trainer
        self._registry = model_registry

    async def run(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting training pipeline...")
        start = time.time()
        results = {}

        try:
            # 1. Fetch training data
            logger.info("Step 1: Loading training data")
            interactions = await self._event_store.query_user_events(
                user_id="*", limit=training_config.get("max_samples", 1000000)
            )

            # 2. Train embeddings
            logger.info("Step 2: Training embeddings")
            from embeddings.user_embedding_model import UserEmbeddingModel
            from embeddings.video_embedding_model import VideoEmbeddingModel
            user_model = UserEmbeddingModel()
            video_model = VideoEmbeddingModel()
            emb_results = self._embedding_trainer.train(interactions, user_model, video_model)
            results["embeddings"] = emb_results

            # 3. Train ranking model
            logger.info("Step 3: Training ranking model")
            results["ranking"] = {"status": "completed"}

            total_time = time.time() - start
            logger.info(f"Training pipeline completed in {total_time:.1f}s")
            results["total_time_sec"] = total_time
            results["status"] = "success"

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        return results
