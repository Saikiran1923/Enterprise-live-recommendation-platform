"""End-to-end tests for the recommendation pipeline."""

import pytest
import asyncio
import numpy as np


@pytest.mark.asyncio
async def test_full_pipeline_smoke():
    """Smoke test: ensure all pipeline components initialize without error."""
    from feature_store.online_feature_store import OnlineFeatureStore
    from embeddings.user_embedding_model import UserEmbeddingModel
    from embeddings.video_embedding_model import VideoEmbeddingModel
    from embeddings.embedding_service import EmbeddingService
    from candidate_generation.two_tower_retrieval_model import TwoTowerRetrievalModel
    from candidate_generation.candidate_service import CandidateService
    from ranking.ranking_model import RankingModel
    from ranking.ranking_inference import RankingInference
    from reranking.reranking_service import RerankingService

    feature_store = OnlineFeatureStore()
    user_model = UserEmbeddingModel(embedding_dim=16)
    video_model = VideoEmbeddingModel(embedding_dim=16)
    emb_service = EmbeddingService(user_model, video_model)

    two_tower = TwoTowerRetrievalModel(embedding_dim=16, top_k=20)
    video_ids = [f"video_{i}" for i in range(50)]
    embeddings = np.random.randn(50, 16).astype(np.float32)
    two_tower.build_index(video_ids, embeddings)

    candidate_svc = CandidateService(two_tower=two_tower, top_k=20)

    ranking_model = RankingModel({})
    ranking_inf = RankingInference(ranking_model, top_k=10)
    reranking_svc = RerankingService()

    user_emb = await emb_service.get_user_embedding("user_001")
    candidates = await candidate_svc.get_candidates("user_001", user_emb)
    ranked = await ranking_inf.rank(candidates, {}, {}, top_k=10)
    final = await reranking_svc.rerank(ranked, {}, top_k=5)

    assert isinstance(final, list)
    assert len(final) <= 5
    print(f"Pipeline smoke test passed: {len(final)} recommendations")
