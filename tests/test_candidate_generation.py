"""Tests for candidate generation components."""

import pytest
import numpy as np
from candidate_generation.two_tower_retrieval_model import TwoTowerRetrievalModel
from candidate_generation.collaborative_filtering import CollaborativeFiltering


class TestTwoTowerRetrieval:
    def setup_method(self):
        self.model = TwoTowerRetrievalModel(embedding_dim=16, top_k=10)
        n_videos = 100
        video_ids = [f"video_{i}" for i in range(n_videos)]
        embeddings = np.random.randn(n_videos, 16).astype(np.float32)
        self.model.build_index(video_ids, embeddings)

    def test_retrieves_correct_count(self):
        query = np.random.randn(16).astype(np.float32)
        results = self.model.retrieve(query, top_k=10)
        assert len(results) == 10

    def test_scores_are_normalized(self):
        query = np.random.randn(16).astype(np.float32)
        results = self.model.retrieve(query)
        for _, score in results:
            assert -1.01 <= score <= 1.01

    def test_excludes_ids(self):
        query = np.random.randn(16).astype(np.float32)
        exclude = {"video_0", "video_1", "video_2"}
        results = self.model.retrieve(query, exclude_ids=list(exclude))
        returned_ids = {vid for vid, _ in results}
        assert len(returned_ids & exclude) == 0

    def test_sorted_descending(self):
        query = np.random.randn(16).astype(np.float32)
        results = self.model.retrieve(query)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestCollaborativeFiltering:
    def setup_method(self):
        self.cf = CollaborativeFiltering(num_candidates=20)
        interactions = [
            {"user_id": f"user_{i % 10}", "video_id": f"video_{j}"}
            for i in range(100) for j in range(i % 5, i % 5 + 5)
        ]
        self.cf.fit(interactions)

    def test_recommend_known_user(self):
        results = self.cf.recommend("user_0")
        assert isinstance(results, list)

    def test_recommend_cold_start_user(self):
        results = self.cf.recommend("unknown_user_999")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_respects_num_candidates(self):
        results = self.cf.recommend("user_0")
        assert len(results) <= 20
