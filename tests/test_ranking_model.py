"""Tests for the ranking model and feature builder."""

import pytest
import numpy as np
from ranking.ranking_model import RankingModel
from ranking.ranking_feature_builder import RankingFeatureBuilder


class TestRankingModel:
    def setup_method(self):
        config = {"n_estimators": 10}
        self.model = RankingModel(config)

    def test_predict_shape(self):
        X = np.random.randn(10, len(RankingModel.FEATURE_NAMES))
        scores = self.model.predict(X)
        assert scores.shape == (10,)

    def test_scores_in_range(self):
        X = np.random.randn(20, len(RankingModel.FEATURE_NAMES))
        scores = self.model.predict(X)
        assert all(0 <= s <= 1 for s in scores)

    def test_feature_matrix_shape(self):
        candidates = [
            {"video_id": f"v{i}", "retrieval_score": 0.9 - i * 0.1,
             "video_features": {"video_like_rate": 0.1, "video_category": "gaming"}}
            for i in range(5)
        ]
        user_features = {"user_avg_watch_sec": 120, "user_like_rate": 0.05,
                         "user_cat_gaming_affinity": 0.3}
        session_features = {"session_length_min": 10, "session_skip_rate": 0.1}
        X, ids = self.model.build_feature_matrix(candidates, user_features, session_features)
        assert X.shape[0] == 5
        assert len(ids) == 5


class TestRankingFeatureBuilder:
    def test_build_returns_dict(self):
        builder = RankingFeatureBuilder()
        features = builder.build(
            {"user_age_days_log": 2.5, "user_like_rate": 0.08,
             "user_cat_gaming_affinity": 0.4},
            {"video_category": "gaming", "video_like_rate": 0.1},
            {"session_length_min": 5},
        )
        assert isinstance(features, dict)
        assert "user_video_category_match" in features
        assert features["user_video_category_match"] == pytest.approx(0.4)
