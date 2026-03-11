"""Tests for reranking components."""

import pytest
from reranking.diversity_optimizer import DiversityOptimizer
from reranking.freshness_boost import FreshnessBoost
from reranking.trending_score import TrendingScore


class TestDiversityOptimizer:
    def setup_method(self):
        self.optimizer = DiversityOptimizer(max_same_creator=2, max_same_category=3)

    def _make_candidates(self, n=30):
        candidates = []
        for i in range(n):
            candidates.append({
                "video_id": f"v{i}",
                "ranking_score": 1.0 - i * 0.03,
                "creator_id": f"creator_{i % 5}",
                "category": ["gaming", "music", "sports", "news"][i % 4],
            })
        return candidates

    def test_returns_top_k(self):
        candidates = self._make_candidates(30)
        result = self.optimizer.optimize(candidates, top_k=10)
        assert len(result) == 10

    def test_creator_diversity(self):
        candidates = self._make_candidates(30)
        result = self.optimizer.optimize(candidates, top_k=20)
        from collections import Counter
        creator_counts = Counter(r["creator_id"] for r in result)
        assert max(creator_counts.values()) <= 2


class TestFreshnessBoost:
    def test_boosts_fresh_content(self):
        boost = FreshnessBoost(decay_hours=24, boost_factor=0.2)
        fresh = [{"video_id": "v1", "ranking_score": 0.5, "video_age_hours": 1}]
        old = [{"video_id": "v2", "ranking_score": 0.5, "video_age_hours": 200}]
        fresh_result = boost.apply(fresh)
        old_result = boost.apply(old)
        assert fresh_result[0]["ranking_score"] > old_result[0]["ranking_score"]


class TestTrendingScore:
    def test_higher_engagement_higher_score(self):
        ts = TrendingScore()
        import time
        now = time.time()
        high = ts.compute({"views": 10000, "likes": 1000, "shares": 200,
                           "watch_sec": 500000, "last_viewed": now - 60})
        low = ts.compute({"views": 10, "likes": 1, "shares": 0,
                          "watch_sec": 500, "last_viewed": now - 60})
        assert high > low
