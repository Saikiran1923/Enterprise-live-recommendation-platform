"""Integration tests for the recommendation API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.recommend = AsyncMock(return_value={
        "user_id": "user_001",
        "session_id": "sess_001",
        "recommendations": [
            {"video_id": f"video_{i}", "rank": i + 1,
             "ranking_score": 0.9 - i * 0.05, "is_exploration": False}
            for i in range(5)
        ],
        "metadata": {"latency_ms": 45, "candidate_count": 200},
    })
    return engine


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_ready_returns_200(self, client):
        response = client.get("/ready")
        assert response.status_code == 200


class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client, mock_engine):
        client.app.state.recommendation_engine = mock_engine
        response = client.post(
            "/recommend/",
            json={"user_id": "user_001", "top_k": 5},
            headers={"X-API-Key": "dev-key-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert data["user_id"] == "user_001"

    def test_recommend_without_api_key_returns_401(self, client, mock_engine):
        client.app.state.recommendation_engine = mock_engine
        response = client.post("/recommend/", json={"user_id": "user_001"})
        assert response.status_code == 401
