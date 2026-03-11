"FastAPI application entry point."

import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.auth_middleware import AuthMiddleware
from api.middleware.logging_middleware import LoggingMiddleware
from api.routes import recommend, health, events, experiments

from monitoring.health_monitor import HealthMonitor
from monitoring.metrics_collector import MetricsCollector

from services.feature_service import FeatureService
from candidate_generation.candidate_service import CandidateService
from candidate_generation.two_tower_retrieval_model import TwoTowerRetrievalModel
from services.preranking_service import PreRankingService
from services.ranking_service import RankingService
from reranking.reranking_service import RerankingService
from services.recommendation_engine import RecommendationEngine
from storage.feature_cache_store import FeatureCacheStore

from embeddings.user_embedding_model import UserEmbeddingModel
from embeddings.video_embedding_model import VideoEmbeddingModel
from embeddings.embedding_service import EmbeddingService
from live_session_engine.session_state_tracker import SessionStateTracker
from live_session_engine.session_interest_model import SessionInterestModel
from live_session_engine.live_recommendation_engine import LiveRecommendationEngine
from exploration.discovery_service import DiscoveryService
from ranking.ranking_inference import RankingInference
from ranking.ranking_model import RankingModel

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 128
NUM_VIDEOS = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all platform components on startup."""
    logger.info("Initializing recommendation platform...")

    app.state.metrics_collector = MetricsCollector()
    app.state.health_monitor = HealthMonitor()

    # --- Feature store ---
    feature_store = FeatureCacheStore()
    feature_service = FeatureService(feature_store)

    # --- Build Two-Tower FAISS index ---
    logger.info(f"Building Two-Tower FAISS index for {NUM_VIDEOS} videos...")
    two_tower = TwoTowerRetrievalModel(embedding_dim=EMBEDDING_DIM, top_k=500)
    video_ids = [str(i) for i in range(NUM_VIDEOS)]
    video_embeddings = np.random.rand(NUM_VIDEOS, EMBEDDING_DIM).astype(np.float32)
    two_tower.build_index(video_ids, video_embeddings)
    logger.info(f"FAISS index built. Using FAISS: {two_tower._use_faiss}")

    # --- Candidate service ---
    candidate_service = CandidateService(
        two_tower=two_tower,
        collab_filter=None,
        matrix_fact=None,
        top_k=500,
    )

    # --- Core recommendation engine ---
    preranking_service = PreRankingService()
    ranking_service = RankingService()
    reranking_service = RerankingService()

    app.state.recommendation_engine = RecommendationEngine(
        feature_service=feature_service,
        candidate_service=candidate_service,
        preranking_service=preranking_service,
        ranking_service=ranking_service,
        reranking_service=reranking_service,
    )

    # --- Embedding service ---
    user_model = UserEmbeddingModel(embedding_dim=EMBEDDING_DIM)
    video_model = VideoEmbeddingModel(embedding_dim=EMBEDDING_DIM)
    embedding_service = EmbeddingService(
        user_model=user_model,
        video_model=video_model,
    )

    # --- Live session engine ---
    session_tracker = SessionStateTracker(session_ttl_minutes=30)
    session_interest_model = SessionInterestModel()
    discovery_service = DiscoveryService(exploration_fraction=0.1)
    from ranking.ranking_model import RankingModel
    ranking_model = RankingModel(config={})
    ranking_inference = RankingInference(ranking_model=ranking_model, feature_store=feature_store)

    app.state.live_engine = LiveRecommendationEngine(
        candidate_service=candidate_service,
        ranking_inference=ranking_inference,
        reranking_service=reranking_service,
        discovery_service=discovery_service,
        session_tracker=session_tracker,
        session_interest_model=session_interest_model,
        embedding_service=embedding_service,
        feature_store=feature_store,
    )

    app.state.embedding_dim = EMBEDDING_DIM
    app.state.event_router = None
    app.state.experiment_manager = None
    app.state.experiment_metrics = None

    logger.info("Platform initialized successfully.")

    yield

    logger.info("Shutting down platform...")


app = FastAPI(
    title="Enterprise Live Recommendation Platform",
    version="1.0.0",
    description="Real-time video recommendation system with live session support",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(events.router)
app.include_router(experiments.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)