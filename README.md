<div align="center">

<img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.128%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/FAISS-ANN_Search-FF6B35?style=for-the-badge&logo=meta&logoColor=white"/>
<img src="https://img.shields.io/badge/Two--Tower-Retrieval-8B5CF6?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LightGBM-Ranking-00B388?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-Passing-22c55e?style=for-the-badge&logo=pytest&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-6366f1?style=for-the-badge"/>

<br/><br/>

# 🎬 Enterprise Live Recommendation Platform

### A production-grade real-time video recommendation system —
### Two-Tower FAISS retrieval · LightGBM ranking · MMR diversity · Live session engine · A/B testing

<br/>

[**Quickstart**](#-quickstart) · [**Architecture**](#-system-architecture) · [**Pipeline**](#-recommendation-pipeline) · [**API**](#-api-reference) · [**Monitoring**](#-monitoring--observability) · [**Data**](#-data) · [**Tests**](#-running-tests) · [**Roadmap**](#-roadmap)

<br/>

</div>

---

## 🧭 Overview

Enterprise Live Recommendation Platform is a **full-stack real-time recommendation system** built for production scale. It orchestrates the complete recommendation lifecycle:

```
Event Ingestion → Feature Extraction → Two-Tower FAISS Retrieval
→ PreRanking → LightGBM Ranking → MMR Diversity Reranking → Live API
```

The system handles **cold-start users** with deterministic embedding fallback, supports **live session modelling** for real-time interest tracking, and includes a complete **A/B testing framework** with UCB1 contextual bandit exploration. Every recommendation is served through a **JWT-secured FastAPI** endpoint with full observability via Prometheus and Grafana.

**Measured performance:** 4ms end-to-end latency · FAISS IndexFlatIP · 1000-video index · 500 candidates per request

Built for **Recommendation Engineers**, **ML Engineers**, and **Data Scientists** who need a real production reference — not a notebook demo.

---

## 🏛 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  Enterprise Live Recommendation Platform                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       Event Ingestion                           │    │
│  │       event_consumer · event_router · stream_processor          │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       Feature Store                             │    │
│  │         user_features · video_features · session_features       │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                           │
│         ┌────────────────────┼────────────────────┐                     │
│         ▼                    ▼                    ▼                     │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Two-Tower  │    │Collaborative │    │   Matrix     │               │
│  │FAISS (True) │    │  Filtering   │    │Factorization │               │
│  └──────┬──────┘    └──────┬───────┘    └──────┬───────┘               │
│         └────────────────┬─┘                   │                        │
│                          ▼                      │                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │           Candidate Service  (merge + deduplicate · top 500)    │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        PreRanking                               │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              LightGBM Ranking Model (fallback: linear)          │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │        Reranking: Trending · Freshness Boost · MMR Diversity    │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              FastAPI Prediction API  (JWT Auth)                 │    │
│  │         ranking_score · retrieval_score · source · rank         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Live Session Engine · UCB1 Bandit · A/B Testing · Trust Safety  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │        Observability: Prometheus · Grafana · Latency Monitor     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
enterprise-live-recommendation-platform/
│
├── api/                          # FastAPI application layer
│   ├── main.py                   # App entry point — full component wiring
│   ├── middleware/
│   │   ├── auth_middleware.py    # JWT authentication
│   │   └── logging_middleware.py # Request/response logging
│   ├── routes/
│   │   ├── recommend.py          # POST /recommend/ · /recommend/batch
│   │   ├── events.py             # Event ingestion endpoints
│   │   ├── experiments.py        # A/B experiment endpoints
│   │   └── health.py             # Health check endpoint
│   └── schemas/
│       ├── request_schema.py     # RecommendationRequest schema
│       └── response_schema.py    # RecommendationResponse schema
│
├── candidate_generation/         # Retrieval layer
│   ├── two_tower_retrieval_model.py  # FAISS Two-Tower ANN retrieval ✅
│   ├── collaborative_filtering.py   # User-based CF
│   ├── matrix_factorization.py      # MF-based retrieval
│   └── candidate_service.py         # Multi-source merge + deduplication
│
├── embeddings/                   # Embedding models
│   ├── user_embedding_model.py   # User tower (cold-start via MD5 hash)
│   ├── video_embedding_model.py  # Video tower
│   ├── embedding_trainer.py      # Training pipeline
│   ├── embedding_service.py      # Cached async serving layer
│   └── vector_index.py           # FAISS vector index wrapper
│
├── feature_store/                # Feature management
│   ├── feature_registry.py       # Feature schema definitions
│   ├── online_feature_store.py   # Low-latency online serving
│   ├── user_feature_builder.py   # User feature pipeline
│   ├── video_feature_builder.py  # Video feature pipeline
│   └── session_feature_builder.py # Real-time session features
│
├── ranking/                      # Ranking model
│   ├── ranking_model.py          # LightGBM pointwise ranking model
│   ├── ranking_trainer.py        # Training pipeline
│   ├── ranking_inference.py      # Async inference with feature enrichment
│   └── ranking_feature_builder.py # Cross-feature engineering (16 features)
│
├── reranking/                    # Post-ranking transformations
│   ├── reranking_service.py      # Full reranking pipeline orchestrator
│   ├── diversity_optimizer.py    # MMR diversity optimization
│   ├── freshness_boost.py        # Recency decay score injection
│   └── trending_score.py         # Trending signal injection
│
├── services/                     # Core service layer
│   ├── recommendation_engine.py  # Main pipeline orchestrator ✅
│   ├── recommendation_service.py # Top-level service facade + A/B logging
│   ├── feature_service.py        # Feature retrieval service
│   ├── ranking_service.py        # Ranking wrapper (ranking_score wired) ✅
│   └── preranking_service.py     # Candidate pre-filtering
│
├── live_session_engine/          # Real-time session modelling ✅
│   ├── live_recommendation_engine.py  # Session-aware recommendation engine
│   ├── session_interest_model.py      # Real-time interest embedding updates
│   └── session_state_tracker.py       # In-memory session state + TTL eviction
│
├── exploration/                  # Exploration strategies
│   ├── contextual_bandit.py      # UCB1 contextual bandit
│   ├── exploration_policy.py     # Epsilon-greedy / Thompson sampling
│   └── discovery_service.py      # New content discovery + injection
│
├── experiments/                  # A/B testing framework
│   ├── ab_router.py              # Traffic splitting and variant assignment
│   ├── experiment_manager.py     # Experiment lifecycle management
│   ├── experiment_metrics.py     # Statistical significance testing
│   └── experiment_logger.py      # Impression and click logging
│
├── trust_safety/                 # Content moderation pipeline
│   ├── safety_pipeline.py        # Full safety orchestration
│   ├── toxicity_classifier.py    # Toxicity detection
│   ├── spam_detector.py          # Spam filtering
│   └── policy_filter.py          # Policy enforcement
│
├── monitoring/                   # Observability
│   ├── metrics_collector.py      # Prometheus metrics
│   ├── health_monitor.py         # System health checks
│   ├── drift_detector.py         # Feature drift detection
│   ├── latency_monitor.py        # Latency percentile tracking
│   └── engagement_tracker.py     # CTR and engagement metrics
│
├── ingestion/                    # Event ingestion
│   ├── event_consumer.py         # Stream consumer
│   ├── event_router.py           # Event routing logic
│   ├── event_schema.py           # Event schema definitions
│   └── stream_processor.py       # Stream processing pipeline
│
├── storage/                      # Storage layer
│   ├── database.py               # Database connection
│   ├── event_store.py            # Event persistence
│   └── feature_cache_store.py    # In-memory feature cache
│
├── pipelines/                    # Batch pipelines
│   ├── training_pipeline.py      # Model training pipeline
│   ├── data_pipeline.py          # Data processing pipeline
│   ├── feature_pipeline.py       # Feature computation pipeline
│   ├── retraining_pipeline.py    # Automated retraining
│   └── batch_recommendation_pipeline.py  # Offline batch recommendations
│
├── mlops/                        # MLOps components
│   ├── model_registry.py         # Model versioning and registry
│   ├── deployment_manager.py     # Deployment orchestration
│   ├── rollback_controller.py    # Safe rollback mechanism
│   └── model_versioning.py       # Version tracking
│
├── scripts/
│   ├── load_youtube_data.py      # Load YouTube trending dataset
│   ├── generate_synthetic_data.py # 1K users · 5K videos · 50K interactions
│   ├── run_local_pipeline.py     # Local end-to-end pipeline runner
│   └── simulate_user_events.py   # Simulate live user event stream
│
├── data/
│   ├── raw/                      # YouTube trending CSVs (10 countries)
│   ├── processed/                # Feature-engineered data
│   └── feature_cache/            # Cached feature vectors
│
├── dashboards/
│   ├── grafana_dashboard.json
│   └── recommendation_metrics.json
│
├── tests/                        # Full test suite ✅
│   ├── test_api_endpoints.py
│   ├── test_candidate_generation.py
│   ├── test_ranking_model.py
│   ├── test_recommendation_pipeline.py
│   └── test_reranking.py
│
├── configs/
│   ├── system_config.yaml
│   ├── model_config.yaml
│   ├── ranking_config.yaml
│   └── experiment_config.yaml
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## ✨ Key Capabilities

### 🔍 Two-Tower FAISS Retrieval
User and video towers encode into a shared 128-dimensional embedding space. FAISS `IndexFlatIP` enables sub-millisecond approximate nearest neighbour search. Cold-start users receive deterministic embeddings via MD5 hash of `user_id` — no session history required to serve recommendations.

### 🎯 Multi-Source Candidate Generation
Candidates are retrieved in parallel from Three sources — Two-Tower ANN, Collaborative Filtering, and Matrix Factorization — then merged, deduplicated, and score-normalized into a unified pool of up to 500 candidates per request.

### 📊 LightGBM Ranking
A 16-feature pointwise ranking model scores each candidate using cross-features: user engagement history, video signals, session context, retrieval score, and category affinity. Falls back to a tuned linear model when no trained LightGBM model is loaded.

### 🔀 MMR Diversity Reranking
Maximal Marginal Relevance optimization balances relevance against diversity — preventing creator concentration and topic repetition. Freshness decay and trending score injection are applied before diversity optimization.

### 📡 Live Session Engine
Real-time `SessionStateTracker` maintains per-session state: videos watched, dwell time, likes, skips, and recent categories — all with TTL-based eviction. The `SessionInterestModel` merges long-term user embeddings with real-time session signals (60% user + 40% session) for adaptive recommendations.

### 🎰 UCB1 Contextual Bandit
Exploration-exploitation balancing via UCB1 contextual bandit. Epsilon-greedy and Thompson sampling strategies available via config. Exploration items are injected into the final ranked list at a configurable rate.

### 🧪 A/B Testing Framework
Full experiment lifecycle: traffic splitting, variant assignment per request, impression logging, click tracking, and statistical significance testing. Experiment assignments are included in recommendation context and logged per-impression.

### 🛡 Trust & Safety Pipeline
Every candidate passes through toxicity classification, spam detection, and policy enforcement before serving. Unsafe content is filtered before reranking.

---

## ⚡ Quickstart

**1. Clone and install**
```bash
git clone https://github.com/Saikiran1923/enterprise-live-recommendation-platform.git
cd enterprise-live-recommendation-platform
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install faiss-cpu
```

**2. Load data**
```bash
# Windows PowerShell
$env:PYTHONPATH = "."

python scripts/load_youtube_data.py
# → videos loaded

python scripts/generate_synthetic_data.py
# → 1,000 users · 5,000 videos · 50,000 interactions
```

**3. Start the server**
```bash
$env:PYTHONPATH = "."
uvicorn api.main:app --reload
# API  → http://localhost:8000
# Docs → http://localhost:8000/docs
```

**4. Or start with Docker**
```bash
docker-compose up --build
```

---

## 🔁 Recommendation Pipeline

```
POST /recommend/
      │
      ▼
Generate user embedding
(from history or deterministic MD5 fallback)
      │
      ▼
Two-Tower FAISS retrieval  ──▶  500 candidates @ sub-ms
      │
      ▼
PreRanking filter  ──▶  ~200 candidates
      │
      ▼
LightGBM ranking  ──▶  scored + sorted (16 features)
      │
      ▼
Reranking: trending · freshness decay · MMR diversity
      │
      ▼
Top-K results  ──▶  4ms avg latency
```

---

## 🌐 API Reference

```bash
curl -X POST http://localhost:8000/recommend/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_1",
    "session_id": "session_1",
    "top_k": 5,
    "context": {},
    "exclude_video_ids": []
  }'
```

**Response:**

```json
{
  "user_id": "user_1",
  "session_id": "session_1",
  "recommendations": [
    {
      "video_id": "337",
      "rank": 1,
      "ranking_score": 1.145,
      "retrieval_score": 0.950,
      "source": "two_tower",
      "is_exploration": false
    },
    {
      "video_id": "183",
      "rank": 2,
      "ranking_score": 1.105,
      "retrieval_score": 0.943,
      "source": "two_tower",
      "is_exploration": false
    }
  ],
  "metadata": { "latency_ms": 4 },
  "request_id": "4fdd3436"
}
```

### Endpoint Reference

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `POST` | `/recommend/` | Public | Real-time top-K recommendations |
| `POST` | `/recommend/batch` | Public | Batch recommendations for multiple users |
| `POST` | `/events/` | Public | Ingest user interaction events |
| `GET` | `/experiments/` | Public | List active A/B experiments |
| `GET` | `/health` | Public | Health check |
| `GET` | `/docs` | Public | Interactive API documentation |

---

## 📡 Monitoring & Observability

### Prometheus Metrics

```
recommendations_served          # Total recommendations served
recommendation_latency_ms       # End-to-end latency histogram
candidates_retrieved            # Candidate pool size per request
ranking_score_distribution      # Score distribution across ranked items
exploration_rate                # % of recommendations from bandit
safety_filtered_count           # Items blocked by trust & safety
```

### Grafana Dashboards

Pre-built dashboards in `dashboards/`:

- **Recommendation Metrics** — request rate, latency, candidate pool size
- **Engagement Tracker** — CTR, watch time, session depth trends
- **Experiment Dashboard** — variant performance, statistical significance
- **System Health** — API uptime, error rates, memory usage

---

## 📊 Data

| Dataset | Size | Description |
|---------|------|-------------|
| USvideos.csv | ~40K rows | US YouTube trending videos |
| GBvideos.csv | ~38K rows | UK YouTube trending videos |
| INvideos.csv | ~37K rows | India YouTube trending videos |
| + 7 more regions | ~35K each | CA · DE · FR · JP · KR · MX · RU |
| users.json | 1,000 | Synthetic user profiles |
| interactions.json | 50,000 | Synthetic watch / like / skip events |
| videos.json | 5,000 | Synthetic video catalogue |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=services --cov=candidate_generation --cov-report=term-missing

# Run specific modules
pytest tests/test_candidate_generation.py -v
pytest tests/test_recommendation_pipeline.py -v
pytest tests/test_ranking_model.py -v
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| API Framework | FastAPI · Uvicorn |
| ANN Retrieval | FAISS IndexFlatIP |
| Ranking Model | LightGBM · NumPy fallback |
| Exploration | UCB1 Contextual Bandit |
| Embeddings | Custom Two-Tower · MD5 cold-start |
| Session Modelling | In-memory TTL state tracker |
| Data Processing | Pandas · NumPy |
| Monitoring | Prometheus · Grafana |
| Containerization | Docker · Docker Compose |
| Testing | pytest |
| Language | Python 3.11+ |

---

## 🗺 Roadmap

- [ ] Train LightGBM ranking model on real YouTube interaction data
- [ ] Redis-backed online feature store for sub-millisecond feature retrieval
- [ ] Kafka integration for real-time event ingestion
- [ ] Wire Collaborative Filtering and Matrix Factorization into candidate service
- [ ] Kubernetes deployment manifests
- [ ] Load testing with Locust — target 1K RPS at < 20ms p99
- [ ] Connect Trust & Safety pipeline to recommendation route

---

## 🎯 Who Is This For?

- **Recommendation Engineers** — complete retrieval-ranking-reranking reference architecture
- **ML Engineers** — production alternative to single-model recommendation scripts
- **Data Scientists** — structured platform for experimenting with ranking and retrieval models
- **MLOps Engineers** — blueprint for serving ML models with full observability

---

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">

Built for engineers who take recommendation systems seriously.

**⭐ Star this repo if it helped you build better ML systems.**

</div>