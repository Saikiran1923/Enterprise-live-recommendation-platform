"""Recommendation endpoint routes."""

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from api.schemas.request_schema import RecommendationRequest, BatchRecommendationRequest
from api.schemas.response_schema import RecommendationResponse, RecommendedVideo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommend", tags=["recommendations"])


@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(
    req: RecommendationRequest,
    request: Request,
):
    """Generate personalized video recommendations for a user."""
    engine = request.app.state.recommendation_engine
    if not engine:
        raise HTTPException(status_code=503, detail="Recommendation engine not ready")

    try:
        result = await engine.recommend(
            user_id=req.user_id,
            session_id=req.session_id or str(uuid.uuid4()),
            context=req.context or {},
            top_k=req.top_k,
        )
        recs = [
            RecommendedVideo(
                video_id=r["video_id"],
                rank=r.get("rank", i + 1),
                ranking_score=r.get("ranking_score", 0.0),
                retrieval_score=r.get("retrieval_score"),
                source=r.get("source"),
                is_exploration=r.get("is_exploration", False),
            )
            for i, r in enumerate(result.get("recommendations", []))
        ]
        return RecommendationResponse(
            user_id=req.user_id,
            session_id=req.session_id,
            recommendations=recs,
            metadata=result.get("metadata", {}),
            request_id=getattr(request.state, "request_id", None),
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal recommendation error")


@router.post("/batch", response_model=list)
async def get_batch_recommendations(req: BatchRecommendationRequest, request: Request):
    """Generate recommendations for multiple users in one request."""
    engine = request.app.state.recommendation_engine
    results = []
    for user_id in req.user_ids:
        try:
            r = await engine.recommend(user_id=user_id, session_id=user_id,
                                       context=req.context or {}, top_k=req.top_k)
            results.append({"user_id": user_id, "recommendations": r.get("recommendations", [])})
        except Exception as e:
            results.append({"user_id": user_id, "error": str(e)})
    return results
