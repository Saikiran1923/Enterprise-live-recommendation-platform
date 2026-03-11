"""Pydantic response schemas for the recommendation API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class RecommendedVideo(BaseModel):
    video_id: str
    rank: int
    ranking_score: float
    retrieval_score: Optional[float] = None
    source: Optional[str] = None
    is_exploration: bool = False
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    user_id: str
    session_id: Optional[str]
    recommendations: List[RecommendedVideo]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    healthy: bool
    checks: Dict[str, Any]
    version: str = "1.0.0"


class EventResponse(BaseModel):
    event_id: str
    accepted: bool
    message: str = "OK"
