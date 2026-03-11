"""Pydantic request schemas for the recommendation API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Current session ID")
    top_k: int = Field(20, ge=1, le=100, description="Number of recommendations")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    exclude_video_ids: Optional[List[str]] = Field(default_factory=list)
    experiment_overrides: Optional[Dict[str, str]] = Field(default_factory=dict)


class EventRequest(BaseModel):
    event_id: str
    user_id: str
    event_type: str
    timestamp: float
    session_id: Optional[str] = None
    video_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchRecommendationRequest(BaseModel):
    user_ids: List[str] = Field(..., max_items=100)
    top_k: int = Field(20, ge=1, le=50)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
