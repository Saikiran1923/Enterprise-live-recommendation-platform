"""Experiment management endpoints."""

from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/")
async def list_experiments(request: Request):
    manager = getattr(request.app.state, "experiment_manager", None)
    if not manager:
        return {"experiments": []}
    active = manager.list_active()
    return {"active_experiments": active, "count": len(active)}


@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str, request: Request):
    metrics = getattr(request.app.state, "experiment_metrics", None)
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not available")
    return metrics.compute_summary(experiment_id, "ctr")
