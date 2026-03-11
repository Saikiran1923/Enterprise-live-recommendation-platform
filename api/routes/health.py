"""Health check endpoints."""

from fastapi import APIRouter, Request
from api.schemas.response_schema import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    monitor = getattr(request.app.state, "health_monitor", None)
    if monitor:
        await monitor.run_checks()
        status_data = monitor.get_status()
    else:
        status_data = {"healthy": True, "checks": {}}
    return HealthResponse(
        status="ok" if status_data["healthy"] else "degraded",
        healthy=status_data["healthy"],
        checks=status_data.get("checks", {}),
    )


@router.get("/ready")
async def readiness_check():
    return {"ready": True}
