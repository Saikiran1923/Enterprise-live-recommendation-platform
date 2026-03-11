"""Event ingestion endpoints."""

import logging
from fastapi import APIRouter, Request, HTTPException
from api.schemas.request_schema import EventRequest
from api.schemas.response_schema import EventResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["events"])


@router.post("/", response_model=EventResponse)
async def ingest_event(event: EventRequest, request: Request):
    """Accept a user event and route it to the stream processor."""
    try:
        router_instance = getattr(request.app.state, "event_router", None)
        if router_instance:
            await router_instance.dispatch(event.dict())
        return EventResponse(event_id=event.event_id, accepted=True)
    except Exception as e:
        logger.error(f"Event ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Event ingestion failed")
