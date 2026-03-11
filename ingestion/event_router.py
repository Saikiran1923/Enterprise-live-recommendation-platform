"""Routes incoming events to appropriate processing pipelines."""

import logging
from typing import Dict, Any, Callable, List
from ingestion.event_schema import EventType

logger = logging.getLogger(__name__)


class EventRouter:
    """
    Routes events to multiple downstream processors based on event type.
    Supports priority routing, filtering, and fanout patterns.
    """

    def __init__(self):
        self._routes: Dict[str, List[Callable]] = {}
        self._middleware: List[Callable] = []
        self._filtered_count = 0
        self._routed_count = 0

    def add_middleware(self, fn: Callable) -> None:
        """Add a middleware function applied to all events before routing."""
        self._middleware.append(fn)

    def route(self, event_type: str):
        """Decorator to register a handler for an event type."""
        def decorator(fn: Callable) -> Callable:
            if event_type not in self._routes:
                self._routes[event_type] = []
            self._routes[event_type].append(fn)
            return fn
        return decorator

    async def dispatch(self, event: Dict[str, Any]) -> None:
        """Dispatch an event through middleware and to all matching handlers."""
        # Run middleware
        for mw in self._middleware:
            event = await mw(event)
            if event is None:
                self._filtered_count += 1
                return

        event_type = event.get("event_type", "unknown")
        handlers = self._routes.get(event_type, []) + self._routes.get("*", [])

        if not handlers:
            logger.debug(f"No handlers for event type: {event_type}")
            return

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event_type}: {e}")

        self._routed_count += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "routed": self._routed_count,
            "filtered": self._filtered_count,
            "routes": list(self._routes.keys()),
        }


# Default router instance
router = EventRouter()


@router.route(EventType.VIDEO_VIEW)
async def handle_video_view(event: Dict[str, Any]) -> None:
    logger.debug(f"Video view: user={event.get('user_id')}, video={event.get('video_id')}")


@router.route(EventType.RECOMMENDATION_CLICK)
async def handle_recommendation_click(event: Dict[str, Any]) -> None:
    logger.debug(f"Rec click: user={event.get('user_id')}, rank={event.get('rank_position')}")
