"""Request/response logging middleware."""

import time
import uuid
import logging
import json
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.time()
        request.state.request_id = request_id

        response = await call_next(request)

        latency_ms = (time.time() - start) * 1000
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "user_agent": request.headers.get("user-agent", ""),
        }))
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(round(latency_ms, 2))
        return response
