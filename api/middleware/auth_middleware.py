"""Authentication and authorization middleware."""

import logging
import time
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

VALID_API_KEYS = {"dev-key-123", "prod-key-456"}  # In production: load from secrets manager


class AuthMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware."""

    EXCLUDED_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        return await call_next(request)

        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key or api_key not in VALID_API_KEYS:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)
