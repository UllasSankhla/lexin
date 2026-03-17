from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

EXEMPT_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS or request.url.path.startswith("/docs"):
            return await call_next(request)

        if settings.app_env == "development":
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key != settings.control_plane_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
