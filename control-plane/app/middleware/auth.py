"""Auth dependencies for the control plane.

Two auth paths:
  - get_current_user: Supabase JWT validation for browser admin clients.
  - require_service_key: Static API key for data-plane service-to-service calls.
"""
import time
import logging
import httpx

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)

# Simple in-memory token → (user_id, expiry) cache to avoid hitting Supabase on every request
_token_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 300  # seconds


async def get_current_user(authorization: str = Header(...)) -> str:
    """Validate a Supabase JWT and return the user's UUID (sub claim).

    Caches valid tokens for 5 minutes.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")

    token = authorization[7:]

    cached = _token_cache.get(token)
    if cached and time.monotonic() < cached[1]:
        return cached[0]

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{settings.supabase_url}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": settings.supabase_anon_key,
                },
            )
    except httpx.RequestError as exc:
        logger.error("Supabase auth request failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")

    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user_id: str = resp.json()["id"]
    _token_cache[token] = (user_id, time.monotonic() + _CACHE_TTL)
    logger.debug("Authenticated user: %s", user_id)
    return user_id


def require_service_key(x_api_key: str = Header(...)) -> None:
    """Validate the static service-to-service API key (used by the data plane)."""
    if x_api_key != settings.control_plane_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service key")
