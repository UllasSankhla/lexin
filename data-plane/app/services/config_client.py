"""HTTP client to fetch configuration from the control plane."""
import asyncio
import logging
import time
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# In-memory cache: token -> (config_dict, expiry_time)
_cache: dict[str, tuple[dict, float]] = {}
_cache_lock = asyncio.Lock()

# Shared config cache (for all calls, not per-token)
_shared_config: dict | None = None
_shared_config_expiry: float = 0.0
_shared_config_lock = asyncio.Lock()


async def get_cached_config(session_token: str) -> dict | None:
    """Return cached config for a session token."""
    async with _cache_lock:
        entry = _cache.get(session_token)
        if entry:
            config, expiry = entry
            if time.monotonic() < expiry:
                return config
            else:
                del _cache[session_token]
    return None


async def set_cached_config(session_token: str, config: dict) -> None:
    async with _cache_lock:
        _cache[session_token] = (config, time.monotonic() + settings.config_cache_ttl_sec)


async def invalidate_token(session_token: str) -> None:
    async with _cache_lock:
        _cache.pop(session_token, None)


async def fetch_config_from_control_plane() -> dict:
    """Fetch full config export from the control plane."""
    global _shared_config, _shared_config_expiry

    async with _shared_config_lock:
        if _shared_config and time.monotonic() < _shared_config_expiry:
            return _shared_config

        headers = {
            "X-API-Key": settings.control_plane_api_key,
            "Accept": "application/json",
        }
        url = f"{settings.control_plane_url}/api/v1/config/export"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                config = resp.json()
                _shared_config = config
                _shared_config_expiry = time.monotonic() + settings.config_cache_ttl_sec
                logger.info("Fetched config from control plane (%d params, %d FAQs)",
                            len(config.get("parameters", [])), len(config.get("faqs", [])))
                return config
        except httpx.HTTPError as e:
            logger.error("Failed to fetch config from control plane: %s", e)
            if _shared_config:
                logger.warning("Using stale cached config")
                return _shared_config
            raise
