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

# Shared config cache: owner_id -> (config_dict, expiry_time)
_owner_config_cache: dict[str, tuple[dict, float]] = {}
_owner_config_lock = asyncio.Lock()


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


async def fetch_config_from_control_plane(owner_id: str) -> dict:
    """Fetch full config export from the control plane for a specific owner."""
    async with _owner_config_lock:
        entry = _owner_config_cache.get(owner_id)
        if entry:
            config, expiry = entry
            if time.monotonic() < expiry:
                return config

        url = f"{settings.control_plane_url}/api/v1/config/export"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    params={"owner_id": owner_id},
                    headers={
                        "x-api-key": settings.control_plane_api_key,
                        "Accept": "application/json",
                    },
                )
                resp.raise_for_status()
                config = resp.json()
                _owner_config_cache[owner_id] = (config, time.monotonic() + settings.config_cache_ttl_sec)
                logger.info(
                    "Fetched config for owner=%s: %d params, %d FAQs",
                    owner_id, len(config.get("parameters", [])), len(config.get("faqs", [])),
                )
                return config
        except httpx.HTTPError as e:
            logger.error("Failed to fetch config from control plane: %s", e)
            # Fall back to stale cache if available
            entry = _owner_config_cache.get(owner_id)
            if entry:
                logger.warning("Using stale cached config for owner=%s", owner_id)
                return entry[0]
            raise
