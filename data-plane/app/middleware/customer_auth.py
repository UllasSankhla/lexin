"""Customer API key authentication for the data plane.

Keys are loaded from a JSON file at startup and re-read on each request so
new keys can be added without restarting the service.

File format (customer_keys.json):
{
  "ck_live_abc123...": { "name": "Acme Corp",  "enabled": true },
  "ck_live_def456...": { "name": "Nexus Law",  "enabled": true },
  "ck_test_xyz789...": { "name": "Test Client","enabled": false }
}

Clients pass the key in the X-Customer-Key request header.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


def _load_keys() -> dict:
    """Read and parse the customer keys file. Returns empty dict on any error."""
    path = Path(settings.customer_keys_path)
    if not path.exists():
        logger.warning("Customer keys file not found: %s — all requests will be rejected", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load customer keys from %s: %s", path, exc)
        return {}


def require_customer_key(x_customer_key: str = Header(...)) -> str:
    """
    FastAPI dependency — validates the X-Customer-Key header.

    Returns the customer name on success; raises 401 / 403 on failure.
    Re-reads the keys file on every call so additions take effect immediately.
    """
    keys = _load_keys()

    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication unavailable — no customer keys configured.",
        )

    entry = keys.get(x_customer_key)
    if not entry:
        logger.warning("Rejected request with unknown customer key: %s...", x_customer_key[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid customer key.",
        )

    if not entry.get("enabled", True):
        logger.warning("Rejected request from disabled customer: %s", entry.get("name"))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Customer key is disabled.",
        )

    logger.info("Authenticated customer: %s", entry.get("name", "unknown"))
    return entry.get("name", "unknown")
