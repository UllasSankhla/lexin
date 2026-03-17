"""Customer API key authentication for the data plane.

Validates keys against the control plane API (no local JSON file needed).
Returns a tuple of (customer_name, owner_id) on success.
"""
from __future__ import annotations

import logging
import httpx

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


async def require_customer_key(x_customer_key: str = Header(...)) -> tuple[str, str]:
    """
    FastAPI dependency — validates the X-Customer-Key header against the control plane.

    Returns (customer_name, owner_id) on success; raises 401/403/503 on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{settings.control_plane_url}/api/v1/customer-keys/lookup",
                params={"key": x_customer_key},
                headers={"x-api-key": settings.control_plane_api_key},
            )
    except httpx.RequestError as exc:
        logger.error("Customer key lookup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable.",
        )

    if resp.status_code == 401:
        logger.warning("Rejected request with unknown customer key: %s...", x_customer_key[:8])
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid customer key.")

    if resp.status_code == 403:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Customer key is disabled.")

    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service error.")

    data = resp.json()
    customer_name = data["name"]
    owner_id = data["owner_id"]
    logger.info("Authenticated customer: %s (owner: %s)", customer_name, owner_id)
    return customer_name, owner_id
