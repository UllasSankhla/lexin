"""Fire webhook notifications on call completion."""
import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)


async def dispatch_webhooks(
    config: dict,
    call_id: str,
    duration_sec: float,
    collected: dict,
    transcript_path: str | None,
    caller_name: str = "",
    ai_summary: str = "",
    booking_details: dict | None = None,
) -> None:
    """Dispatch all enabled webhooks for call.completed event."""
    webhooks = config.get("webhooks", [])
    enabled = [w for w in webhooks if "call.completed" in w.get("events", [])]

    if not enabled:
        return

    payload = {
        "event": "call.completed",
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(duration_sec, 2),
        "caller_name": caller_name,
        "summary": ai_summary,
        "parameters": collected,
        "booking": booking_details,
        "transcript_available": transcript_path is not None,
    }

    tasks = [_fire_webhook(webhook, payload) for webhook in enabled]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for webhook, result in zip(enabled, results):
        if isinstance(result, Exception):
            logger.error("Webhook '%s' failed: %s", webhook["name"], result)
        else:
            logger.info("Webhook '%s' delivered successfully (status=%s)", webhook["name"], result)


async def _fire_webhook(webhook: dict, payload: dict) -> int:
    """Fire a single webhook with retries. Returns final HTTP status."""
    url = webhook["url"]
    timeout = webhook.get("timeout_sec", 10)
    max_retries = webhook.get("retry_count", 3)

    headers = {"Content-Type": "application/json"}
    if webhook.get("secret_header") and webhook.get("secret_value"):
        # HMAC-SHA256 signature
        body = json.dumps(payload).encode()
        sig = hmac.new(webhook["secret_value"].encode(), body, hashlib.sha256).hexdigest()
        headers[webhook["secret_header"]] = f"sha256={sig}"

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.is_success:
                    return resp.status_code
                logger.warning(
                    "Webhook '%s' returned %d (attempt %d/%d)",
                    webhook["name"], resp.status_code, attempt + 1, max_retries + 1
                )
        except httpx.RequestError as e:
            logger.warning(
                "Webhook '%s' request error: %s (attempt %d/%d)",
                webhook["name"], e, attempt + 1, max_retries + 1
            )

        if attempt < max_retries:
            backoff = 2 ** attempt
            await asyncio.sleep(backoff)

    raise Exception(f"All {max_retries + 1} webhook delivery attempts failed for {url}")
