"""Shared LLM utility — single JSON call used by router and all agents."""
from __future__ import annotations

import json
import logging
import re
import time

from cerebras.cloud.sdk import Cerebras
from app.config import settings
from app.pipeline.llm import _call_with_retry  # reuse retry logic

logger = logging.getLogger(__name__)

_client: Cerebras | None = None


def _get_client() -> Cerebras:
    global _client
    if _client is None:
        _client = Cerebras(api_key=settings.cerebras_api_key)
    return _client


def llm_json_call(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 256,
) -> dict:
    """
    Single LLM call that returns a parsed JSON dict.
    Raises ValueError if JSON cannot be parsed.
    All agents and the router use this.
    """
    t0 = time.monotonic()
    response = _call_with_retry(
        _get_client(),
        model=settings.cerebras_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    text = response.choices[0].message.content.strip()
    logger.debug("llm_json_call latency=%.0fms response=%r", latency_ms, text[:200])

    # Strip markdown fencing if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error("llm_json_call JSON parse failed: %s | raw=%r", exc, text[:300])
        raise ValueError(f"LLM returned non-JSON: {text[:200]}") from exc


def llm_text_call(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 150,
) -> str:
    """Single LLM call returning plain text. Used for speak generation."""
    t0 = time.monotonic()
    response = _call_with_retry(
        _get_client(),
        model=settings.cerebras_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    text = response.choices[0].message.content.strip()
    logger.debug("llm_text_call latency=%.0fms response=%r", latency_ms, text[:200])
    return text
