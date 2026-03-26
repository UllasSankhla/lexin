"""Shared LLM utility — single JSON call used by router and all agents."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from cerebras.cloud.sdk import Cerebras
from app.config import settings
from app.pipeline.llm import _call_with_retry  # reuse retry logic

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

_client: Cerebras | None = None


def _get_client() -> Cerebras:
    global _client
    if _client is None:
        _client = Cerebras(api_key=settings.cerebras_api_key)
    return _client


# ── Conversation history ──────────────────────────────────────────────────────

class ConversationHistory:
    """
    Per-agent multi-turn message history for LLM calls.

    Stored in internal_state["llm_history"] as a plain list of dicts so it
    survives serialisation between turns.  Reconstruct with from_list() at the
    start of each process() call; call to_list() to persist it back.

    Usage pattern in an agent:
        history = ConversationHistory.from_list(internal_state.get("llm_history"))
        result  = llm_structured_call(system, user_msg, Model, history=history)
        history.add("user", user_msg)
        history.add("assistant", result.speak or "")
        internal_state["llm_history"] = history.to_list()
    """

    # Keep at most this many (user + assistant) pairs to bound token usage.
    _MAX_PAIRS = 10

    def __init__(self, messages: list[dict] | None = None) -> None:
        self._messages: list[dict] = list(messages or [])

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """Append a single message and trim to _MAX_PAIRS pairs if needed."""
        self._messages.append({"role": role, "content": content})
        # Each pair = 2 messages; trim from the front
        max_msgs = self._MAX_PAIRS * 2
        if len(self._messages) > max_msgs:
            self._messages = self._messages[-max_msgs:]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_list(self) -> list[dict]:
        """Return a serialisable copy for storage in internal_state."""
        return list(self._messages)

    @classmethod
    def from_list(cls, data: list[dict] | None) -> "ConversationHistory":
        return cls(data or [])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def messages(self) -> list[dict]:
        """Return a copy of the stored messages."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)


# ── LLM call helpers ──────────────────────────────────────────────────────────

def _build_messages(
    system_prompt: str,
    user_message: str,
    history: ConversationHistory | None,
) -> list[dict]:
    """Assemble the messages list: system → history turns → current user turn."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        msgs.extend(history.messages())
    msgs.append({"role": "user", "content": user_message})
    return msgs


def llm_json_call(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    history: ConversationHistory | None = None,
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
        messages=_build_messages(system_prompt, user_message, history),
        max_tokens=max_tokens,
        temperature=0.2,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    content = response.choices[0].message.content if response.choices else None
    if not content:
        logger.error("llm_json_call received empty/None content from LLM")
        raise ValueError("LLM returned empty content")

    text = content.strip()
    logger.debug("llm_json_call latency=%.0fms response=%r", latency_ms, text[:200])

    # Strip markdown fencing if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        # Attempt to repair a truncated JSON response by extracting individual
        # fields via regex before giving up. This handles the common case where
        # max_tokens cuts off the response mid-string.
        repaired = _try_repair_json(clean)
        if repaired:
            logger.warning(
                "llm_json_call repaired truncated JSON | recovered=%s | raw=%r",
                list(repaired.keys()), text[:300],
            )
            return repaired
        logger.error("llm_json_call JSON parse failed: %s | raw=%r", exc, text[:300])
        raise ValueError(f"LLM returned non-JSON: {text[:200]}") from exc


def _try_repair_json(text: str) -> dict | None:
    """Best-effort extraction of key fields from a truncated JSON string."""
    result = {}
    # Extract string fields: "key": "value" (value may be truncated)
    for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"?', text):
        result[m.group(1)] = m.group(2)
    # Extract bool/null fields: "key": true|false|null
    for m in re.finditer(r'"(\w+)"\s*:\s*(true|false|null)', text):
        v = {"true": True, "false": False, "null": None}[m.group(2)]
        result[m.group(1)] = v
    return result if result else None


def llm_structured_call(
    system_prompt: str,
    user_message: str,
    response_model: type,
    max_tokens: int = 1024,
    history: ConversationHistory | None = None,
) -> object:
    """
    Single LLM call that returns a validated Pydantic model instance.
    Uses response_format=json_object for structured output where supported,
    then validates the result against the given Pydantic model.
    Raises ValueError if parsing or validation fails.
    """
    # Note: response_format=json_object is intentionally NOT used here.
    # Cerebras truncates or returns empty content when JSON mode is forced.
    # The system prompt already instructs the model to return JSON only;
    # Pydantic validates the result structurally.
    t0 = time.monotonic()
    response = _call_with_retry(
        _get_client(),
        model=settings.cerebras_model,
        messages=_build_messages(system_prompt, user_message, history),
        max_tokens=max_tokens,
        temperature=0.1,
    )

    latency_ms = (time.monotonic() - t0) * 1000
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise ValueError("LLM returned empty content")

    text = content.strip()
    logger.debug("llm_structured_call latency=%.0fms response=%r", latency_ms, text[:300])

    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return response_model.model_validate_json(clean)
    except Exception as exc:
        repaired = _try_repair_json(clean)
        if repaired:
            try:
                return response_model.model_validate(repaired)
            except Exception:
                pass
        logger.error(
            "llm_structured_call validation failed: %s | raw=%r", exc, text[:300]
        )
        raise ValueError(f"LLM structured response validation failed: {exc}") from exc


def llm_text_call(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    history: ConversationHistory | None = None,
) -> str:
    """Single LLM call returning plain text. Used for speak generation."""
    t0 = time.monotonic()
    response = _call_with_retry(
        _get_client(),
        model=settings.cerebras_model,
        messages=_build_messages(system_prompt, user_message, history),
        max_tokens=max_tokens,
        temperature=0.3,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    content = response.choices[0].message.content if response.choices else None
    if not content:
        logger.error("llm_text_call received empty/None content from LLM")
        return ""
    text = content.strip()
    logger.debug("llm_text_call latency=%.0fms response=%r", latency_ms, text[:200])
    return text
