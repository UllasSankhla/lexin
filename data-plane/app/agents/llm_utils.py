"""Shared LLM utility — single JSON call used by router and all agents."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

import anthropic
import openai as openai_sdk
from cerebras.cloud.sdk import Cerebras
from app.config import settings
from app.pipeline.llm import _call_with_retry  # reuse retry logic

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

_client = None
_client_provider: str | None = None


def _get_client():
    global _client, _client_provider
    provider = settings.llm_provider
    if _client is None or _client_provider != provider:
        if provider == "anthropic":
            _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        elif provider == "openai":
            _client = openai_sdk.OpenAI(api_key=settings.openai_api_key)
        else:
            _client = Cerebras(api_key=settings.cerebras_api_key)
        _client_provider = provider
    return _client


def _llm_call(system_prompt: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Route a chat completion through the configured provider and return content text."""
    provider = settings.llm_provider
    client = _get_client()
    if provider == "anthropic":
        response = _call_with_retry(
            client,
            model=settings.anthropic_model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content[0].text if response.content else ""
    elif provider == "openai":
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = _call_with_retry(
            client,
            model=settings.openai_model,
            messages=full_messages,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content if response.choices else ""
    else:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = _call_with_retry(
            client,
            model=settings.cerebras_model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content if response.choices else ""


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
    user_message: str,
    history: ConversationHistory | None,
) -> list[dict]:
    """Assemble the user/assistant messages list (system prompt is passed separately)."""
    msgs: list[dict] = []
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
    content = _llm_call(system_prompt, _build_messages(user_message, history), max_tokens, temperature=0.2)
    latency_ms = (time.monotonic() - t0) * 1000

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


def _resolve_refs(schema: dict, defs: dict) -> dict:
    """Recursively inline $ref references and add additionalProperties: false to objects."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        resolved = _resolve_refs(defs.get(ref_name, {}), defs)
        # Merge any sibling keys (rare in Pydantic output)
        return {**resolved, **{k: v for k, v in schema.items() if k != "$ref"}}

    result: dict = {}
    for key, value in schema.items():
        if key == "$defs":
            continue  # strip from output — all refs are inlined
        elif isinstance(value, dict):
            result[key] = _resolve_refs(value, defs)
        elif isinstance(value, list):
            result[key] = [_resolve_refs(v, defs) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value

    if result.get("type") == "object":
        # Cerebras strict mode requires every object to have `properties` or `anyOf`.
        # Open-ended dicts (dict[str, T]) only have `additionalProperties` — add an
        # empty `properties` so the schema is accepted without changing semantics.
        if "properties" not in result and "anyOf" not in result:
            result["properties"] = {}
        # Strict mode also requires additionalProperties: false on fixed-schema objects.
        # Leave open-ended dicts (those with non-False additionalProperties) untouched.
        if result.get("additionalProperties") is not False and "additionalProperties" not in result:
            result["additionalProperties"] = False

    return result


def _make_cerebras_schema(model) -> dict:
    """Convert a Pydantic model to a Cerebras-compatible strict JSON schema."""
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    return _resolve_refs(raw, defs)


def llm_structured_call(
    system_prompt: str,
    user_message: str,
    response_model: type,
    max_tokens: int = 1024,
    history: ConversationHistory | None = None,
) -> object:
    """
    Single LLM call that returns a validated Pydantic model instance.

    For OpenAI: uses client.beta.chat.completions.parse to enforce the schema.
    For Anthropic/Cerebras: parses JSON from the response text and validates it.
    Raises ValueError if parsing or validation fails.
    """
    t0 = time.monotonic()
    provider = settings.llm_provider

    if provider == "openai":
        # Use json_object mode to guarantee valid JSON, then validate with Pydantic
        client = _get_client()
        messages = [{"role": "system", "content": system_prompt}] + _build_messages(user_message, history)
        response = _call_with_retry(
            client,
            model=settings.openai_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ValueError("LLM returned empty content")
        try:
            return response_model.model_validate_json(content)
        except Exception as exc:
            logger.error("llm_structured_call (openai) validation failed: %s | raw=%r", exc, content[:300])
            raise ValueError(f"LLM structured response validation failed: {exc}") from exc

    client = _get_client()
    user_msgs = _build_messages(user_message, history)
    full_messages = [{"role": "system", "content": system_prompt}] + user_msgs

    if provider == "anthropic":
        # JSON prefill: append assistant turn starting with `{` to force JSON output.
        prefill_msgs = user_msgs + [{"role": "assistant", "content": "{"}]
        response = _call_with_retry(
            client,
            model=settings.anthropic_model,
            system=system_prompt,
            messages=prefill_msgs,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        raw_content = response.content[0].text if response.content else ""

        # If the model ignored the prefill, retry with an explicit correction.
        if raw_content and not raw_content.lstrip().startswith("{"):
            logger.debug("llm_structured_call: prefill ignored, retrying with JSON correction")
            correction = (
                f"{user_message}\n\n"
                "IMPORTANT: Your previous response was not valid JSON. "
                'Respond ONLY with a JSON object starting with {"'
            )
            retry_resp = _call_with_retry(
                client,
                model=settings.anthropic_model,
                system=system_prompt,
                messages=user_msgs + [{"role": "user", "content": correction}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            raw_content = retry_resp.content[0].text if retry_resp.content else raw_content

        # Prepend `{` if the API omitted it (continuation mode).
        stripped = raw_content.lstrip() if raw_content else ""
        if stripped and not stripped.startswith("{"):
            raw_content = "{" + stripped
    else:
        # Cerebras: use native structured outputs (json_schema + strict) —
        # guarantees valid JSON matching the Pydantic model schema.
        schema = _make_cerebras_schema(response_model)
        response = _call_with_retry(
            client,
            model=settings.cerebras_model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__.lower(),
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        raw_content = response.choices[0].message.content if response.choices else ""

    latency_ms = (time.monotonic() - t0) * 1000

    content = raw_content
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
        logger.error("llm_structured_call validation failed: %s | raw=%r", exc, text[:300])
        raise ValueError(f"LLM structured response validation failed: {exc}") from exc


def llm_text_call(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    history: ConversationHistory | None = None,
) -> str:
    """Single LLM call returning plain text. Used for speak generation."""
    t0 = time.monotonic()
    content = _llm_call(system_prompt, _build_messages(user_message, history), max_tokens, temperature=0.3)
    latency_ms = (time.monotonic() - t0) * 1000
    if not content:
        logger.error("llm_text_call received empty/None content from LLM")
        return ""
    text = content.strip()
    logger.debug("llm_text_call latency=%.0fms response=%r", latency_ms, text[:200])
    return text
