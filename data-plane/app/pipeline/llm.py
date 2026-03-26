"""Cerebras LLM integration with dynamic prompt construction."""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Optional

# Delays in seconds between retry attempts (exponential backoff starting at 25 ms,
# factor 2, up to 6 retries: 0.025 → 0.05 → 0.1 → 0.2 → 0.4 → 0.8 s).
# Beyond 6 retries (~1.6 s total wait) we've waited long enough for a sync response.
_RETRY_DELAYS = tuple(0.025 * (2 ** i) for i in range(6))  # (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)

from cerebras.cloud.sdk import Cerebras

from app.config import settings


def _call_with_retry(client: Cerebras, **kwargs) -> object:
    """
    Call client.chat.completions.create with exponential-backoff retries.

    Uses _RETRY_DELAYS so all Cerebras calls share the same retry policy.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if delay is None:
                logger.error(
                    "LLM all %d retries exhausted — giving up (%s)",
                    len(_RETRY_DELAYS), last_exc,
                )
                raise
            logger.warning(
                "LLM attempt %d failed (%s), retrying in %.0fms…",
                attempt, exc, delay * 1000,
            )
            time.sleep(delay)

if TYPE_CHECKING:
    from app.services.calendar_service import TimeSlot

logger = logging.getLogger(__name__)

# System prompt used exclusively by LLMToolkit for focused extraction /
# generation tasks. It must NOT describe the fields to collect — that would
# cause the LLM to respond conversationally instead of returning bare values.
TOOLKIT_SYSTEM_PROMPT = (
    "You are a precise assistant supporting a voice appointment booking system. "
    "When asked to extract a value, respond with ONLY that value — no explanation, "
    "no conversation, no punctuation beyond what is part of the value itself. "
    "When asked to generate a voice phrase, produce only that phrase."
)


def build_system_prompt(config: dict) -> str:
    """Assemble the system prompt from control plane config."""
    assistant = config.get("assistant", {})
    parameters = config.get("parameters", [])
    faqs = config.get("faqs", [])
    context_files = config.get("context_files", [])
    spell_rules = config.get("spell_rules", [])

    lines = [assistant.get("system_prompt", "You are a helpful appointment booking assistant.")]

    if context_files:
        lines.append("\n## Business Context")
        for cf in context_files:
            lines.append(f"\n### {cf['name']}")
            if cf.get("description"):
                lines.append(f"({cf['description']})")
            lines.append(cf.get("content", ""))

    if faqs:
        lines.append("\n## Frequently Asked Questions")
        for faq in faqs:
            lines.append(f"\nQ: {faq['question']}\nA: {faq['answer']}")

    if parameters:
        lines.append("\n## Information to Collect")
        lines.append("You must collect the following information from the caller, one at a time:")
        for p in parameters:
            req = "required" if p.get("required") else "optional"
            hint = ""
            if p.get("extraction_hints"):
                hint = f" (hints: {', '.join(p['extraction_hints'])})"
            validation_note = ""
            if p.get("validation_message"):
                validation_note = f" [{p['validation_message']}]"
            lines.append(f"- {p['display_label']} (field: {p['name']}, {req}){hint}{validation_note}")

    lines.append("\n## Conversation Rules")
    lines.append(
        "- Collect one piece of information at a time.\n"
        "- Confirm each value before moving to the next.\n"
        "- Be friendly, concise, and professional.\n"
        "- If a value seems incorrect, politely ask to confirm or re-enter it.\n"
        "- When all required information is collected, summarize and confirm the appointment.\n"
        "- Keep responses short and conversational — this is a voice call."
    )

    if spell_rules:
        subs = [r for r in spell_rules if r.get("rule_type") == "substitution"]
        if subs:
            lines.append("\n## Speech Corrections")
            lines.append("Apply these corrections to what you hear from the caller:")
            for r in subs:
                lines.append(f"- '{r['wrong_form']}' → '{r['correct_form']}'")

    return "\n".join(lines)


class LLMClient:
    """Synchronous wrapper around Cerebras API (run in executor for async use)."""

    def __init__(self, system_prompt: str):
        self._client = Cerebras(api_key=settings.cerebras_api_key)
        self._system_prompt = system_prompt
        self._model = settings.cerebras_model

    def complete(
        self,
        conversation_history: list[dict],
        extra_system_note: str | None = None,
    ) -> tuple[str, int, float]:
        """
        Returns (response_text, token_count, latency_ms).
        extra_system_note injects a transient instruction (e.g. validation failures).
        """
        system_content = self._system_prompt
        if extra_system_note:
            system_content += f"\n\n[SYSTEM NOTE]: {extra_system_note}"

        messages = [{"role": "system", "content": system_content}] + conversation_history

        # ── Request logging ───────────────────────────────────────────────────
        logger.info(
            "LLM request | model=%s turns=%d%s",
            self._model,
            len(conversation_history),
            f" extra_note={extra_system_note!r}" if extra_system_note else "",
        )
        logger.debug("LLM system prompt | %s", system_content[:300])
        for i, msg in enumerate(conversation_history):
            logger.debug(
                "LLM message[%d] | role=%s content=%r",
                i, msg["role"], msg["content"][:200],
            )

        t0 = time.monotonic()
        response = _call_with_retry(
            self._client,
            model=self._model,
            messages=messages,
            max_tokens=settings.cerebras_max_tokens,
            temperature=settings.cerebras_temperature,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        text = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0

        # ── Response logging ──────────────────────────────────────────────────
        logger.info(
            "LLM response | tokens=%d latency=%.0fms response=%r",
            tokens, latency_ms, text[:200],
        )
        return text, tokens, latency_ms


# ── Focused LLM helpers for the booking workflow ──────────────────────────────

class LLMToolkit:
    """
    Single-turn, task-specific LLM calls used by BookingWorkflow.

    Each method sends a focused prompt with no conversation history —
    fast, cheap, and easy to reason about.

    All methods are synchronous; run them in an executor from async code.
    """

    # Type guidance injected into extraction prompts
    _TYPE_GUIDANCE: dict[str, str] = {
        "email":            "Include the full address with the @ symbol and domain.",
        "phone":            "Include digits only; preserve area code.",
        "date":             "Normalise to YYYY-MM-DD format.",
        "name":             "Capture the full name — first name followed by last name. The caller will often pause briefly between their first and last name; wait for both parts and combine them into a single 'First Last' value.",
        "number":           "Return only the numeric value.",
        "case_description": (
            "Summarise the caller's description into a clear, factual account. "
            "Capture the key facts, dates, parties involved, and the nature of the issue. "
            "Maximum 250 characters. Return only the summary, no labels."
        ),
    }

    # Voice-friendly instructions for asking questions
    _TYPE_ASK_HINT: dict[str, str] = {
        "email":            "Ask them to spell it out if needed.",
        "phone":            "Remind them to include the area code.",
        "date":             "Accept natural language like 'January fifteenth'.",
        "name":             "Ask for their first name first, then their last name. Let them know there's no rush — they can take a brief pause between first and last name. Also invite them to spell it out for convenience.",
        "case_description": (
            "Let them know they can take their time and share as many details as they like. "
            "Ask them to describe what happened, when it occurred, and who was involved."
        ),
    }

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        # Optional callback fired after every LLM call with (purpose, latency_ms, tokens).
        # Set by the handler to record per-call analytics without coupling this module to it.
        self.on_llm_response: Optional[callable] = None

    # ── Private helper ────────────────────────────────────────────────────────

    def _one_shot(self, prompt: str, purpose: str = "one_shot", max_tokens: int = 1024) -> str:
        """Single-turn call with no history; returns stripped response text."""
        logger.info("LLM toolkit call | purpose=%s prompt=%r", purpose, prompt[:200])
        result, tokens, latency_ms = self._client.complete(
            [{"role": "user", "content": prompt}],
        )
        logger.info("LLM toolkit result | purpose=%s result=%r", purpose, result[:200])
        if self.on_llm_response is not None:
            try:
                self.on_llm_response(purpose, latency_ms, tokens)
            except Exception:
                pass
        return result.strip()

    # ── Field question generation ─────────────────────────────────────────────

    def generate_field_question(self, field_label: str, field_type: str) -> str:
        """Generate a short, natural voice question for a booking field."""
        hint = self._TYPE_ASK_HINT.get(field_type, "")
        prompt = (
            f"Generate a single friendly voice question asking a caller for their {field_label}. "
            f"{hint} "
            f"Keep it to one sentence. No preamble, no formatting."
        )
        return self._one_shot(prompt, purpose=f"field_question:{field_label}")

    # ── Value extraction ──────────────────────────────────────────────────────

    def extract_value(self, field_label: str, field_type: str, utterance: str) -> Optional[str]:
        """
        Extract a field value from the caller's utterance.
        Returns None if nothing usable was found.
        """
        guidance = self._TYPE_GUIDANCE.get(field_type, "")
        prompt = (
            f"Extract the caller's {field_label} from the following statement.\n"
            f"Caller said: \"{utterance}\"\n"
            f"{guidance}\n"
            f"Respond with ONLY the extracted value. "
            f"If you cannot confidently extract it, respond with exactly: NONE"
        )
        result = self._one_shot(prompt, purpose=f"extract_value:{field_label}")
        if not result or result.strip().upper() == "NONE":
            return None
        return result.strip()

    def extract_correction(self, field_label: str, field_type: str, utterance: str) -> Optional[str]:
        """
        Extract a corrected value from a rejection/correction utterance.
        Returns None if no clear correction was embedded.
        """
        guidance = self._TYPE_GUIDANCE.get(field_type, "")
        prompt = (
            f"The caller is correcting their {field_label}.\n"
            f"Caller said: \"{utterance}\"\n"
            f"{guidance}\n"
            f"Extract the corrected {field_label} value. "
            f"Respond with ONLY the value, or NONE if it is not present."
        )
        result = self._one_shot(prompt, purpose=f"extract_correction:{field_label}")
        if not result or result.strip().upper() == "NONE":
            return None
        return result.strip()

    def classify_case_type(self, description: str) -> str:
        """
        Classify a case description into a short case type label (2-5 words, title case).
        e.g. 'Personal Injury', 'Contract Dispute', 'Employment Issue'.
        """
        prompt = (
            f"Classify this legal/service case description into a single brief case type label "
            f"(2-5 words, title case).\n"
            f"Description: \"{description}\"\n"
            f"Examples: Personal Injury, Contract Dispute, Employment Issue, Family Law, "
            f"Real Estate Matter, Criminal Defense, Estate Planning, Business Litigation.\n"
            f"Respond with ONLY the case type label."
        )
        return self._one_shot(prompt, purpose="classify_case_type")

    # ── Spell-back ────────────────────────────────────────────────────────────

    def spell_back(self, field_label: str, field_type: str, value: str) -> str:
        """Generate a natural voice read-back phrase asking for confirmation."""
        if field_type == "email":
            prompt = (
                f"The caller provided their email address as: {value}\n"
                f"Generate a single voice sentence reading this back for confirmation. "
                f"Read the email naturally (e.g. 'john dot smith at example dot com'). "
                f"End with 'Is that correct?'"
            )
        elif field_type == "phone":
            prompt = (
                f"The caller provided their phone number as: {value}\n"
                f"Generate a single voice sentence reading the digits back clearly, "
                f"grouped naturally (e.g. area code, then three, then four digits). "
                f"End with 'Is that correct?'"
            )
        else:
            prompt = (
                f"Generate a single voice sentence reading back a {field_label} value for confirmation.\n"
                f"Value: {value}\n"
                f"Example pattern: 'I have your [field] as [value]. Is that correct?'\n"
                f"Keep it natural and concise."
            )
        return self._one_shot(prompt, purpose=f"spell_back:{field_label}")

    # ── Slot presentation ─────────────────────────────────────────────────────

    def present_slots(self, slots: list["TimeSlot"]) -> str:
        """Present available time slots in a voice-friendly way."""
        numbered = "\n".join(f"{i + 1}. {s.description}" for i, s in enumerate(slots))
        prompt = (
            f"Present these available appointment slots to a caller over the phone.\n"
            f"Slots:\n{numbered}\n"
            f"Be friendly and concise. Ask which one they prefer. "
            f"Do NOT use bullet points or markdown. Keep it under three sentences."
        )
        return self._one_shot(prompt, purpose="present_slots", max_tokens=1024)

    def extract_slot_choice(self, utterance: str, slots: list["TimeSlot"]) -> Optional["TimeSlot"]:
        """
        Determine which slot the caller chose.
        Returns the TimeSlot object, or None if unclear.
        """
        numbered = "\n".join(f"{i + 1}. {s.description}" for i, s in enumerate(slots))
        prompt = (
            f"The caller is choosing from these appointment slots:\n{numbered}\n"
            f"Caller said: \"{utterance}\"\n"
            f"Respond with ONLY the slot number (e.g. 1, 2, 3). "
            f"If unclear, respond with NONE."
        )
        result = self._one_shot(prompt, purpose="extract_slot_choice")
        if not result or result.strip().upper() == "NONE":
            return None
        match = re.search(r"\d+", result)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(slots):
                return slots[idx]
        return None

    def confirm_slot(self, slot: "TimeSlot") -> str:
        """Generate a confirmation question for the chosen slot."""
        prompt = (
            f"Generate a single voice sentence confirming an appointment slot and asking for final confirmation.\n"
            f"Slot: {slot.description}\n"
            f"Example: 'I'll book you in for [slot]. Shall I confirm this appointment?'\n"
            f"One sentence only."
        )
        return self._one_shot(prompt, purpose="confirm_slot")

    def confirm_booking(self, slot: "TimeSlot", booking: dict, collected: dict) -> str:
        """Generate a warm booking confirmation message."""
        booking_id = booking.get("booking_id", "N/A")
        has_email = bool(booking.get("cancel_url") or booking.get("reschedule_url"))
        # Build a readable summary of all collected fields so the LLM can
        # personalise the message without the toolkit guessing field names.
        field_summary = "\n".join(f"- {k}: {v}" for k, v in collected.items() if v)
        email_note = (
            " Let them know a confirmation email with cancel and reschedule links has been sent."
            if has_email else ""
        )
        prompt = (
            f"Generate a warm, brief confirmation message for a successfully booked appointment.\n"
            f"Appointment time: {slot.description}\n"
            f"Booking reference: {booking_id}\n"
            f"Caller details:\n{field_summary or '(none provided)'}\n"
            f"Thank them by name if a name is present, mention the booking reference, "
            f"and wish them well.{email_note} "
            f"Two to three sentences, conversational voice style."
        )
        return self._one_shot(prompt, purpose="confirm_booking", max_tokens=1024)

    # ── Date preference extraction ────────────────────────────────────────────

    def extract_date_preference(self, utterance: str, current_date: str) -> Optional[dict]:
        """
        Convert a natural-language time preference to a UTC date range dict.

        Returns {"start_time": "YYYY-MM-DDTHH:MM:SSZ", "end_time": "YYYY-MM-DDTHH:MM:SSZ"}
        or None if the utterance contains no usable date preference.
        """
        import json as _json
        prompt = (
            f"Today's date is {current_date}.\n"
            f"The caller expressed this time preference: \"{utterance}\"\n"
            f"Convert it to an explicit UTC date range using these rules:\n"
            f"- Dates must be in the future\n"
            f"- 'next week' = Monday through Sunday of the next calendar week\n"
            f"- A day name (e.g. 'Tuesday') = the next occurrence of that day\n"
            f"- 'morning' = 08:00–12:00, 'afternoon' = 12:00–17:00, 'evening' = 17:00–20:00\n"
            f"- No time specified = full day 00:00–23:59\n"
            f"- Times in UTC\n"
            f"Respond with ONLY valid JSON in this exact format:\n"
            f'  {{"start_time": "YYYY-MM-DDTHH:MM:SSZ", "end_time": "YYYY-MM-DDTHH:MM:SSZ"}}\n'
            f"If you cannot determine a date range, respond with exactly: NONE"
        )
        result = self._one_shot(prompt, purpose="extract_date_preference", max_tokens=1024)
        if not result or result.strip().upper() == "NONE":
            return None
        try:
            # Strip any markdown fencing the LLM might add
            clean = result.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = _json.loads(clean)
            if "start_time" in parsed and "end_time" in parsed:
                return parsed
        except Exception:
            pass
        return None

    # ── Event type matching ────────────────────────────────────────────────────

    def match_event_type(
        self,
        purpose: str,
        event_types: list[dict],
    ) -> Optional[str]:
        """
        Use LLM to select the best-matching Calendly event type for the caller's purpose.

        event_types is a list of dicts with keys: name, description, event_type_uri.
        Returns the event_type_uri of the best match, or None if no match is found.
        """
        if not event_types:
            return None

        numbered = "\n".join(
            f"{i + 1}. {et['name']}"
            + (f" — {et['description']}" if et.get("description") else "")
            for i, et in enumerate(event_types)
        )
        prompt = (
            f"A caller wants to book an appointment for: \"{purpose}\"\n"
            f"Available appointment types:\n{numbered}\n"
            f"Respond with ONLY the number of the best matching appointment type. "
            f"If none of them match, respond with exactly: NONE"
        )
        result = self._one_shot(prompt, purpose="match_event_type")
        if not result or result.strip().upper() == "NONE":
            return None
        match = re.search(r"\d+", result)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(event_types):
                return event_types[idx]["event_type_uri"]
        return None
