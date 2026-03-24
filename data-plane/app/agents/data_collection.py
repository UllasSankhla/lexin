"""Data collection agent — single mega-prompt approach.

One llm_structured_call per turn handles extraction, confirmation,
out-of-order collection, spelled-out inputs, corrections, and
interruptibility via UNHANDLED status.
"""
from __future__ import annotations

import json
import logging
import re

from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.data_collection_schema import DataCollectionLLMResponse
from app.agents.llm_utils import llm_structured_call

logger = logging.getLogger(__name__)

# Built-in regex patterns for common data types (Python-side validation)
_BUILTIN_PATTERNS = {
    "email":  r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
    "phone":  r"^[\+]?[\d\s\-\(\)]{7,20}$",
    "date":   r"^\d{4}-\d{2}-\d{2}$",
    "number": r"^\d+(\.\d+)?$",
}

# Default extraction hints injected into the prompt for each data type.
# Operator-defined extraction_hints from the DB are appended after these.
_DEFAULT_HINTS: dict[str, list[str]] = {
    "name": [
        "Accept first and last name together; caller may provide them in one or two utterances.",
        "SPELLED-OUT NAMES — convert character-by-character input to words:",
        "  • Hyphenated: 'S-A-R-A-H' → 'Sarah' | 'M-I-T-C-H-E-L-L' → 'Mitchell'",
        "  • Spaced letters: 'S A R A H' → 'Sarah'",
        "  • NATO phonetic: 'sierra alpha romeo alpha hotel' → 'Sarah'",
        "  • Mixed: 'J-O-H-N sierra echo alpha november' → 'John Sean'",
        "If a NAME BUFFER is shown in the state section, the caller already gave a first name. "
        "Combine the buffered first name with any last name in this utterance to form the full name. "
        "Set pending_confirmation.value = '<buffered_first> <new_last>'.",
        "Confirm spelling if the name sounds hyphenated, unusual, compound, or foreign.",
        "Do NOT set pending_confirmation for a single-word name — Python will buffer it and ask for the last name.",
    ],
    "email": [
        "Spoken form: 'john at example dot com' → john@example.com",
        "'dot', 'period' → .  |  'hyphen', 'dash' → -  |  'underscore' → _",
        "If the caller says it quickly and it sounds ambiguous, ask them to spell the local part character by character.",
        "Common spoken domains: 'gmail dot com', 'yahoo dot com', 'hotmail dot com'",
    ],
    "phone": [
        "Accept digit groups with pauses: '415... 555... 0192' → 4155550192",
        "'oh' and 'zero' both mean 0.",
        "US/Canada: 10 digits, area code first. Ask for area code if missing.",
        "'+1' country code is optional — strip it for storage.",
    ],
    "date": [
        "Accept natural language: 'last Tuesday', 'March 4th', 'the 15th of March'.",
        "Normalize to YYYY-MM-DD.",
        "If the year is not stated, assume the most recent past occurrence of that date.",
        "If only month and day are given, confirm the year before moving on.",
    ],
    "text": [
        "Accept the value as spoken; clean up obvious filler words.",
    ],
    "number": [
        "Extract digits only. 'forty-two' → 42. 'one hundred' → 100.",
    ],
}

# Deterministic opening questions per type — no LLM call needed for the first prompt
_OPENING_QUESTIONS: dict[str, str] = {
    "name":   "Could I start with your full name, please?",
    "email":  "What's your email address? Feel free to spell it out.",
    "phone":  "What's the best phone number to reach you? Please include the area code.",
    "date":   "What date are we looking at?",
    "number": "Could you provide that number for me?",
}


def _build_fields_block(parameters: list[dict]) -> str:
    lines = []
    for i, param in enumerate(parameters, 1):
        field_type = param.get("data_type", "text")
        required_str = "yes" if param.get("required", True) else "no"
        lines.append(f"{i}. {param['display_label']}")
        lines.append(f"   key: {param['name']} | type: {field_type} | required: {required_str}")
        if param.get("validation_message"):
            lines.append(f"   Validation: {param['validation_message']}")
        hints = list(_DEFAULT_HINTS.get(field_type, _DEFAULT_HINTS["text"]))
        hints += param.get("extraction_hints") or []
        for hint in hints:
            lines.append(f"   - {hint}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_mega_prompt(
    parameters: list[dict],
    collected: dict,
    pending: dict | None,
    persona_name: str,
    name_buffer: dict | None = None,
) -> str:
    fields_block = _build_fields_block(parameters)
    collected_json = json.dumps(collected, indent=2) if collected else "{}"
    pending_json = json.dumps(pending, indent=2) if pending else "none"

    name_buffer_block = ""
    if name_buffer:
        name_buffer_block = (
            f"\nNAME BUFFER (caller gave first name only in previous turn):\n"
            f"  Field: {name_buffer['field']} | First name so far: {name_buffer['first_name']}\n"
            f"  → Combine with the last name the caller provides now.\n"
            f"    Set pending_confirmation.value = \"{name_buffer['first_name']} <last_name>\".\n"
        )

    return f"""\
You are an AI intake receptionist conducting a voice call on behalf of {persona_name}.
Your role right now is to collect specific information from the caller.
Be warm, patient, and concise — this is a voice call, not a form.

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
FIELDS TO COLLECT
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
{fields_block}

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
CURRENT COLLECTION STATE
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
Already collected and confirmed:
{collected_json}

Awaiting caller's yes/no confirmation for:
{pending_json}
{name_buffer_block}
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
EXTRACTION RULES
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
1. OUT-OF-ORDER: Extract any field the caller mentions, even if it is not the
   current one. If the caller provides field 3 before field 1, accept it.

2. MULTIPLE FIELDS AT ONCE: A single utterance may contain values for several
   fields. Extract all of them. Queue only one pending_confirmation at a time.

3. SPELLED-OUT INPUT: Callers often spell things out on voice calls.
   - Letter-by-letter: "J-O-H-N" -> "John"
   - NATO alphabet: "juliet oscar hotel november" -> "John"
   - Spoken punctuation: see per-field hints above

4. CORRECTIONS: If the caller says "actually...", "wait, I meant...",
   "no it's...", or "make that..." treat the new value as a correction
   for the field currently pending confirmation, or the most recently
   discussed field.

5. INCOMPLETE UTTERANCE: If the caller's speech seems cut off mid-sentence
   (e.g. ends on "my email is..." with no address following), do not guess.
   Set incomplete_utterance=true and ask them to continue.

6. SKIP (optional fields only): If the caller says "skip", "I don't have one",
   "not applicable", or "I'd rather not say" record the field as "" in extracted.

7. DO NOT INVENT: Never guess or fabricate a value. If you cannot confidently
   extract a value, do not put it in extracted.

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
CONFIRMATION RULES
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
ALWAYS read back and confirm every extracted value before recording it.
There are no exceptions - names, dates, phone numbers, emails all require
a confirmation step.

Voice-friendly read-back formats:
- Name:  say naturally - "Sarah Mitchell"
- Email: "sarah dot mitchell at gmail dot com"
         Never say "@" as "at sign" - always say "at".
- Phone: group in natural chunks - "four one five, five five five, zero one nine two"
- Date:  "March fourth, twenty twenty-four"

Always end the confirmation question with: "Is that correct?"

If "Awaiting caller's yes/no confirmation for" above is not "none", interpret
this utterance as a yes/no FIRST before attempting any field extraction:
- YES signals: "yes", "yep", "correct", "that's right", "uh huh", "sure",
               "sounds good", "affirmative", "go ahead"
- NO signals:  "no", "nope", "wrong", "that's not it", "incorrect"
- CORRECTION:  "actually...", "no it's...", "make that...", "wait..."
               Set intent=correction, put new value in correction_value,
               and put the corrected value in pending_confirmation with the
               same field key as the current pending confirmation.
- Ambiguous:   Set intent=answer and ask explicitly in speak:
               "Is [value] correct? Please say yes or no."

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
CANNOT-PROCESS CONDITIONS
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
Set cannot_process=true and status="unhandled" when:
- The utterance is a question unrelated to providing field values
  (e.g. "How long will this take?", "What is this call about?")
- The caller expresses a desire to stop, cancel, or speak to a person
  (e.g. "I want to speak to someone", "Can we stop?", "I need to cancel")
- The utterance is complete gibberish or clearly a transcription error
  (single random syllables, no discernible words or field context)

When cannot_process=true:
- Set speak="" - the router decides the response
- Preserve pending_confirmation exactly as it appears above - do NOT change it
- Set cannot_process_reason to one of:
    "off_topic_question" | "wants_human" | "wants_to_cancel" | "transcription_noise"

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
RESPONSE FORMAT
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
Return ONLY valid JSON matching this exact schema. No markdown. No explanation.

{{
  "intent": "<answer|confirm_yes|confirm_no|correction|skip|off_topic|incomplete_utterance>",
  "extracted": {{"<field_key>": "<normalized_value>"}},
  "correction_value": "<string or null>",
  "speak": "<voice-friendly, warm, 1-2 sentences — blank if cannot_process>",
  "status": "<in_progress|waiting_confirm|completed|unhandled>",
  "pending_confirmation": {{"field": "<field_key>", "value": "<value>"}} or null,
  "incomplete_utterance": false,
  "cannot_process": false,
  "cannot_process_reason": null
}}

Rules:
- extracted contains only NEW values from this turn - never repeat confirmed fields
- status is "waiting_confirm" whenever pending_confirmation is non-null
- status is "completed" only when ALL required fields are confirmed AND pending_confirmation is null
- speak is "" when cannot_process=true
- pending_confirmation is ALWAYS preserved unchanged on unhandled responses
- Queue only one pending_confirmation at a time; extras wait for the next turn
"""


class DataCollectionAgent(AgentBase):
    """
    Collects required parameters using a single mega-prompt per turn.

    Supports out-of-order collection, spelled-out values, corrections,
    and interruptibility: if the LLM cannot interpret the utterance
    (off-topic question, wants to cancel, noise), it returns UNHANDLED
    so the router can dispatch to the appropriate interrupt agent.
    When the interrupt agent finishes, data collection resumes from
    exactly where it left off via the preserved internal_state.

    Internal state keys:
      collected:            dict[str, str]  — confirmed field values
      pending_confirmation: dict | None     — {field, value} awaiting yes/no
      retry_count:          int             — consecutive LLM call failures
    """

    MAX_RETRIES = 3

    def process(
        self,
        utterance: str,
        internal_state: dict,
        config: dict,
        history: list[dict],
    ) -> SubagentResponse:
        parameters = config.get("parameters", [])
        if not parameters:
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak="",
                internal_state=internal_state,
            )

        if not internal_state:
            internal_state = {
                "collected": {},
                "pending_confirmation": None,
                "name_buffer": None,
                "retry_count": 0,
            }

        collected: dict = dict(internal_state.get("collected") or {})
        pending: dict | None = internal_state.get("pending_confirmation")
        name_buffer: dict | None = internal_state.get("name_buffer")

        remaining = [p for p in parameters if p["name"] not in collected]
        if not remaining and not pending:
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak="",
                collected=collected,
                internal_state=internal_state,
            )

        # ── Opening: no utterance yet ─────────────────────────────────────────
        if not utterance:
            if pending:
                # Resuming after an interruption — re-ask the pending confirmation
                speak = self._rephrase_confirmation(pending, parameters)
                return SubagentResponse(
                    status=AgentStatus.WAITING_CONFIRM,
                    speak=speak,
                    pending_confirmation=pending,
                    internal_state=internal_state,
                )
            speak = self._template_question(remaining[0])
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=speak,
                internal_state=internal_state,
            )

        # ── Mega-prompt LLM call ──────────────────────────────────────────────
        persona = config.get("assistant", {}).get("persona_name", "Assistant")
        system = _build_mega_prompt(parameters, collected, pending, persona, name_buffer)

        try:
            result = llm_structured_call(
                system,
                f'Caller said: "{utterance}"',
                DataCollectionLLMResponse,
                max_tokens=800,
            )
        except Exception as exc:
            logger.warning("DataCollection mega-prompt call failed: %s", exc)
            retry = internal_state.get("retry_count", 0) + 1
            internal_state["retry_count"] = retry
            if retry >= self.MAX_RETRIES:
                return SubagentResponse(
                    status=AgentStatus.UNHANDLED,
                    speak="",
                    internal_state={**internal_state, "cannot_process_reason": "llm_parse_failure"},
                    pending_confirmation=pending,
                )
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak="I'm sorry, I didn't quite catch that. Could you please repeat?",
                internal_state=internal_state,
            )

        # ── Handle cannot_process / unhandled ────────────────────────────────
        if result.cannot_process or result.status == "unhandled":
            logger.info(
                "DataCollection: UNHANDLED (reason=%s) utterance=%r",
                result.cannot_process_reason, utterance[:80],
            )
            return SubagentResponse(
                status=AgentStatus.UNHANDLED,
                speak="",
                internal_state={
                    **internal_state,
                    "cannot_process_reason": result.cannot_process_reason,
                },
                pending_confirmation=pending,
            )

        internal_state["retry_count"] = 0

        # ── Step A: Resolve existing pending confirmation ─────────────────────
        if pending:
            field_name = pending["field"]

            if result.intent == "confirm_yes":
                confirmed_value = pending["value"]
                collected[field_name] = confirmed_value
                internal_state["collected"] = collected
                internal_state["pending_confirmation"] = None
                pending = None
                # Clear name_buffer once the full name is confirmed
                if internal_state.get("name_buffer", {}) and \
                        internal_state["name_buffer"].get("field") == field_name:
                    internal_state["name_buffer"] = None
                    name_buffer = None
                logger.info("DataCollection: confirmed %s = %r", field_name, confirmed_value)

            elif result.intent == "confirm_no":
                internal_state["pending_confirmation"] = None
                pending = None
                logger.info("DataCollection: caller rejected value for %s", field_name)

            elif result.intent == "correction":
                correction = result.correction_value
                internal_state["pending_confirmation"] = None
                pending = None
                if correction:
                    param = self._find_param(parameters, field_name)
                    is_valid, msg_or_val = (
                        self._validate(param, correction) if param else (True, correction)
                    )
                    if not is_valid:
                        logger.info(
                            "DataCollection: correction validation failed for %s: %s",
                            field_name, msg_or_val,
                        )
                        return SubagentResponse(
                            status=AgentStatus.IN_PROGRESS,
                            speak=f"{msg_or_val} Could you try again?",
                            internal_state=internal_state,
                        )
                    logger.info(
                        "DataCollection: correction for %s -> %r (validated)",
                        field_name, msg_or_val,
                    )
                    # The corrected value flows through Step C via result.pending_confirmation

        # ── Step B: Handle skipped optional fields ────────────────────────────
        for field_name, value in result.extracted.items():
            if field_name in collected:
                continue
            if value == "":
                param = self._find_param(parameters, field_name)
                if param and not param.get("required", True):
                    collected[field_name] = ""
                    internal_state["collected"] = collected
                    logger.info("DataCollection: optional field %r skipped by caller", field_name)

        # ── Step C: Validate new pending_confirmation from LLM ────────────────
        new_pending: dict | None = None
        validation_error_speak: str | None = None

        if result.pending_confirmation:
            pc = result.pending_confirmation
            if pc.field not in collected:
                param = self._find_param(parameters, pc.field)
                if param:
                    is_valid, msg_or_val = self._validate(param, pc.value)
                    if is_valid:
                        new_pending = {"field": pc.field, "value": msg_or_val}
                    else:
                        validation_error_speak = f"{msg_or_val} Could you try again?"
                        logger.info(
                            "DataCollection: pending validation failed for %s: %s",
                            pc.field, msg_or_val,
                        )
                else:
                    logger.warning(
                        "DataCollection: LLM set pending for unknown field %r — ignoring",
                        pc.field,
                    )

        # ── Step C½: Single-word name buffer ─────────────────────────────────
        # If the LLM set pending_confirmation for a name-type field with only a
        # single word (first name only), buffer it and ask for the last name
        # rather than confirming a possibly incomplete name.
        # Also clear the buffer once we have a full multi-word name.
        if new_pending and not validation_error_speak and name_buffer:
            np_param = self._find_param(parameters, new_pending["field"])
            if np_param and np_param.get("data_type") == "name" and \
                    " " in new_pending["value"].strip():
                # Full name resolved — clear the buffer
                internal_state["name_buffer"] = None
                name_buffer = None

        if new_pending and not validation_error_speak and not name_buffer:
            np_param = self._find_param(parameters, new_pending["field"])
            if np_param and np_param.get("data_type") == "name":
                first_only = new_pending["value"].strip()
                if " " not in first_only:
                    logger.info(
                        "DataCollection: single-word name %r — buffering, asking for last name",
                        first_only,
                    )
                    internal_state["name_buffer"] = {
                        "field": new_pending["field"],
                        "first_name": first_only,
                    }
                    internal_state["pending_confirmation"] = None
                    internal_state["collected"] = collected
                    return SubagentResponse(
                        status=AgentStatus.IN_PROGRESS,
                        speak=f"Thank you, {first_only}. And your last name, please?",
                        internal_state=internal_state,
                    )

        internal_state["pending_confirmation"] = new_pending
        internal_state["collected"] = collected

        # ── Step D: Check completion ──────────────────────────────────────────
        remaining2 = [p for p in parameters if p["name"] not in collected]
        required_remaining = [p for p in remaining2 if p.get("required", True)]

        if not required_remaining and not new_pending:
            speak = result.speak or "Perfect, I have all the information I need."
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak=speak,
                collected=collected,
                internal_state=internal_state,
            )

        # ── Step E: Return ────────────────────────────────────────────────────
        speak = validation_error_speak or result.speak or "Could you please continue?"

        if new_pending and not validation_error_speak:
            status = AgentStatus.WAITING_CONFIRM
        else:
            status = AgentStatus.IN_PROGRESS

        return SubagentResponse(
            status=status,
            speak=speak,
            pending_confirmation=new_pending,
            internal_state=internal_state,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _template_question(self, param: dict) -> str:
        field_type = param.get("data_type", "text")
        if field_type in _OPENING_QUESTIONS:
            return _OPENING_QUESTIONS[field_type]
        label = param["display_label"]
        return f"Could I get your {label}, please?"

    def _rephrase_confirmation(self, pending: dict, parameters: list[dict]) -> str:
        """Generate a re-ask for a pending confirmation after an interruption."""
        field_name = pending["field"]
        value = pending["value"]
        param = self._find_param(parameters, field_name)
        field_type = param.get("data_type", "text") if param else "text"
        label = param["display_label"] if param else field_name
        if field_type == "email":
            readable = value.replace("@", " at ").replace(".", " dot ")
            return f"Just to confirm — your email is {readable}. Is that correct?"
        elif field_type == "phone":
            return f"Just to confirm — your phone number is {value}. Is that correct?"
        elif field_type == "date":
            return f"Just to confirm — the date is {value}. Is that correct?"
        else:
            return f"Just to confirm — your {label} is {value}. Is that correct?"

    def _validate(self, param: dict, value: str) -> tuple[bool, str]:
        if not value and not param.get("required", True):
            return True, ""
        if not value:
            return False, f"I didn't catch your {param['display_label']}."
        pattern = param.get("validation_regex") or _BUILTIN_PATTERNS.get(
            param.get("data_type", "")
        )
        if pattern and not re.match(pattern, value, re.IGNORECASE):
            msg = param.get("validation_message") or (
                f"That doesn't look like a valid {param['display_label']}."
            )
            return False, msg
        return True, value

    @staticmethod
    def _find_param(parameters: list[dict], field_name: str) -> dict | None:
        for p in parameters:
            if p["name"] == field_name:
                return p
        return None
