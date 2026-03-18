"""Data collection agent — gathers and confirms required fields one at a time."""
from __future__ import annotations

import logging
import re
from app.agents.base import AgentBase, AgentStatus, SubagentResponse
from app.agents.llm_utils import llm_json_call, llm_text_call

logger = logging.getLogger(__name__)

# Built-in regex patterns for common data types
_BUILTIN_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
    "phone": r"^[\+]?[\d\s\-\(\)]{7,20}$",
    "date":  r"^\d{4}-\d{2}-\d{2}$",
    "number": r"^\d+(\.\d+)?$",
}

_EXTRACT_SYSTEM = (
    "You are extracting a single field value from a caller's voice utterance. "
    "Return ONLY valid JSON with these fields: "
    "\"found\" (bool), \"value\" (string or null), \"refused\" (bool). "
    "Set refused=true if the caller is explicitly declining to provide the value "
    "(e.g. 'I don't have one', 'skip', 'not applicable', 'I'd rather not say'). "
    "Example: {\"found\": true, \"value\": \"ABC123\", \"refused\": false} "
    "or {\"found\": false, \"value\": null, \"refused\": true}"
)

_CONFIRM_SYSTEM = (
    "You are detecting whether a caller is confirming or rejecting a value. "
    "Return ONLY valid JSON: "
    "{\"intent\": \"yes\"} or {\"intent\": \"no\"} or {\"intent\": \"unclear\", \"correction\": \"<value if embedded>\"}"
)


class DataCollectionAgent(AgentBase):
    """
    Collects required parameters one field at a time.
    Two-step per field: ask → confirm (yes/no).
    Internal state keys:
      stage: "asking" | "waiting_confirm"
      current_field: str (field name)
      pending_value: str
      collected: dict
      retry_count: int
      field_index: int
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

        # Initialise state on first call
        if not internal_state:
            internal_state = {
                "stage": "asking",
                "current_field": None,
                "pending_value": None,
                "collected": {},
                "retry_count": 0,
                "field_index": 0,
            }

        collected = internal_state.get("collected", {})
        stage = internal_state.get("stage", "asking")

        # Find next uncollected field
        remaining = [p for p in parameters if p["name"] not in collected]
        if not remaining:
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak="",
                collected=collected,
                internal_state=internal_state,
            )

        current_param = remaining[0]
        field_name = current_param["name"]
        field_label = current_param["display_label"]
        field_type = current_param.get("data_type", "text")

        internal_state["current_field"] = field_name

        if stage == "asking" and not utterance:
            # Opening — generate first question
            question = self._ask_question(field_label, field_type)
            internal_state["stage"] = "asking"
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=question,
                internal_state=internal_state,
            )

        if stage == "asking":
            # Extract value — also get refused flag in the same LLM call
            extracted, refused = self._extract_value(field_label, field_type, utterance)

            # Optional field refused — skip it and move on
            if refused and not current_param.get("required", True):
                collected[field_name] = ""
                internal_state["collected"] = collected
                internal_state["stage"] = "asking"
                internal_state["retry_count"] = 0
                logger.info("DataCollection: optional field %r skipped by caller refusal", field_name)
                remaining2 = [p for p in parameters if p["name"] not in collected]
                if not remaining2:
                    return SubagentResponse(
                        status=AgentStatus.COMPLETED,
                        speak="No problem. I have all the information I need.",
                        collected=collected,
                        internal_state=internal_state,
                    )
                next_p = remaining2[0]
                q = self._ask_question(next_p["display_label"], next_p.get("data_type", "text"))
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak=f"No problem. {q}",
                    collected=collected,
                    internal_state=internal_state,
                )

            if not extracted:
                retry = internal_state.get("retry_count", 0) + 1
                internal_state["retry_count"] = retry
                if retry >= self.MAX_RETRIES:
                    internal_state["retry_count"] = 0
                    if not current_param.get("required", True):
                        collected[field_name] = ""
                        internal_state["collected"] = collected
                        internal_state["stage"] = "asking"
                        # Advance to next
                        remaining2 = [p for p in parameters if p["name"] not in collected]
                        if not remaining2:
                            return SubagentResponse(
                                status=AgentStatus.COMPLETED,
                                speak="Thank you, I have all the information I need.",
                                collected=collected,
                                internal_state=internal_state,
                            )
                        next_p = remaining2[0]
                        q = self._ask_question(next_p["display_label"], next_p.get("data_type", "text"))
                        return SubagentResponse(
                            status=AgentStatus.IN_PROGRESS,
                            speak=q,
                            internal_state=internal_state,
                        )
                    return SubagentResponse(
                        status=AgentStatus.IN_PROGRESS,
                        speak=f"I'm having difficulty capturing your {field_label}. Could you say it slowly and clearly one more time?",
                        internal_state=internal_state,
                    )
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak=f"Sorry, I didn't catch that. Could you please repeat your {field_label}?",
                    internal_state=internal_state,
                )

            # Validate
            is_valid, msg_or_value = self._validate(current_param, extracted)
            if not is_valid:
                retry = internal_state.get("retry_count", 0) + 1
                internal_state["retry_count"] = retry
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak=f"{msg_or_value} Could you try again?",
                    internal_state=internal_state,
                )

            # Value valid — spell back and wait for confirmation
            normalized = msg_or_value
            internal_state["pending_value"] = normalized
            internal_state["stage"] = "waiting_confirm"
            internal_state["retry_count"] = 0
            confirm_text = self._spell_back(field_label, field_type, normalized)
            return SubagentResponse(
                status=AgentStatus.WAITING_CONFIRM,
                speak=confirm_text,
                pending_confirmation={"field": field_name, "value": normalized},
                internal_state=internal_state,
            )

        if stage == "waiting_confirm":
            pending_value = internal_state.get("pending_value", "")
            intent, correction = self._detect_confirmation(utterance)

            if intent == "yes":
                # Confirmed — record and advance
                collected[field_name] = pending_value
                internal_state["collected"] = collected
                internal_state["pending_value"] = None
                internal_state["stage"] = "asking"
                internal_state["retry_count"] = 0

                remaining2 = [p for p in parameters if p["name"] not in collected]
                if not remaining2:
                    return SubagentResponse(
                        status=AgentStatus.COMPLETED,
                        speak="Perfect, thank you. I have all the information I need.",
                        collected=collected,
                        internal_state=internal_state,
                    )
                next_p = remaining2[0]
                q = self._ask_question(next_p["display_label"], next_p.get("data_type", "text"))
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak=q,
                    collected=collected,
                    internal_state=internal_state,
                )

            if intent == "no":
                # Rejected — check for embedded correction
                if correction:
                    is_valid, msg_or_value = self._validate(current_param, correction)
                    if is_valid:
                        internal_state["pending_value"] = msg_or_value
                        confirm_text = self._spell_back(field_label, field_type, msg_or_value)
                        return SubagentResponse(
                            status=AgentStatus.WAITING_CONFIRM,
                            speak=confirm_text,
                            pending_confirmation={"field": field_name, "value": msg_or_value},
                            internal_state=internal_state,
                        )
                internal_state["stage"] = "asking"
                internal_state["pending_value"] = None
                return SubagentResponse(
                    status=AgentStatus.IN_PROGRESS,
                    speak=f"No problem. Could you please say your {field_label} again?",
                    internal_state=internal_state,
                )

            # Unclear
            return SubagentResponse(
                status=AgentStatus.WAITING_CONFIRM,
                speak=f"Sorry, I didn't catch that. Is your {field_label} {pending_value}? Please say yes or no.",
                pending_confirmation={"field": field_name, "value": pending_value},
                internal_state=internal_state,
            )

        # Fallback — re-ask current field
        q = self._ask_question(field_label, field_type)
        internal_state["stage"] = "asking"
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=q,
            internal_state=internal_state,
        )

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _ask_question(self, field_label: str, field_type: str) -> str:
        type_hints = {
            "email": "Ask them to spell it out if needed.",
            "phone": "Remind them to include the area code.",
            "date": "Accept natural language like 'January fifteenth'.",
            "name": "Ask for their first and last name. They can take a brief pause between first and last name.",
        }
        hint = type_hints.get(field_type, "")
        return llm_text_call(
            "You generate short, friendly voice questions for an AI receptionist. One sentence, no preamble.",
            f"Generate a voice question asking the caller for their {field_label}. {hint}",
        )

    def _extract_value(self, field_label: str, field_type: str, utterance: str) -> tuple[str | None, bool]:
        """Returns (extracted_value_or_None, refused)."""
        type_guidance = {
            "email": "Include the full address with @ symbol and domain.",
            "phone": "Digits only, preserve area code.",
            "date": "Normalise to YYYY-MM-DD.",
            "number": "Return only the numeric value.",
            "name": "Capture full name — first and last.",
        }
        guidance = type_guidance.get(field_type, "")
        try:
            result = llm_json_call(
                _EXTRACT_SYSTEM,
                f"Field: {field_label} (type: {field_type})\n{guidance}\nCaller said: \"{utterance}\"",
            )
            refused = bool(result.get("refused", False))
            if result.get("found") and result.get("value"):
                return str(result["value"]).strip(), False
            return None, refused
        except Exception as exc:
            logger.warning("DataCollection extract_value failed: %s", exc)
        return None, False

    def _validate(self, param: dict, value: str) -> tuple[bool, str]:
        pattern = param.get("validation_regex") or _BUILTIN_PATTERNS.get(param.get("data_type", ""))
        if pattern and not re.match(pattern, value, re.IGNORECASE):
            msg = param.get("validation_message") or f"That doesn't look like a valid {param['display_label']}."
            return False, msg
        return True, value

    def _spell_back(self, field_label: str, field_type: str, value: str) -> str:
        if field_type == "email":
            prompt = f"Read back this email for voice confirmation: {value}. Say the email naturally (dot, at). End with 'Is that correct?'"
        elif field_type == "phone":
            prompt = f"Read back this phone number clearly, grouped in natural speech: {value}. End with 'Is that correct?'"
        else:
            prompt = f"Read back this {field_label} for confirmation: {value}. One sentence. End with 'Is that correct?'"
        return llm_text_call(
            "You generate short voice phrases for an AI receptionist. One sentence only.",
            prompt,
        )

    def _detect_confirmation(self, utterance: str) -> tuple[str, str | None]:
        """Returns ('yes'|'no'|'unclear', correction_value_or_None)."""
        try:
            result = llm_json_call(
                _CONFIRM_SYSTEM,
                f"Caller said: \"{utterance}\"",
            )
            intent = result.get("intent", "unclear")
            correction = result.get("correction") if intent == "unclear" else None
            if intent == "no":
                correction = result.get("correction")
            return intent, correction
        except Exception as exc:
            logger.warning("DataCollection detect_confirmation failed: %s", exc)
            return "unclear", None
