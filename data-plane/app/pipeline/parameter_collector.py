"""Parameter collection, extraction, and validation logic."""
import re
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Built-in regex patterns for common data types
BUILTIN_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
    "phone": r"^[\+]?[\d\s\-\(\)]{7,20}$",
    "date": r"^\d{4}-\d{2}-\d{2}$",
    "number": r"^\d+(\.\d+)?$",
}


@dataclass
class ParameterSpec:
    name: str
    display_label: str
    data_type: str
    required: bool
    collection_order: int
    validation_regex: str | None
    validation_message: str | None
    extraction_hints: list[str]


@dataclass
class CollectionState:
    parameters: list[ParameterSpec]
    collected: dict[str, str] = field(default_factory=dict)  # name -> normalized_value
    failed_validations: dict[str, int] = field(default_factory=dict)  # name -> fail count
    pending_validation_note: str | None = None

    @property
    def required_params(self) -> list[ParameterSpec]:
        return [p for p in self.parameters if p.required]

    @property
    def all_required_collected(self) -> bool:
        return all(p.name in self.collected for p in self.required_params)

    @property
    def next_uncollected(self) -> ParameterSpec | None:
        """Return the next parameter to collect, required or optional, in collection_order."""
        for p in sorted(self.parameters, key=lambda x: x.collection_order):
            if p.name not in self.collected:
                return p
        return None

    def validate_value(self, param: ParameterSpec, value: str) -> tuple[bool, str]:
        """Returns (is_valid, normalized_value)."""
        value = value.strip()

        # Use explicit regex if provided
        pattern = param.validation_regex
        if not pattern:
            pattern = BUILTIN_PATTERNS.get(param.data_type)

        if pattern:
            if not re.match(pattern, value, re.IGNORECASE):
                msg = param.validation_message or f"'{value}' doesn't appear to be a valid {param.display_label}"
                return False, msg

        return True, value

    def record_collected(self, name: str, raw: str, normalized: str) -> None:
        self.collected[name] = normalized
        self.failed_validations.pop(name, None)
        logger.info("Parameter '%s' collected: %s", name, normalized)

    def record_failed(self, name: str, message: str) -> None:
        self.failed_validations[name] = self.failed_validations.get(name, 0) + 1
        self.pending_validation_note = message
        logger.info("Parameter '%s' failed validation (attempt %d)", name, self.failed_validations[name])


def load_parameters(config: dict) -> CollectionState:
    """Build CollectionState from control plane config payload."""
    specs = [
        ParameterSpec(
            name=p["name"],
            display_label=p["display_label"],
            data_type=p["data_type"],
            required=p.get("required", True),
            collection_order=p.get("collection_order", 0),
            validation_regex=p.get("validation_regex"),
            validation_message=p.get("validation_message"),
            extraction_hints=p.get("extraction_hints", []),
        )
        for p in config.get("parameters", [])
    ]
    return CollectionState(parameters=specs)


def try_extract_parameter(
    state: CollectionState,
    llm_text: str,
    caller_text: str,
) -> tuple[str | None, str | None, str | None]:
    """
    Attempt to extract a parameter value from the conversation.

    Uses simple heuristics: if the LLM response contains a confirmation phrase,
    and we're waiting on a specific parameter, try to extract from caller's last utterance.

    Returns (param_name, raw_value, normalized_value) or (None, None, None).
    """
    next_param = state.next_uncollected
    if not next_param:
        return None, None, None

    # Check if LLM response suggests it received an answer (confirmation patterns)
    confirmation_signals = [
        "got it", "thank you", "perfect", "great", "i have your",
        "noted", "recorded", "i've noted", "confirmed", "i see that",
    ]
    llm_lower = llm_text.lower()
    has_confirmation = any(sig in llm_lower for sig in confirmation_signals)

    if not has_confirmation:
        return None, None, None

    # Extract the value from caller's utterance using type-aware parsing
    raw_value = _extract_value_by_type(next_param, caller_text)
    if not raw_value:
        return None, None, None

    is_valid, normalized = state.validate_value(next_param, raw_value)
    if is_valid:
        return next_param.name, raw_value, normalized
    else:
        state.record_failed(next_param.name, normalized)
        return None, None, None


def _extract_value_by_type(param: ParameterSpec, text: str) -> str | None:
    """Type-aware value extraction from caller text."""
    text = text.strip()
    if not text:
        return None

    if param.data_type == "email":
        # Look for email pattern in text
        match = re.search(r"[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", text)
        return match.group(0) if match else text

    if param.data_type == "phone":
        # Extract digit sequence
        digits = re.sub(r"[^\d+]", "", text)
        return digits if len(digits) >= 7 else text

    if param.data_type == "date":
        # Try to find date patterns
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        return match.group(0) if match else text

    if param.data_type == "number":
        match = re.search(r"\d+(\.\d+)?", text)
        return match.group(0) if match else text

    # Default: return full text
    return text
