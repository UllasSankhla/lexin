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
from app.agents.data_collection_schema import ConfirmationSignal, DataCollectionLLMResponse
from app.agents.llm_utils import llm_structured_call, ConversationHistory

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
        "Confirm spelling if the name sounds hyphenated, unusual, compound, or foreign.",
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

# ── Typed field shape enforcement ─────────────────────────────────────────────
# Applied after LLM extraction to reject structurally impossible type assignments
# (e.g. "VW3R9KJ2" assigned to a "state" field). The regex _validate() handles
# format correctness; this guard handles semantic type mismatches.

_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
    "district of columbia", "washington dc", "dc",
}
_US_STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
}
_OPAQUE_ID_RE = re.compile(r'^[A-Z0-9]{4,}$', re.IGNORECASE)


def _shape_matches_type(value: str, data_type: str) -> bool:
    """
    Returns True if value's shape is plausible for data_type.
    Returns False only for structurally impossible matches to prevent
    the LLM from assigning values (like MetLife IDs) to wrong field types.
    """
    v = value.strip()
    if not v:
        return True

    if data_type in ("state", "us_state"):
        norm = v.lower()
        if norm in _US_STATE_NAMES or norm in _US_STATE_CODES:
            return True
        # Reject alphanumeric IDs and pure numbers as state values
        if _OPAQUE_ID_RE.match(v) and len(v) > 2:
            return False
        if re.match(r'^\d', v):
            return False
        return True  # uncertain — let _validate() handle

    if data_type == "email":
        if "@" not in v and " at " not in v.lower():
            return False
        return True

    if data_type == "phone":
        digit_count = sum(c.isdigit() for c in v)
        return digit_count >= 7

    if data_type == "number":
        return bool(re.match(r'^\d+(\.\d+)?$', v))

    # name, text, id, date — permissive; _validate handles format
    return True


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


def _build_intake_flow_block(parameters: list[dict]) -> str:
    """Build a compact ordered summary of fields to collect."""
    if not parameters:
        return ""
    sorted_params = sorted(parameters, key=lambda x: x.get("collection_order", 999))
    lines = [
        "Collect the following fields in this exact order.",
        "Confirm each one before moving to the next. Required fields cannot be skipped.",
    ]
    for i, p in enumerate(sorted_params, 1):
        req = "required" if p.get("required", True) else "optional — skip if caller declines"
        lines.append(f"  {i}. {p['display_label']} ({req})")
    return "\n".join(lines)


def _build_policy_docs_block(policy_docs: list[dict]) -> str:
    """Concatenate global policy document contents for injection into the prompt."""
    if not policy_docs:
        return ""
    parts = []
    for doc in policy_docs:
        content = doc.get("content", "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _build_mega_prompt(
    parameters: list[dict],
    collected: dict,
    pending: dict | None,
    persona_name: str,
    workflow_stages: str = "",
    call_transcript: str = "",
    policy_docs: list[dict] | None = None,
) -> str:
    fields_block = _build_fields_block(parameters)
    intake_flow = _build_intake_flow_block(parameters)
    collected_json = json.dumps(collected, indent=2) if collected else "{}"
    pending_json = json.dumps(pending, indent=2) if pending else "none"
    stages_block = (
        f"\n{workflow_stages}\n"
        if workflow_stages else ""
    )
    intake_flow_block = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"INTAKE COLLECTION ORDER\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{intake_flow}\n"
        if intake_flow else ""
    )
    policy_content = _build_policy_docs_block(policy_docs or [])
    policy_block = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"FIRM INTAKE PROCEDURE (follow exactly)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{policy_content}\n"
        if policy_content else ""
    )
    transcript_block = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"CALL TRANSCRIPT SO FAR\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{call_transcript}\n"
        if call_transcript else ""
    )

    return f"""\
You are an AI intake receptionist conducting a voice call on behalf of {persona_name}.
Your role right now is to collect specific information from the caller.
Be warm, patient, and concise — this is a voice call, not a form.
{stages_block}{intake_flow_block}{policy_block}{transcript_block}

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

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
EXTRACTION RULES
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
1. OUT-OF-ORDER: Extract any field the caller mentions, even if it is not the
   field currently being asked for or pending confirmation.
   - If the caller provides field 3 while field 1 is pending confirmation:
     extract field 3 into the extracted list, keep the pending confirmation, and
     re-ask for the pending confirmation in the same response.
   - If the caller provides a value for the field that IS pending confirmation:
     treat it as a correction (intent=correction) — update the pending value and
     ask the caller to confirm the new value.
   Never discard a field value just because it arrived out of order.

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
   A correction utterance may also contain values for OTHER fields — apply
   Rule 2 (multiple fields at once) even during a correction. Put the
   corrected field in pending_confirmation; put other extracted values in
   the extracted list.

   PARTIAL CORRECTION — when the caller corrects only PART of a field value
   (e.g. pending phone is "two zero six seven seven nine one four six" and
   caller says "No, it's four one four six" meaning only the last 4 digits
   differ), use Rule 5B PARTIAL DATA RECONSTRUCTION to combine the corrected
   segment with the confirmed parts from transcript history. For phone:
   - Use area code and middle digits from the prior read-back in the transcript.
   - Replace only the digits the caller specified as wrong.
   - Put the fully reconstructed 10-digit number in pending_confirmation.
   Never store a partial segment (e.g. "4146") as the field value — always
   assemble the complete value before setting pending_confirmation.

5. INCOMPLETE UTTERANCE AND PARTIAL DATA RECONSTRUCTION:

   A. SENTENCE CONTINUATION: If the caller's speech seems cut off mid-sentence
      (e.g. ends on "my email is..." with no address following), do not guess.
      Set incomplete_utterance=true and ask them to continue.
      Also check the CALL TRANSCRIPT for the rest of a sentence the caller
      started in a prior turn (e.g. they said "my email is" last turn and
      "john at gmail dot com" this turn — combine them).

   B. PARTIAL VALUE RECONSTRUCTION — CRITICAL: When the current utterance
      contains only part of a field value (e.g. only a domain fragment like
      "at gmail dot com" with no local part, or only digits without an area
      code), do NOT immediately ask for it again. Instead:

      1. Scan the last 5 caller utterances in the CALL TRANSCRIPT above.
      2. Look for any earlier utterance that could supply the missing component
         for the same field:
         - Email: look for a local-part (letters/digits before "@"), a spelled
           name, or letter-by-letter spelling in prior turns.
         - Phone: look for digit groups that could be an area code or suffix
           provided in an earlier turn.
         - Name: look for a first or last name mentioned separately in prior turns.
      3. If the missing component is found in history, combine the pieces and
         treat the assembled result as the field value. Proceed with read-back
         and confirmation of the reconstructed value.
      4. Only ask the caller to repeat if the missing component is genuinely
         absent from the last 5 utterances.

      Examples:
      - Caller said "john dot doe" two turns ago, now says "at gmail dot com"
        → reconstruct email as john.doe@gmail.com, read it back to confirm.
      - Caller said "four one five" one turn ago, now says "five five five zero one
        nine two" → reconstruct phone as 4155550192, read it back to confirm.
      - Caller said "sarah" three turns ago, now says "mitchell" → reconstruct
        name as "Sarah Mitchell", read it back to confirm.

6. SKIP (optional fields only): If the caller says "skip", "I don't have one",
   "not applicable", or "I'd rather not say" record the field as "" in extracted.

7. TYPE SHAPE RULE — never assign a value to a field if its shape is structurally
   wrong for the field's declared type:
   - Fields with type "state" or "us_state": ONLY accept US state names ("California",
     "Washington") or 2-letter codes ("CA", "WA"). NEVER assign alphanumeric IDs,
     numbers, or non-state words. Example: if caller says "VW3R9KJ2" while the
     current field is "state_of_matter", do NOT extract it — they likely gave an
     unrelated ID (e.g. a MetLife member ID), not a state name.
   - Fields with type "email": value MUST contain "@" or spoken "at" between parts.
   - Fields with type "phone": value MUST be primarily digits (at least 7 digits total).
   - Fields with type "number": value MUST be a numeric value only.
   When in doubt about which field a value belongs to, DO NOT assign it to a field
   whose type clearly doesn't match. Leave it unextracted.

8. STRICT NO-FABRICATION RULE — applies to every field, no exceptions:
   NEVER guess, infer, construct, or fabricate any value. Extract only what the
   caller explicitly stated in this utterance or in the CALL TRANSCRIPT.

   NAMES: If the name is garbled, phonetically ambiguous, or you heard only
   part of it — set incomplete_utterance=true and ask the caller to spell it.
   Do NOT construct a name from sounds (e.g. hearing "Prastina" and writing
   "Christina"). The only acceptable source is the caller's own words or spelling.

   PHONE NUMBERS: Extract only digits the caller clearly stated. If any digit
   group is missing or unclear, ask for it. Never fill in area codes or
   missing digits from context.

   ADDRESSES: Extract only the components the caller explicitly gave. Do not
   infer city from state, state from zip code, or fill in any part of an address
   that was not stated. If a component is missing, ask for it.

   EMAIL: Transcribe exactly what was spelled or spoken. If ambiguous, ask the
   caller to spell the local part character by character.

   DATES: Only use what the caller provided. Do not assume the year.

   If in doubt — ask. Silence is better than a wrong value.

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

When presenting a value for confirmation, read it back plainly and ask
"Is that correct?" — no "Got it" or acknowledgment before the question,
since you haven't confirmed anything yet.

  GOOD: "I have Sarah Mitchell — is that correct?"
  GOOD: "Just to confirm, your email is sarah dot mitchell at gmail dot com. Is that correct?"
  BAD:  "Got it, Sarah Mitchell. Is that correct?" — contradicts itself

SINGLE QUESTION RULE — CRITICAL:
When setting pending_confirmation (value awaiting yes/no), your speak MUST
contain ONLY the read-back question for that pending value.
Do NOT add a next-field question, a MetLife question, or any other request in
the same speak. One question at a time. The next field is asked in the turn
AFTER the confirmation is received.

  GOOD: "I have your phone number as two zero six, seven seven nine, four one
         four six — is that correct?"
  BAD:  "I have your phone number as ... — is that correct? Also, do you have
         a MetLife ID?" ← two questions; the caller's yes/no becomes ambiguous

After a confirmation is resolved (yes/no/correction), if there are still
uncollected fields, your speak MUST seamlessly continue to the next field
in the same sentence. Never end a turn with a bare acknowledgment when
work remains.

  GOOD: "Great, and what's the best number to reach you?"
  BAD:  "Thank you." — leaves the caller with nothing to do

The speak for a resolved confirmation is always:
  <brief acknowledgment> + <question for next uncollected field>

CALLER UTTERANCE TAKES PRIORITY — always read the full utterance and extract
any concrete field values present BEFORE deciding whether it is a yes/no response.

If the utterance contains concrete field data (phone digits, an email address,
a name, a date, an address, etc.) — regardless of pending_confirmation state:
  - Do NOT interpret it as "yes" or "no" — it is new information
  - A string of spoken digits is NEVER a confirmation signal ("Two zero two..." = phone, not "yes")
  - If the value matches the pending field → intent=correction; update
    pending_confirmation with the new value; speak the read-back and ask to confirm
  - If the value matches a DIFFERENT field → extract it (Rule 1 out-of-order),
    queue it, and re-ask for the pending field in the same response
  - Never discard field data because a confirmation was outstanding

Only interpret as yes/no when the utterance contains NO extractable field values:
- YES signals: "yes", "yep", "correct", "that's right", "uh huh", "sure",
               "sounds good", "affirmative", "go ahead"
- NO signals:  "no", "nope", "wrong", "that's not it", "incorrect"
- CORRECTION (explicit verbal): "actually...", "no it's...", "make that...", "wait..."
               Set intent=correction, put new value in correction_value,
               and put the corrected value in pending_confirmation with the
               same field key as the current pending confirmation.
- Ambiguous:   Set intent=answer and ask explicitly in speak:
               "Is [value] correct? Please say yes or no."

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
CANNOT-PROCESS CONDITIONS
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
INFORMATION REQUEST RULE — handle BEFORE considering cannot_process:
If the caller asks any of the following, do NOT set cannot_process=true:
  a) "What information do you need?", "what do you need from me?", "what are you
     collecting?", "what details do you require?" — questions about the fields
     being gathered right now.
     → Name the remaining uncollected required fields in plain conversational
       language, then pivot back to the current field question in the same sentence.
     Example (phone and email remain):
       speak = "I still need your phone number and email address. Let's continue —
                what's the best number to reach you, including the area code?"

  b) "What do I need to book an appointment?", "what are the steps?", "what
     happens after this?", "what information is needed before we can schedule?" —
     questions about the overall booking process or requirements.
     → Use the BOOKING STAGES block shown above (if present) to explain all
       remaining stages in plain conversational language, then pivot back to the
       current field question.
     Example:
       speak = "To book an appointment I'll need to collect your contact details
                first — which is what we're doing now — then I'll ask you to
                describe your matter, and after that we'll check whether it falls
                within our practice areas before scheduling a slot. Let's continue:
                what's the best number to reach you?"

  Set status="in_progress" for both cases.

Set cannot_process=true and status="unhandled" when:
- The utterance is a question unrelated to providing field values AND unrelated
  to what information is being collected (e.g. "How long will this take?", "What are your fees?")
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
  "extracted": [{{"key": "<field_key>", "value": "<normalized_value>"}}],
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


_CONFIRM_CLASSIFIER_SYSTEM = """\
The AI is waiting for a caller to confirm or reject a specific value.
Classify the caller's response into exactly one of four categories:

"confirm"        — pure yes/agreement, no new field data
                   (e.g. "yes", "yeah", "correct", "that's right", "sounds good")

"reject"         — pure no/disagreement, no new field data
                   (e.g. "no", "nope", "wrong", "that's not right")

"correct_or_add" — contains new field data (phone digits, email, name, date, etc.)
                   regardless of whether the caller also agrees or disagrees
                   (e.g. "no it's 415-555-1234", "yes and my email is...", "four one five...")

"unrelated"      — a question, off-topic remark, or anything that is NOT a yes/no/correction
                   (e.g. "how long does this take?", "do you need my address?", "what is this for?")

Also set is_affirmative: true if the caller's tone is confirming/agreeing, false if rejecting/correcting.
Only relevant for "correct_or_add" to help the next processing step.

Respond with JSON only: {"signal": "confirm"|"reject"|"correct_or_add"|"unrelated", "is_affirmative": bool}
"""


def _format_call_transcript(history: list[dict]) -> str:
    """Format the call history into a readable transcript string."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not content:
            continue
        label = "Caller" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


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
      extraction_queue:     list[dict]      — extracted values awaiting confirmation
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
                confidence=1.0,
            )

        if not internal_state:
            internal_state = {
                "collected": {},
                "pending_confirmation": None,
                "extraction_queue": [],
                "retry_count": 0,
            }

        collected: dict = dict(internal_state.get("collected") or {})
        pending: dict | None = internal_state.get("pending_confirmation")

        remaining = [p for p in parameters if p["name"] not in collected]
        if not remaining and not pending:
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak="",
                collected=collected,
                internal_state=internal_state,
                confidence=1.0,
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
                    confidence=1.0,
                )
            speak = self._template_question(remaining[0])
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak=speak,
                internal_state=internal_state,
                confidence=0.9,
            )

        # ── Confirmation classifier (when a value is awaiting yes/no) ───────────
        # A dedicated two-boolean LLM call determines whether the utterance is
        # purely a yes/no control signal or contains new field data.
        #
        # Pure yes/no (has_new_data=false):  handled deterministically — no
        #   mega-prompt call needed, no risk of the model returning intent=answer.
        #
        # Contains new data (has_new_data=true): fall through to the mega-prompt
        #   with a hint so it knows to confirm/reject AND extract the new data.
        mega_prompt_hint: str = ""
        if pending:
            signal = self._classify_confirmation_signal(utterance, pending, parameters)
            logger.debug(
                "DC confirm-classifier: signal=%s is_affirmative=%s",
                signal.signal, signal.is_affirmative,
            )
            if signal.signal == "confirm":
                return self._apply_confirm_yes(
                    utterance, pending, collected, internal_state, parameters, config
                )
            if signal.signal == "reject":
                return self._apply_confirm_no(
                    utterance, pending, collected, internal_state, parameters
                )
            if signal.signal == "correct_or_add":
                # Let mega-prompt handle extraction; pass tone hint
                hint_word = "CONFIRMED" if signal.is_affirmative else "REJECTED"
                mega_prompt_hint = (
                    f"[Pre-classified: caller {hint_word} the pending value and "
                    f"also provided new information. Handle both.] "
                )
            # signal == "unrelated": fall through to mega-prompt with no hint
            # (mega-prompt will return UNHANDLED for off-topic questions)

        # ── Mega-prompt LLM call ──────────────────────────────────────────────
        persona = config.get("assistant", {}).get("persona_name", "Assistant")
        workflow_stages = config.get("_workflow_stages", "")
        call_transcript = _format_call_transcript(history)
        policy_docs = config.get("global_policy_documents", [])
        system = _build_mega_prompt(parameters, collected, pending, persona, workflow_stages, call_transcript, policy_docs)
        llm_history = ConversationHistory.from_list(internal_state.get("llm_history"))
        user_msg = f'{mega_prompt_hint}Caller said: "{utterance}"'

        try:
            result = llm_structured_call(
                system,
                user_msg,
                DataCollectionLLMResponse,
                max_tokens=2048,
                history=llm_history,
            )
            logger.debug(
                "DC LLM: utt=%r intent=%s status=%s pending=%s extracted=%s",
                utterance[:40], result.intent, result.status,
                result.pending_confirmation, result.extracted,
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
                    confidence=0.0,
                )
            return SubagentResponse(
                status=AgentStatus.IN_PROGRESS,
                speak="I'm sorry, I didn't quite catch that. Could you please repeat?",
                internal_state=internal_state,
                confidence=0.3,
            )

        # ── Update conversation history ───────────────────────────────────────
        # Cap at last 8 entries (4 agent turns) — the full call transcript is
        # already in the system prompt so the LLM has complete context there.
        # Unbounded growth wastes tokens and can cause stale-context confusion.
        llm_history.add("user", user_msg)
        llm_history.add("assistant", result.speak or "")
        history_list = llm_history.to_list()
        if len(history_list) > 8:
            history_list = history_list[-8:]
        internal_state["llm_history"] = history_list

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
                confidence=0.0,
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
                logger.info("DataCollection: confirmed %s = %r", field_name, confirmed_value)

            elif result.intent == "confirm_no":
                internal_state["pending_confirmation"] = None
                pending = None
                logger.info("DataCollection: caller rejected value for %s", field_name)

            elif result.intent == "correction":
                correction = result.correction_value
                if correction:
                    param = self._find_param(parameters, field_name)
                    is_valid, msg_or_val = (
                        self._validate(param, correction) if param else (True, correction)
                    )
                    if not is_valid:
                        logger.info(
                            "DataCollection: correction validation failed for %s: %s — "
                            "preserving original pending_confirmation",
                            field_name, msg_or_val,
                        )
                        # Do NOT clear pending_confirmation — the original value is still
                        # valid and must not be lost because a bad correction was attempted.
                        return SubagentResponse(
                            status=AgentStatus.WAITING_CONFIRM,
                            speak=f"{msg_or_val} Could you try again?",
                            internal_state=internal_state,
                            pending_confirmation=pending,
                            confidence=0.7,
                        )
                    logger.info(
                        "DataCollection: correction for %s -> %r (validated)",
                        field_name, msg_or_val,
                    )
                    # Correction is valid — now clear the old pending so the corrected
                    # value can flow through Step C via result.pending_confirmation.
                    internal_state["pending_confirmation"] = None
                    pending = None
                else:
                    # No correction value supplied — clear pending and re-ask normally
                    internal_state["pending_confirmation"] = None
                    pending = None

        # ── Step B: Handle extracted values ───────────────────────────────────
        # Non-empty values that aren't the immediate pending_confirmation are
        # queued so they survive across turns (multi-field utterance support).
        pending_field_from_llm = (
            result.pending_confirmation.field if result.pending_confirmation else None
        )
        queue: list[dict] = list(internal_state.get("extraction_queue") or [])
        queued_fields = {q["field"] for q in queue}

        for field_name, value in ((ef.key, ef.value) for ef in result.extracted):
            if field_name in collected:
                continue
            if value == "":
                # Skip — record empty for optional fields
                param = self._find_param(parameters, field_name)
                if param and not param.get("required", True):
                    collected[field_name] = ""
                    internal_state["collected"] = collected
                    logger.info("DataCollection: optional field %r skipped by caller", field_name)
            elif field_name != pending_field_from_llm and field_name not in queued_fields:
                # Shape-match guard: reject structurally impossible type assignments
                param = self._find_param(parameters, field_name)
                if param and not _shape_matches_type(value, param.get("data_type", "text")):
                    logger.info(
                        "DataCollection: shape mismatch — rejecting %r for field %r (type=%s)",
                        value, field_name, param.get("data_type"),
                    )
                    continue
                queue.append({"field": field_name, "value": value})
                queued_fields.add(field_name)
                logger.info("DataCollection: queued extracted field %r = %r", field_name, value)

        internal_state["extraction_queue"] = queue

        # ── Step C: Validate new pending_confirmation from LLM ────────────────
        new_pending: dict | None = None
        validation_error_speak: str | None = None

        if result.pending_confirmation:
            pc = result.pending_confirmation
            # Allow re-confirmation when correcting an already-confirmed field
            is_correction_of_confirmed = (
                result.intent == "correction" and pc.field in collected
            )
            if pc.field not in collected or is_correction_of_confirmed:
                param = self._find_param(parameters, pc.field)
                if param:
                    # Shape guard: reject structurally impossible type assignments.
                    # On mismatch, silently drop the bad pending (the field will be
                    # asked normally in a future turn).  Only surface an error to the
                    # caller when they explicitly provided a correction and the value
                    # still fails shape — that means the caller typed the wrong thing.
                    if not _shape_matches_type(pc.value, param.get("data_type", "text")):
                        logger.info(
                            "DataCollection: shape mismatch in pending_confirmation — "
                            "field=%r value=%r type=%s — dropping silently",
                            pc.field, pc.value, param.get("data_type"),
                        )
                        if result.intent == "correction":
                            validation_error_speak = (
                                f"I'm sorry, that doesn't seem right for {param['display_label']}. "
                                "Could you try again?"
                            )
                        # else: LLM extraction error — drop without telling the caller
                    else:
                        # Skip validation entirely if the LLM gave an empty value —
                        # that's an extraction miss, not a user error.  Let Step C¾
                        # promote from the queue or the normal question flow ask again.
                        if not pc.value or not pc.value.strip():
                            logger.info(
                                "DataCollection: empty pending_confirmation for %r — skipping silently",
                                pc.field,
                            )
                        else:
                            is_valid, msg_or_val = self._validate(param, pc.value)
                            if is_valid:
                                new_pending = {"field": pc.field, "value": msg_or_val}
                                if is_correction_of_confirmed:
                                    del collected[pc.field]
                                    internal_state["collected"] = collected
                                    logger.info(
                                        "DataCollection: re-opening confirmed field %r for correction → %r",
                                        pc.field, msg_or_val,
                                    )
                            else:
                                # Only show validation errors for user corrections,
                                # not for LLM extraction failures on regular questions.
                                if result.intent == "correction":
                                    validation_error_speak = f"{msg_or_val} Could you try again?"
                                else:
                                    logger.info(
                                        "DataCollection: pending validation failed for %s: %s — dropping",
                                        pc.field, msg_or_val,
                                    )
                else:
                    logger.warning(
                        "DataCollection: LLM set pending for unknown field %r — ignoring",
                        pc.field,
                    )

        # ── Step C¾: Pop from extraction queue if no pending ─────────────────
        # When the LLM returns no new pending_confirmation (e.g. after a "yes"
        # confirmation), promote the next queued extracted value so the caller
        # doesn't have to repeat information they already provided.
        queue_speak: str | None = None
        if not new_pending and not validation_error_speak:
            queue = list(internal_state.get("extraction_queue") or [])
            while queue:
                queued = queue.pop(0)
                fn = queued["field"]
                if fn in collected:
                    continue  # already confirmed since it was queued
                param = self._find_param(parameters, fn)
                if not param:
                    continue
                is_valid, msg_or_val = self._validate(param, queued["value"])
                if is_valid:
                    new_pending = {"field": fn, "value": msg_or_val}
                    queue_speak = self._rephrase_confirmation(new_pending, parameters)
                    logger.info(
                        "DataCollection: dequeued field %r = %r for confirmation",
                        fn, msg_or_val,
                    )
                    break
                else:
                    logger.info(
                        "DataCollection: dropped invalid queued value for %r — will re-ask",
                        fn,
                    )
            internal_state["extraction_queue"] = queue

        # ── Step C+: Re-surface unresolved pending confirmation ───────────────
        # If a pending confirmation existed at entry but the LLM's intent was
        # not confirm_yes / confirm_no / correction (e.g. mis-classified as
        # "answer"), AND neither Step C nor C¾ produced a new pending, the old
        # pending would be silently erased at line 630 — the field is never
        # added to collected and the agent re-asks for it indefinitely.
        # Fix: preserve and re-surface the original pending so the caller is
        # asked again rather than the confirmation being dropped.
        if (
            pending
            and not new_pending
            and not validation_error_speak
            and result.intent not in ("confirm_yes", "confirm_no", "correction")
        ):
            new_pending = pending
            queue_speak = self._rephrase_confirmation(new_pending, parameters)
            logger.info(
                "DataCollection: pending for %r not resolved by intent=%r — re-surfacing",
                pending["field"], result.intent,
            )

        internal_state["pending_confirmation"] = new_pending
        internal_state["collected"] = collected

        # ── Step D: Check completion ──────────────────────────────────────────
        remaining_all = [p for p in parameters if p["name"] not in collected]
        required_remaining = [p for p in remaining_all if p.get("required", True)]

        if not remaining_all and not new_pending:
            speak = result.speak or "Perfect, I have all the information I need."
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak=speak,
                collected=collected,
                internal_state=internal_state,
                confidence=1.0,
            )

        # ── Step E: Return ────────────────────────────────────────────────────
        # queue_speak takes priority when we dequeued a value — it's a precise
        # readback. validation_error_speak takes priority over everything else.
        speak = validation_error_speak or queue_speak or result.speak or "Could you please continue?"

        if new_pending and not validation_error_speak:
            status = AgentStatus.WAITING_CONFIRM
            # High confidence: agent cleanly extracted a field and is confirming it
            conf = 0.9
        else:
            status = AgentStatus.IN_PROGRESS
            # Confidence depends on how cleanly the agent handled the utterance
            if result.intent in ("confirm_yes", "confirm_no", "correction"):
                conf = 0.9  # clean confirmation/correction handling
            elif result.intent == "answer":
                conf = 0.75  # agent processed an answer but no field confirmed
            else:
                conf = 0.6  # ambiguous or general progress

        return SubagentResponse(
            status=status,
            speak=speak,
            pending_confirmation=new_pending,
            internal_state=internal_state,
            confidence=conf,
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

    def _record_history(self, utterance: str, speak: str, internal_state: dict) -> None:
        """Add a classifier fast-path turn to llm_history so future mega-prompt calls have context."""
        history = ConversationHistory.from_list(internal_state.get("llm_history"))
        history.add("user", f'Caller said: "{utterance}"')
        history.add("assistant", speak)
        history_list = history.to_list()
        if len(history_list) > 8:
            history_list = history_list[-8:]
        internal_state["llm_history"] = history_list

    @staticmethod
    def _find_param(parameters: list[dict], field_name: str) -> dict | None:
        for p in parameters:
            if p["name"] == field_name:
                return p
        return None

    def _classify_confirmation_signal(
        self,
        utterance: str,
        pending: dict,
        parameters: list[dict],
    ) -> ConfirmationSignal:
        """
        Tiny focused LLM call: classify whether the utterance is a pure yes/no
        control signal or contains new field data (and whether it's affirmative).
        Falls back to has_new_data=True (safe: sends to mega-prompt) on error.
        """
        param = self._find_param(parameters, pending["field"])
        label = param["display_label"] if param else pending["field"]
        user_msg = (
            f'Pending confirmation — {label}: "{pending["value"]}"\n'
            f'Caller said: "{utterance}"'
        )
        try:
            return llm_structured_call(
                _CONFIRM_CLASSIFIER_SYSTEM,
                user_msg,
                ConfirmationSignal,
                max_tokens=64,
            )
        except Exception as exc:
            logger.warning("Confirmation classifier failed (%s) — defaulting to mega-prompt", exc)
            return ConfirmationSignal(signal="correct_or_add", is_affirmative=True)

    def _apply_confirm_yes(
        self,
        utterance: str,
        pending: dict,
        collected: dict,
        internal_state: dict,
        parameters: list[dict],
        config: dict,
    ) -> SubagentResponse:
        """Handle a deterministic 'yes' confirmation without calling the mega-prompt."""
        field_name = pending["field"]
        confirmed_value = pending["value"]
        collected = dict(collected)
        collected[field_name] = confirmed_value
        logger.info("DataCollection: confirmed %s = %r (classifier fast-path)", field_name, confirmed_value)

        # Promote next queued extraction if available
        queue: list[dict] = list(internal_state.get("extraction_queue") or [])
        new_pending: dict | None = None
        queue_speak: str | None = None
        while queue:
            queued = queue.pop(0)
            fn = queued["field"]
            if fn in collected:
                continue
            param = self._find_param(parameters, fn)
            if not param:
                continue
            is_valid, msg_or_val = self._validate(param, queued["value"])
            if is_valid:
                new_pending = {"field": fn, "value": msg_or_val}
                queue_speak = self._rephrase_confirmation(new_pending, parameters)
                logger.info("DataCollection: dequeued %r = %r for confirmation", fn, msg_or_val)
                break

        internal_state = {
            **internal_state,
            "collected": collected,
            "pending_confirmation": new_pending,
            "extraction_queue": queue,
        }

        remaining_all = [p for p in parameters if p["name"] not in collected]
        if not remaining_all and not new_pending:
            speak = "Perfect, I have everything I need."
            self._record_history(utterance, speak, internal_state)
            return SubagentResponse(
                status=AgentStatus.COMPLETED,
                speak=speak,
                collected=collected,
                internal_state=internal_state,
                confidence=1.0,
            )

        if new_pending:
            speak = queue_speak or self._rephrase_confirmation(new_pending, parameters)
            self._record_history(utterance, speak, internal_state)
            return SubagentResponse(
                status=AgentStatus.WAITING_CONFIRM,
                speak=speak,
                pending_confirmation=new_pending,
                internal_state=internal_state,
                confidence=0.95,
            )

        # Move to next uncollected field
        next_param = next((p for p in parameters if p["name"] not in collected), None)
        next_q = self._template_question(next_param) if next_param else "I have all your details."
        speak = f"Great, and {next_q[0].lower()}{next_q[1:]}" if next_param else "Great — " + next_q
        self._record_history(utterance, speak, internal_state)
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=speak,
            internal_state=internal_state,
            confidence=0.95,
        )

    def _apply_confirm_no(
        self,
        utterance: str,
        pending: dict,
        collected: dict,
        internal_state: dict,
        parameters: list[dict],
    ) -> SubagentResponse:
        """Handle a deterministic 'no' rejection without calling the mega-prompt."""
        field_name = pending["field"]
        param = self._find_param(parameters, field_name)
        label = param["display_label"] if param else field_name
        logger.info("DataCollection: caller rejected value for %s (classifier fast-path)", field_name)
        internal_state = {**internal_state, "pending_confirmation": None}
        speak = f"No problem. Could you give me your {label} again?"
        self._record_history(utterance, speak, internal_state)
        return SubagentResponse(
            status=AgentStatus.IN_PROGRESS,
            speak=speak,
            internal_state=internal_state,
            confidence=0.95,
        )
