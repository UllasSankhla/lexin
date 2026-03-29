"""Transcript ingestion — parse Smith.ai exports into TestCase objects."""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from pathlib import Path

from evaluator.models import (
    AgentTurn, CallerTurn, RawConversation, TestCase, Turn,
)

logger = logging.getLogger(__name__)

# Speaker label sets — lowercased
_CALLER_LABELS = {"caller", "client", "customer", "lead", "prospect"}
_AGENT_LABELS  = {"agent", "smith", "smith.ai", "receptionist", "virtual receptionist",
                  "operator", "staff", "representative", "intake", "assistant"}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise_speaker(raw_label: str) -> str:
    """Return 'caller' or 'agent'; falls back to 'agent' for unknowns."""
    label = raw_label.strip().lower().rstrip(":")
    if label in _CALLER_LABELS:
        return "caller"
    if label in _AGENT_LABELS:
        return "agent"
    # Heuristic: if the label contains "caller" or "client" it's caller
    if any(k in label for k in ("caller", "client", "customer")):
        return "caller"
    logger.debug("Unknown speaker label %r — treating as agent", raw_label)
    return "agent"


# ── Format-specific parsers ───────────────────────────────────────────────────

# Smith.ai export format:
#   <Speaker Name>
#   <Role>          ← e.g. "Virtual Receptionist" or "Caller"
#   <text line(s)>
#   <next Speaker Name>
#   ...
_SMITH_ROLE_RE = re.compile(
    r"^(caller|client|customer|lead|prospect|agent|virtual receptionist|"
    r"receptionist|operator|staff|representative|intake|assistant|smith\.?ai?)$",
    re.IGNORECASE,
)


def _parse_smith_ai_blocks(text: str) -> list[Turn] | None:
    """
    Parse Smith.ai multi-block exports where each turn is structured as:
        <Speaker Name>
        <Role>
        <speech text — one or more lines>

    Returns None if the format is not detected (fewer than 2 role lines found).
    """
    lines = text.splitlines()

    # Find indices of pure role lines (preceded by a non-empty name line).
    role_indices = [
        i for i, ln in enumerate(lines)
        if _SMITH_ROLE_RE.match(ln.strip()) and i > 0 and lines[i - 1].strip()
    ]

    if len(role_indices) < 2:
        return None

    turns: list[Turn] = []
    turn_index = 0

    for k, role_idx in enumerate(role_indices):
        role_line = lines[role_idx].strip()

        # Text runs from the line after the role until the line before the
        # next speaker-name line (which is role_indices[k+1] - 1).
        text_start = role_idx + 1
        text_end   = role_indices[k + 1] - 1 if k + 1 < len(role_indices) else len(lines)

        text_body = " ".join(
            ln.strip() for ln in lines[text_start:text_end] if ln.strip()
        )
        if not text_body:
            continue

        speaker = _normalise_speaker(role_line)
        turns.append(Turn(speaker=speaker, text=text_body, turn_index=turn_index))
        turn_index += 1

    return turns if turns else None


def _parse_csv(text: str) -> list[Turn] | None:
    """Try to parse as CSV. Returns None if the format doesn't look right."""
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except Exception:
        return None

    if not rows:
        return None

    # Find speaker and text columns (case-insensitive)
    fieldnames = [f.lower() for f in (reader.fieldnames or [])]
    speaker_col = next((f for f in fieldnames if f in
                        ("speaker", "role", "party", "from", "name")), None)
    text_col = next((f for f in fieldnames if f in
                     ("text", "content", "utterance", "message", "transcript")), None)

    if not speaker_col or not text_col:
        return None

    # Re-map original fieldnames
    orig_fields = reader.fieldnames or []
    orig_speaker = orig_fields[fieldnames.index(speaker_col)]
    orig_text    = orig_fields[fieldnames.index(text_col)]

    turns: list[Turn] = []
    for i, row in enumerate(rows):
        speaker = _normalise_speaker(row.get(orig_speaker, ""))
        text    = (row.get(orig_text) or "").strip()
        if text:
            turns.append(Turn(speaker=speaker, text=text, turn_index=i))
    return turns if turns else None


_SPEAKER_LINE_RE = re.compile(
    r"^(?P<speaker>[A-Za-z][A-Za-z0-9 ._\-]{0,30}?)\s*:\s*(?P<text>.+)$"
)


def _parse_plain_text(text: str) -> list[Turn]:
    """
    Parse plain-text transcripts with lines like:
        Caller: Hi I was in an accident...
        Agent: I'm sorry to hear that...

    Multi-line turns (text that continues without a new speaker label) are
    appended to the previous turn.
    """
    turns: list[Turn] = []
    current_speaker: str | None = None
    current_parts: list[str] = []
    turn_index = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = _SPEAKER_LINE_RE.match(line)
        if m:
            # Flush previous turn
            if current_speaker is not None and current_parts:
                turns.append(Turn(
                    speaker=current_speaker,
                    text=" ".join(current_parts),
                    turn_index=turn_index,
                ))
                turn_index += 1
                current_parts = []

            current_speaker = _normalise_speaker(m.group("speaker"))
            rest = m.group("text").strip()
            if rest:
                current_parts.append(rest)
        else:
            # Continuation of previous turn
            if current_speaker is not None:
                current_parts.append(line)

    # Flush last turn
    if current_speaker is not None and current_parts:
        turns.append(Turn(
            speaker=current_speaker,
            text=" ".join(current_parts),
            turn_index=turn_index,
        ))

    return turns


# ── Public API ────────────────────────────────────────────────────────────────

def parse_transcript(path: Path) -> RawConversation:
    """Load and parse a transcript file into a RawConversation."""
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    file_hash = _hash_file(path)

    turns = _parse_csv(raw_text)
    if turns is None:
        turns = _parse_smith_ai_blocks(raw_text)
        if turns is not None:
            logger.debug("Parsed %s as Smith.ai block format", path.name)
    if turns is None:
        logger.debug("CSV/Smith.ai parse failed for %s — trying plain text", path.name)
        turns = _parse_plain_text(raw_text)

    logger.info("Parsed %s: %d turns", path.name, len(turns))
    return RawConversation(
        source_file=str(path),
        file_hash=file_hash,
        raw_text=raw_text,
        turns=turns,
    )


def build_test_case(raw: RawConversation) -> TestCase:
    """
    Convert a RawConversation into a TestCase.

    Consecutive same-speaker turns are collapsed into one.  Then caller and
    agent turns are extracted separately and paired: each CallerTurn is paired
    with the AgentTurn that immediately follows it.
    """
    # Collapse consecutive same-speaker turns
    collapsed: list[Turn] = []
    for turn in raw.turns:
        if collapsed and collapsed[-1].speaker == turn.speaker:
            # Merge into previous
            collapsed[-1] = Turn(
                speaker=collapsed[-1].speaker,
                text=collapsed[-1].text + " " + turn.text,
                turn_index=collapsed[-1].turn_index,
            )
        else:
            collapsed.append(turn)

    # caller_turns will be built from paired after re-indexing (below).
    # human_agent_turns keeps original collapsed indices (used only for reference).
    human_agent_turns = [
        AgentTurn(text=t.text, turn_index=t.turn_index)
        for t in collapsed if t.speaker == "agent"
    ]

    # Pair each caller turn with the agent turn that follows it (if any).
    # Use sequential 1-based caller turn indices so evaluation logs show
    # 1, 2, 3, ... regardless of how many agent turns precede the first caller.
    paired: list[tuple[CallerTurn, AgentTurn]] = []
    caller_seq = 0
    turn_iter = iter(collapsed)
    for turn in turn_iter:
        if turn.speaker == "caller":
            caller_seq += 1
            caller_t = CallerTurn(text=turn.text, turn_index=caller_seq)
            # Find next agent turn
            for nxt in turn_iter:
                if nxt.speaker == "agent":
                    paired.append((caller_t, AgentTurn(text=nxt.text, turn_index=nxt.turn_index)))
                    break
            else:
                # No following agent turn
                paired.append((caller_t, AgentTurn(text="", turn_index=-1)))

    # Derive caller_turns from the sequentially-indexed paired list so that
    # caller_turns[i].turn_index == i+1 and replay.py uses the same indices.
    caller_turns = [ct for ct, _ in paired]

    conv_id = Path(raw.source_file).stem
    logger.info(
        "TestCase %s: %d caller turns, %d pairs",
        conv_id, len(caller_turns), len(paired),
    )
    return TestCase(
        conversation_id=conv_id,
        source_file=raw.source_file,
        file_hash=raw.file_hash,
        caller_turns=caller_turns,
        human_agent_turns=human_agent_turns,
        paired_exchanges=paired,
    )
