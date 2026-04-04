"""Pre-generated empathy fillers for masking LLM processing latency spikes.

Used by the handler when agent processing exceeds the filler threshold (default
500 ms). A primary filler fires first, then continuation phrases cycle until the
agent completes.

Phrases are intentionally generic — no semantic content that could be wrong.
"""
from __future__ import annotations

import random
from typing import Iterator

_PRIMARY: list[str] = [
    "Sure, one moment.",
    "Let me check that for you.",
    "Of course, just a second.",
    "Got it, one moment.",
]

_CONTINUATION: list[str] = [
    "Thanks for your patience.",
    "Almost there.",
    "Just a moment longer.",
    "Still on it, thank you.",
]

# Maximum number of filler phrases to send per turn (primary + continuations).
# Caps total filler coverage at roughly MAX_FILLERS × ~1–2 s of TTS audio.
MAX_FILLERS = 3


def filler_sequence() -> Iterator[str]:
    """Yield one primary filler, then cycle through continuations without repeating."""
    yield random.choice(_PRIMARY)
    pool = list(_CONTINUATION)
    random.shuffle(pool)
    for phrase in pool:
        yield phrase
    # If somehow the agent still hasn't responded, cycle again
    while True:
        random.shuffle(pool)
        for phrase in pool:
            yield phrase
