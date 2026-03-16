"""Workflow simulation test.

Runs a scripted caller through the full parameter-collection loop using real
LLM (Cerebras) and real TTS (Deepgram).  Every assistant turn is synthesised
to a WAV file in /tmp/workflow_test/ so you can play back the whole
conversation.

Usage:
    cd data-plane
    python test_workflow.py

Output files:
    /tmp/workflow_test/00_greeting.wav
    /tmp/workflow_test/01_opening.wav
    /tmp/workflow_test/02_*.wav  ...one per assistant turn
"""
import asyncio
import os
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from app.pipeline.booking_workflow import BookingWorkflow, WorkflowStage
from app.pipeline.llm import LLMClient, LLMToolkit, TOOLKIT_SYSTEM_PROMPT
from app.pipeline.parameter_collector import load_parameters
from app.pipeline.tts import TTSClient
from app.config import settings

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/tmp/workflow_test")

# Scripted caller utterances in order.
# Each entry is (label, utterance).  The workflow alternates between asking
# for a value and asking for confirmation, so utterances alternate between
# providing a value and saying yes/no.
SCRIPT = [
    # Field 0 – Full Name (text, required)
    ("name_value",         "My name is John Doe"),
    ("name_confirm",       "Yes that's correct"),
    # Field 1 – Date of Birth (date, required)
    ("dob_value",          "I was born on March 15th 1985"),
    ("dob_confirm",        "Yes that's correct"),
    # Field 2 – Phone Number (phone, required)
    ("phone_value",        "My number is 555 867 5309"),
    ("phone_confirm",      "Yes"),
    # Field 3 – Email Address (email, required)
    ("email_value",        "It's john dot doe at example dot com"),
    ("email_confirm",      "Yes that's right"),
    # Field 4 – Insurance Provider (text, optional)
    ("insurance_value",    "Blue Cross Blue Shield"),
    ("insurance_confirm",  "Yes"),
    # Field 5 – Reason for Visit (text, required)
    ("reason_value",       "I need a consultation about a contract dispute"),
    ("reason_confirm",     "Correct"),
    # Field 6 – Preferred Appointment Date (text, required)
    ("prefdate_value",     "Next Monday morning would work for me"),
    ("prefdate_confirm",   "Yes that's fine"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_wav(audio_bytes: bytes, path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # linear16
        wf.setframerate(24000)
        wf.writeframes(audio_bytes)


def separator(label: str = "") -> None:
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


async def speak(tts: TTSClient, text: str, filepath: Path) -> None:
    audio, latency_ms = await tts.synthesize(text)
    save_wav(audio, filepath)
    print(f"    🔊 {latency_ms:.0f} ms  →  {filepath.name}  ({len(audio):,} bytes)")


# ── Main simulation ───────────────────────────────────────────────────────────

async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load config from control plane ───────────────────────────────────────
    print("Fetching config from control plane...")
    import httpx
    try:
        resp = httpx.get(
            f"{settings.control_plane_url}/api/v1/config/export",
            headers={"X-API-Key": settings.control_plane_api_key},
            timeout=10,
        )
        resp.raise_for_status()
        config = resp.json()
    except Exception as exc:
        print(f"ERROR: Could not reach control plane: {exc}")
        sys.exit(1)

    params = sorted(config.get("parameters", []), key=lambda p: p["collection_order"])
    print(f"Parameters to collect ({len(params)} total):")
    for p in params:
        req = "required" if p["required"] else "optional"
        print(f"  {p['collection_order']}. {p['display_label']} ({p['data_type']}, {req})")

    # ── Initialise pipeline ───────────────────────────────────────────────────
    collection_state = load_parameters(config)
    llm_client  = LLMClient(system_prompt=TOOLKIT_SYSTEM_PROMPT)
    llm_toolkit = LLMToolkit(llm_client)
    workflow    = BookingWorkflow(collection_state, llm_toolkit, config)
    tts         = TTSClient()
    loop        = asyncio.get_event_loop()

    turn = 0  # file counter

    # ── Greeting ──────────────────────────────────────────────────────────────
    separator("CALL START")
    greeting = config["assistant"].get("greeting_message", "Hello, how can I help?")
    print(f"\nASSISTANT: {greeting}")
    await speak(tts, greeting, OUTPUT_DIR / f"{turn:02d}_greeting.wav")
    turn += 1

    # ── Opening question (first field) ────────────────────────────────────────
    separator("workflow.get_opening()")
    opening = await loop.run_in_executor(None, workflow.get_opening)
    print(f"\nASSISTANT: {opening}")
    await speak(tts, opening, OUTPUT_DIR / f"{turn:02d}_opening.wav")
    turn += 1

    # ── Scripted turns ────────────────────────────────────────────────────────
    collected: dict[str, str] = {}

    for label, utterance in SCRIPT:
        separator(f"USER: {label}")
        print(f"\n    USER: {utterance}")

        result = await loop.run_in_executor(
            None,
            lambda u=utterance: workflow.process_utterance(u),
        )

        print(f"\nASSISTANT: {result.speak}")
        filename = f"{turn:02d}_{label}.wav"
        await speak(tts, result.speak, OUTPUT_DIR / filename)
        turn += 1

        if result.field_confirmed:
            name, raw, value = result.field_confirmed
            collected[name] = value
            print(f"\n    ✓  CONFIRMED  {name!r:25s} = {value!r}")

        if result.booking_details:
            print(f"\n    📅 BOOKING: {result.booking_details}")

        if result.call_complete:
            print("\n    [workflow marked call_complete]")
            break

        # Stop once we've entered scheduling — out of scope for this test
        if workflow.stage == WorkflowStage.SCHEDULING:
            separator("SCHEDULING STAGE REACHED")
            print("All parameters collected. Scheduling stage would follow.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    separator("SUMMARY")
    print(f"\nCollected {len(collected)}/{len([p for p in params if p['required']])} required parameters:\n")
    for name, value in collected.items():
        print(f"  {name:25s} = {value!r}")

    print(f"\nAudio files saved to: {OUTPUT_DIR}/")
    files = sorted(OUTPUT_DIR.glob("*.wav"))
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")

    print(f"\nPlay all in order:")
    print(f"  for f in {OUTPUT_DIR}/*.wav; do aplay -r 24000 -f S16_LE -c 1 \"$f\"; done")


if __name__ == "__main__":
    asyncio.run(main())
