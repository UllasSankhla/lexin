"""Quick TTS smoke test.

Synthesizes a short phrase and saves the output to /tmp/tts_test.wav
so you can play it back and verify the voice sounds correct.

Usage:
    cd data-plane
    python test_tts.py
"""
import asyncio
import wave
import sys
import os

# Load .env so the API key is available without starting the full app
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from app.pipeline.tts import TTSClient
from app.config import settings

TEST_PHRASE = (
    "Hello! I'm your appointment booking assistant. "
    "Could you please tell me your full name?"
)
OUTPUT_PATH = "/tmp/tts_test.wav"


async def main():
    print(f"Model : {settings.deepgram_tts_model}")
    print(f"Phrase: {TEST_PHRASE!r}")
    print()

    client = TTSClient()
    print("Synthesizing...")
    audio_bytes, latency_ms = await client.synthesize(TEST_PHRASE)

    print(f"Latency : {latency_ms:.0f} ms")
    print(f"Audio   : {len(audio_bytes):,} bytes")

    # Write as a playable WAV file (linear16, 24 kHz, mono)
    with wave.open(OUTPUT_PATH, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit = 2 bytes
        wf.setframerate(24000)
        wf.writeframes(audio_bytes)

    print(f"Saved   : {OUTPUT_PATH}")
    print()
    print("Play with:  aplay -r 24000 -f S16_LE -c 1 /tmp/tts_test.wav")
    print("        or: ffplay /tmp/tts_test.wav")


if __name__ == "__main__":
    asyncio.run(main())
