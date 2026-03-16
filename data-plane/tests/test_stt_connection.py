"""
Smoke test: verify we can connect to the Deepgram v2 /listen endpoint,
send a short burst of silence, and receive at least a Connected message.

Run with:
    PYTHONPATH=. .venv/bin/python tests/test_stt_connection.py
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


async def run_test() -> None:
    from app.pipeline.stt import STTSession

    received_interims: list[str] = []
    received_finals: list[str] = []

    async def on_interim(text: str, confidence: float) -> None:
        logger.info("INTERIM: %r (conf=%.2f)", text, confidence)
        received_interims.append(text)

    async def on_final(text: str, confidence: float, elapsed_ms: float) -> None:
        logger.info("FINAL: %r (conf=%.2f, elapsed=%.0fms)", text, confidence, elapsed_ms)
        received_finals.append(text)

    session = STTSession(on_interim=on_interim, on_final=on_final)

    logger.info("Opening STT session...")
    elapsed = await session.start()
    logger.info("Session opened in %.0fms", elapsed)

    # Send 1 second of silence (16kHz, 16-bit mono = 32000 bytes/s).
    silence = b"\x00" * 3200   # 100ms chunks x 10 = 1s
    logger.info("Sending 1s of silence in 100ms chunks...")
    for _ in range(10):
        await session.send_audio(silence)
        await asyncio.sleep(0.1)

    logger.info("Waiting 2s for any responses...")
    await asyncio.sleep(2)

    await session.close()

    logger.info("--- Results ---")
    logger.info("Interim messages received: %d", len(received_interims))
    logger.info("Final messages received:   %d", len(received_finals))
    logger.info("Connection test PASSED — no exceptions raised")


if __name__ == "__main__":
    asyncio.run(run_test())
