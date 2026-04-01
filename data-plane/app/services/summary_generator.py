"""Generate post-call AI summary and extract caller name via configured LLM provider."""
import logging
from cerebras.cloud.sdk import Cerebras

from app.config import settings
from app.pipeline.llm import _call_with_retry

logger = logging.getLogger(__name__)


def generate_call_summary(
    transcript_lines: list[str],
    collected_params: dict,
    config: dict | None = None,
) -> tuple[str, str]:
    """
    Generate a brief AI summary of a completed call.
    Returns (caller_name, summary_text).
    Runs synchronously — call via run_in_executor.
    """
    # Try to find the caller name using the config's parameter definitions first.
    # This handles any field name the operator configured with data_type="name".
    caller_name = None
    if config:
        for param in config.get("parameters", []):
            if param.get("data_type") == "name":
                caller_name = collected_params.get(param["name"]) or None
                if caller_name:
                    break

    # Fall back to common hardcoded key names if config lookup didn't find one
    if not caller_name:
        caller_name = (
            collected_params.get("client_name")
            or collected_params.get("client_full_name")
            or collected_params.get("caller_name")
            or collected_params.get("name")
        )

    # Try concatenating first_name + last_name variants
    if not caller_name:
        first = next(
            (collected_params[k] for k in collected_params if "first_name" in k),
            None,
        )
        last = next(
            (collected_params[k] for k in collected_params if "last_name" in k),
            None,
        )
        if first or last:
            caller_name = " ".join(filter(None, [first, last]))

    if not transcript_lines:
        return caller_name or "Unknown Caller", "No transcript available for this call."

    # Build condensed transcript (limit to avoid token overflow)
    transcript_text = "\n".join(transcript_lines[-40:])  # last 40 lines

    client = Cerebras(api_key=settings.cerebras_api_key)
    model = settings.cerebras_model

    def _complete(system: str, user: str, temperature: float = 0.3) -> str:
        r = _call_with_retry(client, model=model,
                             messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                             max_tokens=1024, temperature=temperature)
        return r.choices[0].message.content.strip() if r.choices else ""

    # If no name found from collected params, try extracting it from the transcript
    if not caller_name:
        try:
            extracted = _complete(
                "Extract the caller's full name from a call transcript. "
                "Reply with ONLY the name (e.g. 'Jane Smith'). "
                "If no name is mentioned, reply with exactly: Unknown Caller",
                transcript_text,
                temperature=0,
            )
            if extracted and extracted.lower() != "unknown caller":
                caller_name = extracted
                logger.info("Extracted caller name from transcript: %r", caller_name)
        except Exception as exc:
            logger.warning("Transcript name extraction failed: %s", exc)

    caller_name = caller_name or "Unknown Caller"

    prompt = (
        "You are summarizing a completed voice call.\n"
        "Write a concise 2-3 sentence summary covering the caller's purpose, "
        "what information was collected, and the outcome. "
        "Be factual, write in third person, no bullet points.\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"COLLECTED DATA:\n"
        f"{', '.join(f'{k}: {v}' for k, v in collected_params.items()) if collected_params else 'None'}\n\n"
        "SUMMARY:"
    )

    try:
        summary = _complete(
            "You are a concise call summary assistant. Output only the summary text, no headers or labels.",
            prompt,
        )
        logger.info("Generated summary for call | caller=%s summary=%r", caller_name, summary[:100])
        return caller_name, summary
    except Exception as e:
        logger.error("Failed to generate call summary after retries: %s", e)
        collected_str = ", ".join(f"{k}: {v}" for k, v in collected_params.items())
        fallback = f"Call completed. Collected information: {collected_str}." if collected_params else "Call completed."
        return caller_name, fallback
