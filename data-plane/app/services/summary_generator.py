"""Generate post-call AI summary and extract caller name using Cerebras."""
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

    caller_name = caller_name or "Unknown Caller"

    if not transcript_lines:
        return caller_name, "No transcript available for this call."

    # Build condensed transcript (limit to avoid token overflow)
    transcript_text = "\n".join(transcript_lines[-40:])  # last 40 lines

    prompt = f"""You are summarizing a completed voice appointment booking call.
Based on the transcript below, write a concise 2-3 sentence summary covering:
- The caller's purpose/reason for calling
- What information was collected
- The outcome of the call

Be friendly and factual. Write in third person. Do not use bullet points.

TRANSCRIPT:
{transcript_text}

COLLECTED DATA:
{', '.join(f"{k}: {v}" for k, v in collected_params.items()) if collected_params else "None"}

SUMMARY:"""

    try:
        client = Cerebras(api_key=settings.cerebras_api_key)
        response = _call_with_retry(
            client,
            model=settings.cerebras_model,
            messages=[
                {"role": "system", "content": "You are a concise call summary assistant. Output only the summary text, no headers or labels."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        logger.info("Generated summary for call | caller=%s summary=%r", caller_name, summary[:100])
        return caller_name, summary
    except Exception as e:
        logger.error("Failed to generate call summary after retries: %s", e)
        collected_str = ", ".join(f"{k}: {v}" for k, v in collected_params.items())
        fallback = f"Call completed. Collected information: {collected_str}." if collected_params else "Call completed."
        return caller_name, fallback
