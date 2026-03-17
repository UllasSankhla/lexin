import json
import logging
from sqlalchemy.orm import Session

from app.models.assistant import AssistantConfig
from app.models.parameter import CollectionParameter
from app.models.faq import FAQ
from app.models.context_file import ContextFile
from app.models.spell_rule import SpellRule
from app.models.webhook import WebhookEndpoint
from app.models.calendly_config import CalendlyConfig
from app.models.calendly_event_type import CalendlyEventType
from app.services.file_storage import read_context_file

logger = logging.getLogger(__name__)


def build_config_export(db: Session, owner_id: str) -> dict:
    """Assemble complete configuration payload for the data plane."""

    assistant = db.query(AssistantConfig).filter(AssistantConfig.owner_id == owner_id).first()
    if not assistant:
        logger.warning("No assistant config found for owner=%s — returning empty shell", owner_id)

    parameters = (
        db.query(CollectionParameter)
        .filter(CollectionParameter.owner_id == owner_id)
        .order_by(CollectionParameter.collection_order)
        .all()
    )

    faqs = db.query(FAQ).filter(FAQ.owner_id == owner_id, FAQ.enabled == True).all()  # noqa

    context_files = db.query(ContextFile).filter(ContextFile.owner_id == owner_id, ContextFile.enabled == True).all()  # noqa

    spell_rules = db.query(SpellRule).filter(SpellRule.owner_id == owner_id, SpellRule.enabled == True).all()  # noqa

    webhooks = db.query(WebhookEndpoint).filter(WebhookEndpoint.owner_id == owner_id, WebhookEndpoint.enabled == True).all()  # noqa

    calendly_config = db.query(CalendlyConfig).filter(CalendlyConfig.owner_id == owner_id, CalendlyConfig.enabled == True).first()  # noqa
    calendly_event_types = db.query(CalendlyEventType).filter(CalendlyEventType.enabled == True).all()  # noqa

    context_content = []
    for cf in context_files:
        content = read_context_file(cf.file_path)
        if content:
            context_content.append({
                "name": cf.original_name,
                "description": cf.description,
                "content": content,
            })

    export = {
        "assistant": {
            "id": assistant.id if assistant else None,
            "persona_name": assistant.persona_name if assistant else "Assistant",
            "persona_voice": assistant.persona_voice if assistant else "aura-2-thalia-en",
            "nature": assistant.nature if assistant else "friendly",
            "system_prompt": assistant.system_prompt if assistant else "",
            "greeting_message": assistant.greeting_message if assistant else "Hello! How can I help you today?",
            "max_call_duration_sec": assistant.max_call_duration_sec if assistant else 600,
            "silence_timeout_sec": assistant.silence_timeout_sec if assistant else 10,
            "language": assistant.language if assistant else "en-US",
        },
        "parameters": [
            {
                "id": p.id,
                "name": p.name,
                "display_label": p.display_label,
                "data_type": p.data_type,
                "required": p.required,
                "collection_order": p.collection_order,
                "validation_regex": p.validation_regex,
                "validation_message": p.validation_message,
                "extraction_hints": json.loads(p.extraction_hints) if p.extraction_hints else [],
            }
            for p in parameters
        ],
        "faqs": [
            {"question": f.question, "answer": f.answer, "category": f.category}
            for f in faqs
        ],
        "context_files": context_content,
        "spell_rules": [
            {"wrong_form": r.wrong_form, "correct_form": r.correct_form, "rule_type": r.rule_type}
            for r in spell_rules
        ],
        "webhooks": [
            {
                "id": w.id,
                "name": w.name,
                "url": w.url,
                "secret_header": w.secret_header,
                "secret_value": w.secret_value,
                "events": json.loads(w.events),
                "timeout_sec": w.timeout_sec,
                "retry_count": w.retry_count,
            }
            for w in webhooks
        ],
        "calendly": {
            "api_token": calendly_config.api_token,
            "user_uri": calendly_config.user_uri,
            "organization_uri": calendly_config.organization_uri,
            "lookahead_days": calendly_config.lookahead_days,
            "timezone": calendly_config.timezone,
        } if calendly_config else None,
        "calendly_event_types": [
            {
                "id": et.id,
                "name": et.name,
                "description": et.description,
                "event_type_uri": et.event_type_uri,
            }
            for et in calendly_event_types
        ],
    }

    logger.info(
        "Config export built for owner=%s: %d params, %d FAQs, %d context files, %d webhooks",
        owner_id, len(parameters), len(faqs), len(context_files), len(webhooks),
    )
    return export
