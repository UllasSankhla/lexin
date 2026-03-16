"""
Bootstrap script: seed the control plane with a sample appointment assistant configuration.
Run: python seed_data.py
"""
import requests

BASE_URL = "http://localhost:8000/api/v1"


def seed():
    print("Seeding control plane with sample data...")

    # 1. Assistant Configuration
    assistant = {
        "persona_name": "Aria",
        "persona_voice": "aura-2-thalia-en",
        "nature": "friendly and professional",
        "system_prompt": (
            "You are Aria, a warm and professional AI assistant for scheduling appointments. "
            "Your tone is empathetic, clear, and concise — perfect for voice interactions. "
            "Guide callers step-by-step through providing their information. "
            "Never ask for multiple pieces of information at once. "
            "If the caller seems confused, gently re-explain. "
            "Always verify information before moving on."
        ),
        "greeting_message": (
            "Hello! This is Aria, your appointment scheduling assistant. "
            "I'm here to help you book an appointment today."
        ),
        "max_call_duration_sec": 600,
        "silence_timeout_sec": 30,
        "language": "en-US",
    }
    r = requests.put(f"{BASE_URL}/assistant", json=assistant)
    print(f"  Assistant config: {r.status_code}")

    # 2. Collection Parameters
    parameters = [
        {
            "name": "client_full_name",
            "display_label": "Full Name",
            "data_type": "name",
            "required": True,
            "collection_order": 0,
            "extraction_hints": ["name", "called", "I'm", "My name is", "first name", "last name"],
        },
        {
            "name": "date_of_birth",
            "display_label": "Date of Birth",
            "data_type": "date",
            "required": True,
            "collection_order": 1,
            "validation_message": "Please provide your date of birth in year-month-day format",
            "extraction_hints": ["born", "birthday", "DOB"],
        },
        {
            "name": "phone_number",
            "display_label": "Phone Number",
            "data_type": "phone",
            "required": True,
            "collection_order": 2,
            "validation_regex": r"^[\+]?[\d\s\-\(\)]{7,20}$",
            "validation_message": "Please provide a valid phone number",
            "extraction_hints": ["phone", "number", "reach", "call"],
        },
        {
            "name": "email_address",
            "display_label": "Email Address",
            "data_type": "email",
            "required": True,
            "collection_order": 3,
            "validation_regex": r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
            "validation_message": "Please provide a valid email address",
            "extraction_hints": ["email", "@", "at"],
        },
        {
            "name": "insurance_provider",
            "display_label": "Insurance Provider",
            "data_type": "text",
            "required": False,
            "collection_order": 4,
            "extraction_hints": ["insurance", "provider", "plan", "coverage"],
        },
        {
            "name": "reason_for_visit",
            "display_label": "Reason for Visit",
            "data_type": "text",
            "required": True,
            "collection_order": 5,
            "extraction_hints": ["reason", "appointment for", "visit", "issue", "symptoms"],
        },
        {
            "name": "preferred_date",
            "display_label": "Preferred Appointment Date",
            "data_type": "text",
            "required": True,
            "collection_order": 6,
            "extraction_hints": ["prefer", "would like", "available", "next week", "Monday", "Tuesday"],
        },
    ]

    for param in parameters:
        r = requests.post(f"{BASE_URL}/parameters", json=param)
        print(f"  Parameter '{param['name']}': {r.status_code}")

    # 3. FAQs
    faqs = [
        {
            "question": "What are your office hours?",
            "answer": "Our office is open Monday through Friday from 8 AM to 6 PM, and Saturday from 9 AM to 1 PM. We are closed on Sundays.",
            "category": "Hours",
        },
        {
            "question": "How long does a typical appointment take?",
            "answer": "Most appointments last between 20 to 45 minutes. New client appointments may take up to an hour.",
            "category": "Appointments",
        },
        {
            "question": "Do you accept insurance?",
            "answer": "Yes, we accept most major insurance plans. Please have your insurance card ready and we'll verify coverage before your appointment.",
            "category": "Insurance",
        },
        {
            "question": "What should I bring to my first appointment?",
            "answer": "Please bring a valid photo ID, your insurance card, and any relevant records or referrals.",
            "category": "Appointments",
        },
        {
            "question": "Can I cancel or reschedule an appointment?",
            "answer": "Yes, you can cancel or reschedule with at least 24 hours notice. Please call our front desk or use our client portal.",
            "category": "Appointments",
        },
    ]

    for faq in faqs:
        r = requests.post(f"{BASE_URL}/faqs", json=faq)
        print(f"  FAQ '{faq['question'][:40]}...': {r.status_code}")

    # 4. Spell Rules (common STT corrections)
    spell_rules = [
        {"wrong_form": "aria", "correct_form": "Aria", "rule_type": "substitution"},
        {"wrong_form": "auria", "correct_form": "Aria", "rule_type": "substitution"},
        {"wrong_form": "apointment", "correct_form": "appointment", "rule_type": "substitution"},
        {"wrong_form": "insurence", "correct_form": "insurance", "rule_type": "substitution"},
    ]

    r = requests.post(f"{BASE_URL}/spell-rules/import", json={"rules": spell_rules})
    print(f"  Spell rules import: {r.status_code}")

    # 5. Sample Webhook (placeholder)
    webhook = {
        "name": "Sample Zapier Webhook",
        "url": "https://hooks.zapier.com/hooks/catch/example/",
        "events": ["call.completed"],
        "enabled": False,  # Disabled by default until real URL is configured
        "timeout_sec": 10,
        "retry_count": 3,
    }
    r = requests.post(f"{BASE_URL}/webhooks", json=webhook)
    print(f"  Webhook: {r.status_code}")

    print("\nSeeding complete! Config export:")
    r = requests.get(f"{BASE_URL}/config/export")
    data = r.json()
    print(f"  - Assistant: {data.get('assistant', {}).get('persona_name')}")
    print(f"  - Parameters: {len(data.get('parameters', []))}")
    print(f"  - FAQs: {len(data.get('faqs', []))}")
    print(f"  - Spell rules: {len(data.get('spell_rules', []))}")
    print(f"  - Webhooks: {len(data.get('webhooks', []))}")


if __name__ == "__main__":
    seed()
