"""Mock firm configuration used by the evaluator — Nexus Law (immigration/employment/family).

This mirrors the shape of the config dict produced by the control-plane's
config_export service, so agents behave identically to production.
"""
from __future__ import annotations

MOCK_CONFIG: dict = {
    # ── Assistant identity ────────────────────────────────────────────────────
    "assistant": {
        "persona_name":          "Aria",
        "persona_description":   "intake coordinator for Nexus Law, an immigration and employment law firm",
        "greeting_message":      "Thank you for calling Nexus Law. My name is Aria, how can I help you today?",
        "narrative_topic":       "your legal matter",
        "max_call_duration_sec": 600,
    },

    # ── Required caller fields ────────────────────────────────────────────────
    # Mirrors the full intake script used by the human agents at Nexus Law.
    "parameters": [
        {
            "name":             "full_name",
            "display_label":    "Full Name",
            "data_type":        "name",
            "required":         True,
            "collection_order": 0,
            "extraction_hints": ["name", "called", "I'm", "My name is", "first name", "last name"],
        },
        {
            "name":             "email_address",
            "display_label":    "Email Address",
            "data_type":        "email",
            "required":         True,
            "collection_order": 1,
            "extraction_hints": ["email", "e-mail", "@"],
        },
        {
            "name":             "phone_number",
            "display_label":    "Phone Number",
            "data_type":        "phone",
            "required":         True,
            "collection_order": 2,
            "extraction_hints": ["phone", "number", "call me at", "reach me", "area code"],
        },
        {
            "name":             "physical_address",
            "display_label":    "Physical Address",
            "data_type":        "text",
            "required":         False,
            "collection_order": 3,
            "extraction_hints": ["address", "street", "city", "state", "zip", "live at", "located at"],
        },
        {
            "name":             "referral_source",
            "display_label":    "How Did You Hear About Us",
            "data_type":        "text",
            "required":         False,
            "collection_order": 5,
            "extraction_hints": ["found", "heard", "referred", "MetLife", "website", "Google", "friend"],
        },
        {
            "name":             "spoken_before",
            "display_label":    "Spoken With Us Before",
            "data_type":        "text",
            "required":         False,
            "collection_order": 6,
            "extraction_hints": ["before", "previously", "first time", "spoke", "called before"],
        },
        {
            "name":             "calling_on_behalf",
            "display_label":    "Calling on Behalf of Someone Else",
            "data_type":        "text",
            "required":         False,
            "collection_order": 7,
            "extraction_hints": ["behalf", "for someone", "my spouse", "my child", "my parent", "for myself"],
        },
        {
            "name":             "matter_state",
            "display_label":    "State / Country of the Matter",
            "data_type":        "text",
            "required":         True,
            "collection_order": 8,
            "extraction_hints": ["state", "county", "country", "jurisdiction", "Washington", "California"],
        },
        {
            "name":             "metlife_member_id",
            "display_label":    "MetLife Member ID",
            "data_type":        "text",
            "required":         False,
            "collection_order": 9,
            "extraction_hints": ["MetLife", "member ID", "member number", "plan number", "ID number"],
        },
        {
            "name":             "employer_name",
            "display_label":    "Employer Name",
            "data_type":        "text",
            "required":         False,
            "collection_order": 10,
            "extraction_hints": ["employer", "company", "work for", "employer is", "Microsoft", "Google"],
        },
        {
            "name":             "job_title",
            "display_label":    "Job Title",
            "data_type":        "text",
            "required":         False,
            "collection_order": 11,
            "extraction_hints": ["job", "title", "role", "position", "engineer", "manager", "analyst"],
        },
    ],

    # ── Practice areas ────────────────────────────────────────────────────────
    "practice_areas": [
        {
            "name":                     "Immigration Law",
            "description":              "H1B, green cards, visas, citizenship, deportation defence.",
            "qualification_criteria":   (
                "H1B transfers, H1B travel questions, visa renewals, PERM/green card filings, "
                "work authorisation questions, employer sponsorship questions, OPT/CPT queries, "
                "MetLife legal plan members seeking immigration advice."
            ),
            "disqualification_signals": "Criminal immigration violations requiring specialist criminal defence.",
            "ambiguous_signals":        "Asylum claims — may be handled depending on attorney availability.",
            "referral_suggestion":      None,
        },
        {
            "name":                     "Employment Law",
            "description":              "Wrongful termination, discrimination, wage disputes.",
            "qualification_criteria":   "Fired without cause, unpaid overtime, workplace discrimination, hostile work environment.",
            "disqualification_signals": "Contractor disputes, business partnership breakdowns.",
            "ambiguous_signals":        "Non-compete agreement violations.",
            "referral_suggestion":      "Consider contacting the Department of Labor for wage complaints.",
        },
        {
            "name":                     "Family Law",
            "description":              "Divorce, custody, child support, adoption.",
            "qualification_criteria":   "Divorce proceedings, child custody disputes, domestic violence orders.",
            "disqualification_signals": "Business disputes, probate, non-family civil matters.",
            "ambiguous_signals":        "Pre-nuptial agreements — depends on complexity.",
            "referral_suggestion":      None,
        },
    ],

    # ── FAQs ──────────────────────────────────────────────────────────────────
    "faqs": [
        {
            "question": "What areas of law do you practice?",
            "answer":   (
                "Nexus Law handles three main areas: Immigration Law — including H1B visas, "
                "green cards, work authorisation, and employer sponsorship; Employment Law — "
                "including wrongful termination, workplace discrimination, and unpaid wages; "
                "and Family Law — including divorce, child custody, and domestic violence orders. "
                "If you're unsure whether your matter fits, just describe your situation and "
                "we can let you know."
            ),
            "category": "Practice Areas",
        },
        {
            "question": "Can you help me with immigration assistance?",
            "answer":   (
                "Yes, absolutely. We handle a full range of immigration matters including "
                "H1B visa applications and transfers, H1B travel questions, green card and "
                "PERM filings, work authorisation, OPT and CPT queries, and employer "
                "sponsorship cases. We also work with MetLife legal plan members. "
                "Please go ahead and describe your situation and we will get you scheduled."
            ),
            "category": "Practice Areas",
        },
        {
            "question": "Do you handle immigration matters?",
            "answer":   (
                "Yes, we handle immigration matters including H1B visas, transfers, travel, "
                "green cards, and other work authorisation questions. We work with both "
                "individuals and employer-sponsored cases."
            ),
            "category": "Practice Areas",
        },
        {
            "question": "Do you handle employment law cases?",
            "answer":   (
                "Yes, we handle employment law matters including wrongful termination, "
                "workplace discrimination, unpaid overtime, and hostile work environment claims. "
                "If you were fired without cause or weren't paid correctly, we can help."
            ),
            "category": "Practice Areas",
        },
        {
            "question": "Do you handle family law matters?",
            "answer":   (
                "Yes, we handle family law matters including divorce proceedings, child custody "
                "disputes, child support, and domestic violence protective orders. "
                "Please describe your situation and we will get you connected with an attorney."
            ),
            "category": "Practice Areas",
        },
        {
            "question": "Do you work with MetLife legal plan members?",
            "answer":   (
                "Yes, we accept MetLife legal plan coverage. Please have your member ID ready "
                "and we can schedule a consultation covered under your plan."
            ),
        },
        {
            "question": "What are your consultation fees?",
            "answer":   (
                "For MetLife plan members, consultations are covered under your plan. "
                "For non-members, we offer a free initial 30-minute consultation."
            ),
        },
        {
            "question": "How do I schedule an appointment?",
            "answer":   (
                "I can schedule you for a 30-minute video or phone consultation. "
                "The earliest available slot depends on attorney availability — "
                "I can check right now and book you in."
            ),
        },
        {
            "question": "Will I get a confirmation after booking?",
            "answer":   (
                "Yes, you will receive a confirmation email and a text message with the details "
                "of your appointment."
            ),
        },
        {
            "question": "Can I be added to a cancellation list?",
            "answer":   (
                "Absolutely. I can note your preference for an earlier slot and if anything "
                "opens up we will reach out to you as soon as possible."
            ),
        },
        {
            "question": "What is the MetLife website or member portal link?",
            "answer":   (
                "The main MetLife member portal is https://www.metlife.com/ and the specific "
                "login page for members is https://www.metlife.com/member-login/."
            ),
        },
        {
            "question": "How quickly do you respond to emails?",
            "answer":   (
                "We have a 24-hour turnaround to reply to all emails. "
                "If you send us an email today you can expect to hear back within 24 hours."
            ),
        },
    ],

    # ── Context documents ─────────────────────────────────────────────────────
    "context_files": [],

    # ── Calendly / booking (not used in evaluator — MockSchedulingAgent stubs this) ──
    "calendly_api_token":   None,
    "calendly_event_types": [],
    "timezone":             "America/Los_Angeles",

    # ── Webhook (not used in evaluator) ──────────────────────────────────────
    "webhook_endpoints": [],

    # ── Global policy documents ───────────────────────────────────────────────
    "global_policy_documents": [
        {
            "name":        "intake_confirmation_guidelines.txt",
            "description": "Intake call confirmation procedure and data collection flow",
            "content": (
                "INTAKE CALL CONFIRMATION PROCEDURE\n"
                "===================================\n\n"
                "1. COLLECTION ORDER AND CONFIRMATION\n"
                "   Collect and confirm fields in this exact sequence. Do not move to the next\n"
                "   field until the current one has been explicitly confirmed by the caller.\n\n"
                "2. NAME CONFIRMATION\n"
                "   - After the caller provides their name, read it back in full:\n"
                "     \"I have [Full Name] — is that correct?\"\n"
                "   - If the name sounds phonetically ambiguous, unusual, or compound, ask the\n"
                "     caller to spell it before confirming: \"Could you spell that for me?\"\n"
                "   - Never move to the next field until the caller confirms the name is correct.\n\n"
                "3. PHONE NUMBER CONFIRMATION\n"
                "   - Immediately after name is confirmed, ask for phone number.\n"
                "   - Read back in grouped chunks: \"four one five, five five five, zero one nine two\"\n"
                "   - Ask for area code if not provided.\n\n"
                "4. EMAIL ADDRESS CONFIRMATION\n"
                "   - After phone is confirmed, ask for email address.\n"
                "   - Read back with verbal punctuation: \"sarah dot mitchell at gmail dot com\"\n"
                "   - If unclear, ask the caller to spell the local part character by character.\n"
                "   - IMPORTANT: Never confuse the name field with the email field. If the caller\n"
                "     provides their name, ask for the email separately — do not treat the name\n"
                "     as an email address.\n\n"
                "5. PHYSICAL ADDRESS CONFIRMATION\n"
                "   - After email is confirmed, ask for physical address.\n"
                "   - Collect street number, street name, unit/apt if applicable, city, state, ZIP.\n"
                "   - Read the COMPLETE address back before confirming:\n"
                "     \"I have 123 Main Street, Apartment 4B, Seattle, WA 98101 — is that correct?\"\n"
                "   - Do not confirm partial addresses. If any component is missing, ask for it.\n\n"
                "6. CALL CLOSURE\n"
                "   - After all required fields are collected and confirmed, transition gracefully.\n"
                "   - Do not ask for personal data after intake is complete.\n"
                "   - End with a courteous closing if the caller indicates they are done.\n\n"
                "7. HANDLING CONFUSION\n"
                "   - If the caller's utterance is unclear or garbled, acknowledge confusion and\n"
                "     ask for clarification. Never guess or invent a value for a garbled field.\n"
            ),
        },
    ],
}
