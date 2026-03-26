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
            "name":             "phone_number",
            "display_label":    "Phone Number",
            "data_type":        "phone",
            "required":         True,
            "collection_order": 1,
            "extraction_hints": ["phone", "number", "call me at", "reach me", "area code"],
        },
        {
            "name":             "email_address",
            "display_label":    "Email Address",
            "data_type":        "email",
            "required":         True,
            "collection_order": 2,
            "extraction_hints": ["email", "e-mail", "@"],
        },
        {
            "name":             "physical_address",
            "display_label":    "Physical Address",
            "data_type":        "text",
            "required":         True,
            "collection_order": 3,
            "extraction_hints": ["address", "street", "city", "state", "zip", "live at", "located at"],
        },
        {
            "name":             "date_of_birth",
            "display_label":    "Date of Birth",
            "data_type":        "date",
            "required":         True,
            "collection_order": 4,
            "validation_message": "Please provide your date of birth as month, day, and year.",
            "extraction_hints": ["born", "birthday", "date of birth", "DOB"],
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
            "question": "Do you handle immigration matters?",
            "answer":   (
                "Yes, we handle immigration matters including H1B visas, transfers, travel, "
                "green cards, and other work authorisation questions. We work with both "
                "individuals and employer-sponsored cases."
            ),
        },
        {
            "question": "What areas of law do you practice?",
            "answer":   (
                "We handle employment-related matters, immigration, and family-related matters."
            ),
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
    "global_policy_documents": [],
}
