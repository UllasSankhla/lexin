"""Static workflow definition for the appointment booking use-case."""
from app.agents.workflow import (
    WorkflowDefinition, ActivityNode, Edge,
    GoalSpec, DeciderSpec, InterruptPolicy, ErrorPolicy,
)

APPOINTMENT_BOOKING = WorkflowDefinition(

    id="appointment_booking",
    description="Collect caller information and schedule an appointment",

    goal=GoalSpec(
        primary_agents=["data_collection", "scheduling"],
        description=(
            "You must complete two primary goals in order:\n"
            "1. data_collection — collect all required caller information\n"
            "2. scheduling — present available appointment slots and confirm a booking\n"
            "Always work toward the next incomplete primary goal. "
            "Interrupt agents (faq, context_docs, fallback) exist only to answer "
            "side questions; the invoker will return to the primary goal automatically."
        ),
    ),

    decider=DeciderSpec(
        system_prompt="""\
You are the routing decider for a voice appointment booking assistant.

RULES:
1. If the caller is answering a question or providing information, route to the
   active or next primary goal agent (data_collection or scheduling).
2. If the caller is asking a question (not providing information), route to an
   interrupt-eligible agent (faq, context_docs, or fallback).
3. After any interrupt-eligible agent, the invoker handles returning to the
   primary goal — you do not need to re-select it.
4. Only select agents listed under AVAILABLE AGENTS.

Respond ONLY with valid JSON:
{"agent_id": "<id>", "interrupt": <true|false>, "reasoning": "<one line>"}
""",
        max_tokens=1024,
    ),

    nodes=[

        ActivityNode(
            id="faq",
            agent_class="FAQAgent",
            description="Answers specific questions from the curated FAQ list.",
            interrupt_eligible=True,
            on_complete=Edge("resume"),
            on_failed=Edge("context_docs"),
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="context_docs",
            agent_class="ContextDocsAgent",
            description="Answers general questions from uploaded business documents.",
            interrupt_eligible=True,
            on_complete=Edge("resume"),
            on_failed=Edge("fallback"),
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="fallback",
            agent_class="FallbackAgent",
            description="Acknowledges unanswerable questions and records a note.",
            interrupt_eligible=True,
            on_complete=Edge("resume"),
            on_failed=Edge("resume"),
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="data_collection",
            agent_class="DataCollectionAgent",
            description=(
                "PRIMARY GOAL 1: Collect all required caller information "
                "one field at a time with confirmation."
            ),
            is_primary_goal=True,
            goal_order=1,
            interrupt_eligible=False,
            depends_on=[],
            on_complete=Edge("scheduling"),
            on_failed=Edge("end", "data_collection_failed"),
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="scheduling",
            agent_class="SchedulingAgent",
            description=(
                "PRIMARY GOAL 2: Present available appointment slots, "
                "take the caller's choice, confirm and book."
            ),
            is_primary_goal=True,
            goal_order=2,
            interrupt_eligible=False,
            depends_on=["data_collection"],
            on_complete=Edge("end", "completed"),
            on_failed=Edge("end", "scheduling_failed"),
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="webhook",
            agent_class="WebhookAgent",
            description="Dispatches call summary and collected data to configured endpoints.",
            auto_run=True,
            depends_on=["scheduling"],
            on_complete=Edge("end"),
            on_failed=Edge("end", "webhook_failed"),
            on_continue=Edge("end"),
        ),
    ],

    interrupt_policy=InterruptPolicy(
        eligible_agents=["faq", "context_docs", "fallback"],
        resume_strategy="stack",
    ),

    error_policy=ErrorPolicy(
        max_consecutive_errors=3,
        transient_error_speak=(
            "I'm sorry, something went wrong on my end. Could you please repeat that?"
        ),
        on_max_errors=Edge("end", "agent_error"),
    ),
)
