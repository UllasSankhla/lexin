"""Static workflow definition for the appointment booking use-case."""
from app.agents.workflow import (
    WorkflowDefinition, ActivityNode, Edge,
    GoalSpec, DeciderSpec, InterruptPolicy, ErrorPolicy, ToolBinding,
)
from app.tools.base import ToolTrigger

APPOINTMENT_BOOKING = WorkflowDefinition(

    id="appointment_booking",
    description="Collect caller information, understand their matter, qualify it, and schedule an appointment",

    goal=GoalSpec(
        primary_agents=["data_collection", "narrative_collection", "intake_qualification", "scheduling"],
        description=(
            "You must complete four primary goals in order:\n"
            "1. data_collection       — collect all required caller information (name, contact details, etc.)\n"
            "2. narrative_collection  — listen to the caller describe their legal matter in free-form speech\n"
            "3. intake_qualification  — assess whether the matter falls within the firm's practice areas\n"
            "4. scheduling            — present available appointment slots and confirm a booking\n"
            "Always work toward the next incomplete primary goal. "
            "Interrupt agents (faq, context_docs, fallback) exist only to answer "
            "side questions; the invoker will return to the primary goal automatically.\n"
            "Note: intake_qualification runs automatically after narrative_collection completes "
            "and does not require a caller utterance."
        ),
    ),

    decider=DeciderSpec(
        system_prompt="""\
You are the routing decider for a voice appointment booking assistant.

RULES:
1. If the caller is answering a question or providing information, route to the
   active or next primary goal agent (data_collection, narrative_collection, or scheduling).
2. narrative_collection accepts free-form speech — route to it whenever the caller
   is describing their legal matter (not asking a question).
3. If the caller is asking a question (not providing information), route to an
   interrupt-eligible agent (faq, context_docs, or fallback).
4. After any interrupt-eligible agent, the invoker handles returning to the
   primary goal — you do not need to re-select it.
5. intake_qualification is auto-run — never select it directly.
6. Only select agents listed under AVAILABLE AGENTS.

NARRATIVE PRIORITY RULE (overrides rule 3 when narrative_collection is in_progress):
Callers routinely embed social or rhetorical questions inside their narrative
opening. Examples:
  - "I want help with a divorce case, can you help me with this?"
  - "I was fired without cause, is this something you handle?"
  - "I was in a car accident and I'm in pain. Am I in the right place?"
  - "My employer didn't pay me overtime — does that make sense?"
These are narrative-opening moves, NOT FAQ interruptions. The caller is
describing their matter and seeking reassurance, not requesting a policy answer.
WHEN narrative_collection status is in_progress, route to narrative_collection
UNLESS the utterance is PRIMARILY a specific, concrete question about firm
policy, fees, location, or process with little or no narrative content.
  ROUTE TO narrative_collection: "I need help with a divorce, can you help?"
  ROUTE TO faq:                  "What are your consultation fees?"
  ROUTE TO faq:                  "I was in an accident. What are your fees?"

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
            on_complete=Edge("narrative_collection"),
            on_failed=Edge("end", "data_collection_failed"),
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="narrative_collection",
            agent_class="NarrativeCollectionAgent",
            description=(
                "PRIMARY GOAL 2: Listen to the caller describe their legal matter "
                "in free-form speech across multiple turns."
            ),
            is_primary_goal=True,
            goal_order=2,
            interrupt_eligible=False,
            depends_on=["data_collection"],
            on_complete=Edge("intake_qualification"),
            on_failed=Edge("end", "narrative_collection_failed"),
            on_continue=Edge("decider"),
        ),

        ActivityNode(
            id="intake_qualification",
            agent_class="IntakeQualificationAgent",
            description=(
                "PRIMARY GOAL 3: Assess whether the caller's matter falls within "
                "the firm's practice areas (runs immediately after narrative completes)."
            ),
            is_primary_goal=True,
            goal_order=3,
            interrupt_eligible=False,
            depends_on=["narrative_collection"],
            on_complete=Edge("scheduling"),
            on_failed=Edge("end", "not_qualified"),
            on_continue=Edge("end", "qualification_error"),
        ),

        ActivityNode(
            id="scheduling",
            agent_class="SchedulingAgent",
            description=(
                "PRIMARY GOAL 4: Present available appointment slots, "
                "take the caller's choice, confirm and book."
            ),
            is_primary_goal=True,
            goal_order=4,
            interrupt_eligible=False,
            depends_on=["intake_qualification"],
            on_complete=Edge("end", "completed"),
            on_failed=Edge("end", "scheduling_failed"),
            on_continue=Edge("decider"),
        ),

    ],

    # ── Tools ─────────────────────────────────────────────────────────────────
    # Non-conversational async work units invoked by the workflow at lifecycle
    # trigger points.  They do not appear as graph nodes and cannot speak to
    # the caller.  CALL_END tools run sequentially in declaration order inside
    # a single background task (so each can read ctx.shared written by the
    # previous one).  AGENT_COMPLETE tools with fire_and_forget=False are
    # started immediately as asyncio.Tasks and awaited before the next agent
    # in the chain runs, enabling parallel network calls.
    tools=[
        # Start fetching calendar slots the moment intake_qualification finishes
        # its LLM call — overlapping with the graph edge resolution so the slots
        # are ready (or nearly so) by the time SchedulingAgent._present_slots runs.
        ToolBinding(
            tool_class="CalendarPrefetchTool",
            trigger=ToolTrigger.AGENT_COMPLETE,
            agent_id="narrative_collection",  # starts when narrative completes
            fire_and_forget=False,
            await_before_agent="scheduling",  # only awaited before scheduling runs,
                                              # so it overlaps with the intake_qualification LLM call
        ),
        # Post-call: summarise first so WebhookTool can read caller_name/ai_summary
        # from ctx.shared without an extra DB query.
        ToolBinding(
            tool_class="SummarizationTool",
            trigger=ToolTrigger.CALL_END,
            fire_and_forget=True,    # entire CALL_END group runs as one background task
        ),
        ToolBinding(
            tool_class="WebhookTool",
            trigger=ToolTrigger.CALL_END,
            fire_and_forget=True,
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
