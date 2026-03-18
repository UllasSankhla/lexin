"""Static workflow graph definition. Add new agents here."""
from app.agents.workflow import WorkflowNode

WORKFLOW_NODES = [
    WorkflowNode(
        id="faq",
        description=(
            "Answers specific questions from a curated FAQ list. "
            "Use when the caller asks a question that may be covered by the FAQ."
        ),
        depends_on=[],
        interrupt_eligible=True,
        auto_run=False,
        completion_criteria="Caller's specific question answered from FAQ list",
    ),
    WorkflowNode(
        id="context_docs",
        description=(
            "Answers general questions about the business using provided documents. "
            "Use when the FAQ doesn't cover the question."
        ),
        depends_on=[],
        interrupt_eligible=True,
        auto_run=False,
        completion_criteria="Caller's general business question answered from documents",
    ),
    WorkflowNode(
        id="fallback",
        description=(
            "Handles questions that cannot be answered. Acknowledges the caller, "
            "takes a note for follow-up. Use when no other agent can answer."
        ),
        depends_on=[],
        interrupt_eligible=True,
        auto_run=False,
        completion_criteria="Caller acknowledged and note recorded",
    ),
    WorkflowNode(
        id="data_collection",
        description=(
            "Collects required caller information one field at a time with confirmation. "
            "Use when the caller is providing personal or case information, "
            "or when no other agent is handling the turn."
        ),
        depends_on=[],
        interrupt_eligible=False,
        auto_run=False,
        completion_criteria="All required fields collected and confirmed by caller",
    ),
    WorkflowNode(
        id="scheduling",
        description=(
            "Presents available appointment slots, takes the caller's choice, "
            "confirms and books. Use after all data is collected."
        ),
        depends_on=["data_collection"],
        interrupt_eligible=False,
        auto_run=False,
        completion_criteria="Appointment slot chosen and booking confirmed",
    ),
    WorkflowNode(
        id="webhook",
        description="Sends call summary and collected data to configured endpoints.",
        depends_on=["scheduling"],
        interrupt_eligible=False,
        auto_run=True,
        completion_criteria="All webhooks fired",
    ),
]
