"""Tool for submitting a final SQL query answer to the evaluation pipeline."""

from framework.agent import ANSWER_SUBMITTED_PREFIX, Tool


def submit_answer(query: str, reasoning: str = "") -> str:
    """Submit a SQL query as the final answer to the evaluation.

    Args:
        query: The SQL query to submit as the final answer.
        reasoning: Brief explanation of why each filter/condition is present.

    Returns:
        A confirmation message indicating that the submission has been received.
    """
    # The reasoning param is intentionally unused — it exists to force the
    # agent to articulate its justification before submitting, which reduces
    # spurious filters. The prefix signals the agent to stop.
    return f"{ANSWER_SUBMITTED_PREFIX}{query}"


SUBMIT_ANSWER: Tool = Tool(
    name="submit_answer",
    description=(
        "Submit your final SQL query. Provide reasoning justifying each "
        "filter/condition from the question or business rules."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The final SQL query (schema-qualified table names).",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Justify each WHERE, JOIN, and HAVING clause from "
                    "the question text or business rules."
                ),
            },
        },
        "required": ["query", "reasoning"],
    },
    function=submit_answer,
)
