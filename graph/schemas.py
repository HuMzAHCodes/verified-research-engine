"""Output schemas: the shape each model call must return.

CONCEPT: structured-output-planning. A schema is the contract for one prompt.
The gateway binds it with with_structured_output(), so the model's reply comes
back as a validated object or as an error, never as free text to parse.

Everything in a schema is sent to the model as part of a tool definition:
- the class docstring becomes the tool's description,
- each Field description tells the model what to put in that field,
- Literal types close a choice to a fixed set of values,
- field order is the order the model writes them in.
So docstrings and descriptions here are written for the model, and notes for
developers go in # comments, which are not sent.

Schemas are replies, not state: the node checks a reply and converts it into
the state models in graph/state.py. Ids are never part of a schema; code
assigns them (PROMPTS.md, rule G-5).
"""

from pydantic import BaseModel, Field

from graph.state import Route

# ---------------------------------------------------------------------------
# P-01 Plan sub-questions (planner node, decision tier)
# ---------------------------------------------------------------------------


class PlannedQuestion(BaseModel):
    """One sub-question of the research plan."""

    # text comes before route, so the model writes the question first and
    # chooses where to look after it has decided what to ask.
    text: str = Field(description="One specific question that sources can answer.")
    route: Route = Field(
        description=(
            "document: the uploaded files. web: public sources. "
            "both: check one against the other."
        )
    )


class PlannerOutput(BaseModel):
    """The sub-questions that together cover the research topic."""

    sub_questions: list[PlannedQuestion]