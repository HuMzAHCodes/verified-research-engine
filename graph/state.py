"""Run state: what a research run keeps from one node to the next.

CONCEPT: sequential-graph (completed in concept 4). State models are what the
run holds, after code has checked them. They are separate from the output
schemas in graph/schemas.py, which are what one model call returns. A node is
the only place that converts one into the other.

Phase 1 starts with SubQuestion; RunState and the other models are added as
the nodes that write them are built (DATABASE_SCHEMA.md).
"""

from typing import Literal

from pydantic import BaseModel

# Where a sub-question is researched. Shared by the planner's schema and the
# run state, so the allowed values are written down once.
Route = Literal["document", "web", "both"]


class SubQuestion(BaseModel):
    """One sub-question as the run keeps it: checked, with an id from code."""

    id: str
    text: str
    route: Route