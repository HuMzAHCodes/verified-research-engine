"""Run state: what a research run keeps from one node to the next.

CONCEPT: sequential-graph. LangGraph keeps one state object per run. Every
node receives the whole state and returns only the fields it changed;
LangGraph merges that partial update into the state. How a field is merged
is decided by its reducer: without one, the new value replaces the old.

State models are what the run holds, after code has checked them. They are
separate from the output schemas in graph/schemas.py, which are what one model
call returns. A node is the only place that converts one into the other.

Fields are added as the nodes that write them are built (DATABASE_SCHEMA.md):
phase 1 has the topic, the sub-questions, the findings, and the draft; phase 3
adds search_failures, and WebResearchTask, the input of one parallel branch.
"""

import operator
from typing import Annotated, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict

# Where a sub-question is researched. Shared by the planner's schema and the
# run state, so the allowed values are written down once.
Route = Literal["document", "web", "both"]


class SubQuestion(BaseModel):
    """One sub-question as the run keeps it: checked, with an id from code."""

    id: str
    text: str
    route: Route


class Finding(BaseModel):
    """One piece of evidence for one sub-question.

    The id is numbered within its sub-question (Q2-F1, Q2-F2), so researchers
    running in parallel (phase 3) never need a shared counter. found=False
    records that a sub-question was searched and nothing came back.
    """

    id: str
    sub_question_id: str
    found: bool
    source_type: Literal["document", "web"]
    locator: str = ""  # the url for web findings; file and page for documents
    title: str = ""  # the page or document title, shown to the writer
    passage: str = ""  # the source text, cut to VRE_PASSAGE_MAX_CHARS
    answer: str = ""  # a one-sentence answer; written by P-02 (phase 2)


class Section(BaseModel):
    """One section of the draft brief."""

    title: str
    text: str


class RunState(TypedDict, total=False):
    """Everything one research run holds.

    total=False: a field is absent until the node that writes it has run.
    """

    topic: str
    document_ids: list[str]
    sub_questions: list[SubQuestion]
    # CONCEPT: sequential-graph, reducer. operator.add appends each node's
    # list to the existing one instead of replacing it. A node must therefore
    # return only new findings, never the whole list again, or every finding
    # would be doubled.
    # CONCEPT: parallel-fan-out. Several web_researcher branches write this
    # field in the same step. With the reducer their lists are appended, in the
    # order of the Send list (the planner's order), whatever order the branches
    # finish in; without one, LangGraph would refuse two values in one step.
    findings: Annotated[list[Finding], operator.add]
    # CONCEPT: parallel-fan-out, decision 3. Why a sub-question's search
    # failed, one line each ("Q3: search timed out after 15 s"). Each branch
    # sees only its own search, so the D-40 check ("every search failed")
    # runs after all of them, in research_check, by counting these lines.
    search_failures: Annotated[list[str], operator.add]
    draft: list[Section]


class WebResearchTask(TypedDict):
    """What one web_researcher branch receives: one sub-question, nothing else.

    CONCEPT: parallel-fan-out. A Send carries its own input to the node it
    starts, instead of the whole run state. Giving a branch only its
    sub-question means it cannot read or depend on another branch's work,
    which is what makes running them at the same time safe.
    """

    sub_question: SubQuestion
