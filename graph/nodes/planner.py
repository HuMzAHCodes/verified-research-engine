"""The planner node: turns the topic into checked, numbered sub-questions.

CONCEPT: structured-output-planning. The flow is always the same for a model
decision in this project:
1. build the prompt (graph/prompts/planner.py),
2. call the model through the gateway with the schema bound,
3. check the reply in code,
4. convert the reply into state, with ids assigned by code.

Requirements: FR-PLN-01 (structured plan), FR-PLN-03 (cap), FR-PLN-04
(web only when no documents were uploaded).

CONCEPT: subgraph-wrapper. From Phase 4 the planner is shown the documents'
file names, read from the store, not their ids (decision 7), and once the
plan is checked it sizes the run's request budget to the plan (D-58, choice
C): the plan is what decides how many branches of each kind will run.

Outside test review at the Phase 4 close: a document id that is not in the
store is refused before any model request (F-5), not found mid-run inside a
document branch after requests were spent; and a blank sub-question is
dropped (F-18), not sent to search as an empty query.
"""

import logging
from collections.abc import Mapping
from typing import Any

from graph.document_store import DocumentNotStored, DocumentStore
from graph.llm import ModelGateway
from graph.nodes.document_researcher import file_names_by_document_id
from graph.prompts import planner as planner_prompt
from graph.run_budget import fit_budget_to_plan
from graph.schemas import PlannedQuestion, PlannerOutput
from graph.state import SubQuestion

logger = logging.getLogger(__name__)


class PlannerError(RuntimeError):
    """The planner produced no usable sub-questions."""


async def planner_node(
    state: Mapping[str, Any],
    *,
    gateway: ModelGateway,
    max_sub_questions: int,
    document_store: DocumentStore | None = None,
) -> dict[str, list[SubQuestion]]:
    """Plan the run: one model call, then checks in code.

    The gateway, the cap, and the store are passed in by build_graph (D-25),
    so tests can pass a scripted gateway and a small cap. build_graph always
    passes the store; without one (node tests, the planner's live check) the
    ids stand in for the file names. The node returns only the field it
    changes, as every node does.
    """
    topic: str = state["topic"]
    document_ids: list[str] = list(state.get("document_ids", []))
    if document_store is not None:
        ensure_documents_are_stored(document_store, document_ids)
    # Decision 7: the names P-01 can judge a file by, not a hex id.
    uploaded_file_names = (
        list(file_names_by_document_id(document_store, document_ids).values())
        if document_store is not None
        else document_ids
    )
    preferences = state.get("preferences")
    report_format = getattr(preferences, "report_format", "") if preferences else ""

    planner_messages = planner_prompt.build_planner_messages(
        topic=topic,
        uploaded_file_names=uploaded_file_names,
        report_format=report_format,
        max_sub_questions=max_sub_questions,
    )
    planner_output = await gateway.structured(
        PlannerOutput,
        planner_messages,
        prompt_id=planner_prompt.PROMPT_ID,
        prompt_version=planner_prompt.VERSION,
        tier="decision",
    )

    checked_questions = check_planned_questions(
        planner_output.sub_questions,
        max_sub_questions=max_sub_questions,
        documents_uploaded=bool(uploaded_file_names),
    )
    sub_questions = assign_ids(checked_questions)
    # D-58: from here on, the run may send only what this plan can need.
    fit_budget_to_plan(sub_questions)
    return {"sub_questions": sub_questions}


def ensure_documents_are_stored(document_store: DocumentStore, document_ids: list[str]) -> None:
    """Raise DocumentNotStored, naming them, if any of the run's documents is not stored."""
    missing_ids = [
        document_id
        for document_id in dict.fromkeys(document_ids)
        if not document_store.has_document(document_id)
    ]
    if missing_ids:
        raise DocumentNotStored(
            f"not in the document store: {', '.join(missing_ids)}; ingest the file first"
        )


def check_planned_questions(
    planned_questions: list[PlannedQuestion],
    *,
    max_sub_questions: int,
    documents_uploaded: bool,
) -> list[PlannedQuestion]:
    """Enforce in code what the prompt only asks for.

    The model was told the cap and told whether files exist, but a prompt is
    a request. These checks make the rules hold whatever the model wrote.
    """
    non_blank_questions = [question for question in planned_questions if question.text.strip()]
    if len(non_blank_questions) < len(planned_questions):
        logger.warning(
            "planner returned %d blank sub-questions; dropped",
            len(planned_questions) - len(non_blank_questions),
        )
    planned_questions = non_blank_questions
    if not planned_questions:
        raise PlannerError("the planner returned no sub-questions")

    if len(planned_questions) > max_sub_questions:
        # Drop from the end: models usually list the most important questions
        # first. This is an assumption, checked by the evals in phase 10.
        logger.warning(
            "planner returned %d sub-questions; keeping the first %d",
            len(planned_questions),
            max_sub_questions,
        )
        planned_questions = planned_questions[:max_sub_questions]

    if not documents_uploaded:
        # With no files, "document" and "both" would send a researcher to an
        # empty store. Code knows for certain that no files exist; the model
        # only knows what the prompt said.
        overridden_count = sum(1 for question in planned_questions if question.route != "web")
        if overridden_count:
            logger.info("no documents uploaded; %d routes set to web", overridden_count)
        planned_questions = [
            question.model_copy(update={"route": "web"}) for question in planned_questions
        ]

    return planned_questions


def assign_ids(planned_questions: list[PlannedQuestion]) -> list[SubQuestion]:
    """Convert checked replies into state, numbering them Q1, Q2, ... in order.

    Ids come from code, never from the model: later stages cite by id, and ids
    made by position are unique and in order by construction (rule G-5).
    """
    return [
        SubQuestion(id=f"Q{position}", text=question.text.strip(), route=question.route)
        for position, question in enumerate(planned_questions, start=1)
    ]
