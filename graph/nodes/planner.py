"""The planner node: turns the topic into checked, numbered sub-questions.

CONCEPT: structured-output-planning. The flow is always the same for a model
decision in this project:
1. build the prompt (graph/prompts/planner.py),
2. call the model through the gateway with the schema bound,
3. check the reply in code,
4. convert the reply into state, with ids assigned by code.

Requirements: FR-PLN-01 (structured plan), FR-PLN-03 (cap), FR-PLN-04
(web only when no documents were uploaded).
"""

import logging
from collections.abc import Mapping
from typing import Any

from graph.llm import ModelGateway
from graph.prompts import planner as planner_prompt
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
) -> dict[str, list[SubQuestion]]:
    """Plan the run: one model call, then checks in code.

    The gateway and the cap are passed in by build_graph (D-25), so tests can
    pass a scripted gateway and a small cap. The node returns only the field
    it changes, as every node does.
    """
    topic: str = state["topic"]
    # Phase 4 resolves document ids to file names from the documents table;
    # until then the ids stand in for the names. In phase 1 the list is empty.
    uploaded_file_names: list[str] = list(state.get("document_ids", []))
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
    return {"sub_questions": assign_ids(checked_questions)}


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
