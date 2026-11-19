"""Tests of the run budget sized to the plan (D-58).

CONCEPT: subgraph-wrapper, choice C. The formula is checked against the two
plans it was agreed on, and the sizing against the run's budget object.
"""

from langchain_core.runnables import RunnableLambda

from graph.llm import MODEL_CALL_BUDGET_KEY, ModelCallBudget
from graph.run_budget import (
    SELF_RAG_WORST_CHAT_CALLS,
    SELF_RAG_WORST_EMBEDDINGS,
    fit_budget_to_plan,
    planned_request_budget,
)
from graph.state import SubQuestion


def plan_of(*routes: str) -> list[SubQuestion]:
    return [
        SubQuestion(id=f"Q{n}", text=f"Question {n}?", route=route)
        for n, route in enumerate(routes, start=1)
    ]


def test_self_rags_worst_path_follows_its_caps():
    # Two passes of P-15, P-10, P-11, P-12, P-14, plus the one revision (P-13).
    assert (SELF_RAG_WORST_CHAT_CALLS, SELF_RAG_WORST_EMBEDDINGS) == (11, 2)


def test_a_phase_3_plan_of_six_web_questions_gives_28():
    # (1 + 6 + 1 + 6) x 2: close to D-43's fixed 30.
    assert planned_request_budget(plan_of(*["web"] * 6)) == 28


def test_the_largest_plan_gives_172_under_the_ceiling_of_200():
    # (1 + 6 + 6 x 11 + 1 + 6) x 2 + 6 x 2.
    assert planned_request_budget(plan_of(*["both"] * 6)) == 172


def test_a_document_question_costs_more_than_a_web_one():
    assert planned_request_budget(plan_of("document")) == (1 + 11 + 1 + 1) * 2 + 2 == 30
    assert planned_request_budget(plan_of("web")) == (1 + 1 + 1 + 1) * 2 == 8


async def in_a_run(budget: ModelCallBudget, sub_questions: list[SubQuestion]):
    async def size(_: object):
        return fit_budget_to_plan(sub_questions)

    return await RunnableLambda(size).ainvoke(
        None, config={"configurable": {MODEL_CALL_BUDGET_KEY: budget}}
    )


async def test_the_budget_is_lowered_to_the_plan():
    budget = ModelCallBudget(limit=200)

    assert await in_a_run(budget, plan_of("web", "web")) == 12
    assert budget.limit == 12


async def test_the_budget_is_never_raised_above_where_the_run_started():
    # A ceiling set low (VRE_MAX_MODEL_CALLS_PER_RUN, or a test) still holds.
    budget = ModelCallBudget(limit=10)

    assert await in_a_run(budget, plan_of(*["both"] * 6)) == 10


def test_outside_a_run_there_is_no_budget_to_size():
    assert fit_budget_to_plan(plan_of("web")) is None
