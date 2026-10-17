"""Live check of the planner node against the real Mistral API.

CONCEPT: structured-output-planning. Runs only on request (`uv run pytest -m
live`). One real P-01 call per topic; prints the plan and how the reply fared,
for the "What we measured" section of the concept notes.
"""

import logging
import time

import pytest

from graph.llm import ModelGateway
from graph.nodes.planner import planner_node
from settings import settings

pytestmark = pytest.mark.live

LIVE_TOPICS = [
    "Due diligence on Symbotic: business model, customers, and main risks",
    "Should a mid-size retailer adopt warehouse automation in 2027?",
]


@pytest.mark.parametrize("topic", LIVE_TOPICS)
async def test_planner_makes_a_valid_plan_for_a_real_topic(topic, caplog):
    caplog.set_level(logging.INFO)
    started_at = time.perf_counter()

    planner_update = await planner_node(
        {"topic": topic},
        gateway=ModelGateway(),
        max_sub_questions=settings.vre_max_sub_questions,
    )

    elapsed_seconds = time.perf_counter() - started_at
    sub_questions = planner_update["sub_questions"]
    schema_retries = sum("did not fit" in record.message for record in caplog.records)
    cut_to_cap = any("keeping the first" in record.message for record in caplog.records)
    routes_overridden = [r.message for r in caplog.records if "routes set to web" in r.message]

    print(f"\nTopic: {topic}")
    for sub_question in sub_questions:
        print(f"  {sub_question.id} [{sub_question.route}] {sub_question.text}")
    print(
        f"  {len(sub_questions)} sub-questions in {elapsed_seconds:.1f} s; "
        f"schema retries: {schema_retries}; cut to cap: {cut_to_cap}; "
        f"route overrides: {routes_overridden or 'none'}"
    )

    assert 1 <= len(sub_questions) <= settings.vre_max_sub_questions
    assert [sub_question.id for sub_question in sub_questions] == [
        f"Q{position}" for position in range(1, len(sub_questions) + 1)
    ]
    assert all(sub_question.route == "web" for sub_question in sub_questions)
