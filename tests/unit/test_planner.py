"""Scripted tests of the planner node.

CONCEPT: structured-output-planning, tested with scripted-model-testing.
Each test scripts the planner's reply and checks what the node puts in state.
"""

import pytest

from graph.nodes.planner import PlannerError, planner_node
from graph.state import SubQuestion
from tests.conftest import ScriptedModel

TOPIC = "Due diligence on Symbotic"


def planned(*routes: str) -> dict:
    """A scripted P-01 reply with one sub-question per route given."""
    return {
        "sub_questions": [
            {"text": f"Question {number}?", "route": route}
            for number, route in enumerate(routes, start=1)
        ]
    }


async def run_planner(planner_model, make_gateway, state=None, max_sub_questions=6):
    return await planner_node(
        state or {"topic": TOPIC},
        gateway=make_gateway(decision_model=planner_model),
        max_sub_questions=max_sub_questions,
    )


async def test_tc_pln_01_planner_output_becomes_sub_questions_with_ids(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[planned("web", "web", "web")])

    planner_update = await run_planner(planner_model, make_gateway)

    assert planner_update == {
        "sub_questions": [
            SubQuestion(id="Q1", text="Question 1?", route="web"),
            SubQuestion(id="Q2", text="Question 2?", route="web"),
            SubQuestion(id="Q3", text="Question 3?", route="web"),
        ]
    }
    assert planner_model.scripted_replies == []  # exactly one model call


async def test_prompt_states_the_topic_the_cap_and_no_files(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[planned("web")])

    await run_planner(planner_model, make_gateway, max_sub_questions=4)

    system_message, human_message = planner_model.received_conversations[0]
    assert "at most 4 sub-questions" in system_message.content
    assert f"Topic: {TOPIC}" in human_message.content
    assert "Uploaded files: none" in human_message.content


async def test_tc_pln_03_extra_sub_questions_are_cut_from_the_end(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[planned(*["web"] * 8)])

    planner_update = await run_planner(planner_model, make_gateway, max_sub_questions=6)

    kept_ids = [sub_question.id for sub_question in planner_update["sub_questions"]]
    assert kept_ids == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert planner_update["sub_questions"][-1].text == "Question 6?"


async def test_tc_pln_04_routes_are_forced_to_web_without_documents(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[planned("document", "both", "web")])

    planner_update = await run_planner(planner_model, make_gateway)

    assert [sub_question.route for sub_question in planner_update["sub_questions"]] == [
        "web",
        "web",
        "web",
    ]


async def test_routes_are_kept_when_documents_were_uploaded(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[planned("document", "both", "web")])
    state_with_documents = {"topic": TOPIC, "document_ids": ["annual-report.pdf"]}

    planner_update = await run_planner(planner_model, make_gateway, state=state_with_documents)

    assert [sub_question.route for sub_question in planner_update["sub_questions"]] == [
        "document",
        "both",
        "web",
    ]


async def test_an_empty_plan_fails_the_node(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[{"sub_questions": []}])

    with pytest.raises(PlannerError, match="no sub-questions"):
        await run_planner(planner_model, make_gateway)
