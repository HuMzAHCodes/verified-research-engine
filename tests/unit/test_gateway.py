"""Scripted tests of the model gateway: every path, no network.

CONCEPT: model-gateway, tested with scripted-model-testing. Each test scripts
the model's replies and checks what the gateway returns, raises, and sends.
"""

from typing import Literal

import pytest
from pydantic import BaseModel

from graph.llm import ModelCallError
from tests.conftest import ScriptedModel, http_error, rate_limit_error


class PlannedQuestion(BaseModel):
    text: str
    route: Literal["document", "web", "both"]


class PlannerOutput(BaseModel):
    sub_questions: list[PlannedQuestion]


SUB_QUESTION_TEXT = "What does the company sell?"
VALID_PLAN = {"sub_questions": [{"text": SUB_QUESTION_TEXT, "route": "web"}]}
# "internet" is not one of the allowed routes, so this reply fails validation.
PLAN_WITH_BAD_ROUTE = {"sub_questions": [{"text": SUB_QUESTION_TEXT, "route": "internet"}]}


async def call_planner(gateway) -> PlannerOutput:
    return await gateway.structured(PlannerOutput, [], prompt_id="P-TEST", prompt_version="v1")


async def test_valid_reply_is_returned_as_the_schema(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[VALID_PLAN])

    planner_output = await call_planner(make_gateway(decision_model=planner_model))

    assert planner_output == PlannerOutput.model_validate(VALID_PLAN)
    assert len(planner_model.received_conversations) == 1
    assert planner_model.scripted_replies == []


async def test_reply_that_does_not_fit_is_retried_once_with_the_error(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[PLAN_WITH_BAD_ROUTE, VALID_PLAN])

    planner_output = await call_planner(make_gateway(decision_model=planner_model))

    assert planner_output.sub_questions[0].route == "web"
    assert len(planner_model.received_conversations) == 2
    repair_message = planner_model.received_conversations[1][-1].content
    assert "did not fit the required PlannerOutput format" in repair_message
    assert "internet" in repair_message  # the model is shown its own bad value


async def test_two_replies_that_do_not_fit_raise(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[PLAN_WITH_BAD_ROUTE, PLAN_WITH_BAD_ROUTE])

    with pytest.raises(ModelCallError, match="after 2 attempts"):
        await call_planner(make_gateway(decision_model=planner_model))

    assert len(planner_model.received_conversations) == 2


async def test_text_instead_of_a_tool_call_is_retried(make_gateway):
    planner_model = ScriptedModel(scripted_replies=["Here are some questions...", VALID_PLAN])

    planner_output = await call_planner(make_gateway(decision_model=planner_model))

    assert planner_output.sub_questions[0].route == "web"
    repair_message = planner_model.received_conversations[1][-1].content
    assert "answered in text" in repair_message


async def test_rate_limit_is_retried(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[rate_limit_error(), VALID_PLAN])

    planner_output = await call_planner(make_gateway(decision_model=planner_model))

    assert planner_output.sub_questions[0].route == "web"
    assert len(planner_model.received_conversations) == 2


async def test_bad_key_fails_at_once(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[http_error(401), VALID_PLAN])

    with pytest.raises(ModelCallError, match="401"):
        await call_planner(make_gateway(decision_model=planner_model))

    assert len(planner_model.received_conversations) == 1
    assert planner_model.scripted_replies == [VALID_PLAN]  # never asked again
