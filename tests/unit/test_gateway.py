"""Scripted tests of the model gateway: every path, no network.

CONCEPT: model-gateway, tested with scripted-model-testing. Each test scripts
the model's replies and checks what the gateway returns, raises, and sends.

CONCEPT: shared-throttle. TC-RES-05 (rate-limit responses): a 429 is retried
after a growing, randomised wait long enough to reach the next minute (D-55),
or the wait the provider asks for, and gives up after three attempts; other
transient failures keep the short waits. The provider's limit headers reach
the log and the final error. Each tier's model has its own limiter, shared by
every gateway in the process, and calls on a tier are spaced by its rate. A
run's budget counts every request it sends, retries included.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from itertools import pairwise
from typing import Literal

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

import graph.llm
from graph.llm import (
    MAX_RETRY_WAIT_SECONDS,
    MODEL_CALL_BUDGET_KEY,
    RATE_LIMIT_WAIT_SECONDS,
    ModelCallBudget,
    ModelCallBudgetExceeded,
    ModelCallError,
    ModelGateway,
    current_call_budget,
    rate_limit_report,
    retry_after_seconds,
)
from settings import settings
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


# --- shared-throttle: TC-RES-05, the limiters, and the call budget ---------


def rate_limit_with_retry_after(header_value: str) -> httpx.HTTPStatusError:
    """A 429 whose response says how long to wait."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"Retry-After": header_value})
    return httpx.HTTPStatusError("Error response 429", request=request, response=response)


@pytest.fixture
def recorded_waits(monkeypatch) -> list[float]:
    """The waits the gateway asks for before retries, recorded instead of slept."""
    waits: list[float] = []

    async def record_wait(wait_seconds: float) -> None:
        waits.append(wait_seconds)

    monkeypatch.setattr(graph.llm, "wait_before_retry", record_wait)
    return waits


def rate_limit_with_limit_headers() -> httpx.HTTPStatusError:
    """A 429 as Mistral sent it in Phase 4: no Retry-After, only its limit headers."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={
            "x-ratelimit-limit-req-minute": "2",
            "x-ratelimit-remaining-req-minute": "0",
            "content-type": "application/json",
        },
    )
    return httpx.HTTPStatusError("Error response 429", request=request, response=response)


def gateway_with_real_waits(planner_model: ScriptedModel) -> ModelGateway:
    """Like make_gateway, but with the production waits: 2 s, and 20 s after a 429."""
    return ModelGateway(
        decision_model=planner_model,
        writer_model=ScriptedModel(),
        requests_per_second=1000,
        retry_wait_seconds=2.0,
        rate_limit_wait_seconds=RATE_LIMIT_WAIT_SECONDS,
    )


async def test_tc_res_05_a_rate_limit_waits_long_enough_to_reach_the_next_minute(
    recorded_waits, caplog
):
    caplog.set_level(logging.WARNING)
    planner_model = ScriptedModel(
        scripted_replies=[rate_limit_error(), rate_limit_error(), VALID_PLAN]
    )

    planner_output = await call_planner(gateway_with_real_waits(planner_model))

    assert planner_output.sub_questions[0].route == "web"
    first_wait, second_wait = recorded_waits
    # D-55: 20 s, then 40 s, each with up to half as much again added at
    # random (the second capped at 60 s). Together at least a whole minute,
    # so the last attempt falls in a new minute; 2 s and 4 s never did.
    assert 20.0 <= first_wait <= 30.0
    assert 40.0 <= second_wait <= MAX_RETRY_WAIT_SECONDS
    assert first_wait + second_wait >= 60.0
    # Each retry names the tier and the model (shared-throttle, decision 4).
    assert f"P-TEST (decision, {settings.vre_chat_model}): transient error" in caplog.text


async def test_a_server_error_keeps_the_short_growing_waits(recorded_waits):
    planner_model = ScriptedModel(scripted_replies=[http_error(503), http_error(502), VALID_PLAN])

    await call_planner(gateway_with_real_waits(planner_model))

    first_wait, second_wait = recorded_waits
    # A 5xx is not a limit counted per minute: 2 s, then 4 s, as in D-43.
    assert 2.0 <= first_wait <= 3.0
    assert 4.0 <= second_wait <= 6.0


async def test_the_providers_limit_headers_reach_the_log_and_the_final_error(
    recorded_waits, caplog
):
    caplog.set_level(logging.WARNING)
    planner_model = ScriptedModel(scripted_replies=[rate_limit_with_limit_headers()] * 3)

    with pytest.raises(ModelCallError) as raised:
        await call_planner(gateway_with_real_waits(planner_model))

    reported_limits = "[limits: limit-req-minute=2, remaining-req-minute=0]"
    assert reported_limits in caplog.text
    assert reported_limits in str(raised.value)


def test_a_limit_report_is_empty_without_limit_headers():
    assert rate_limit_report(rate_limit_error()) == ""
    assert rate_limit_report(httpx.ConnectError("connection reset")) == ""


async def test_tc_res_05_a_retry_after_header_is_followed_and_capped(recorded_waits):
    planner_model = ScriptedModel(
        scripted_replies=[
            rate_limit_with_retry_after("7"),
            rate_limit_with_retry_after("120"),
            VALID_PLAN,
        ]
    )

    await call_planner(gateway_with_real_waits(planner_model))

    assert recorded_waits == [7.0, MAX_RETRY_WAIT_SECONDS]


def test_a_retry_after_date_is_read_as_seconds_from_now():
    in_ten_seconds = format_datetime(datetime.now(UTC) + timedelta(seconds=10), usegmt=True)

    assert 8.0 <= retry_after_seconds(rate_limit_with_retry_after(in_ten_seconds)) <= 10.0
    assert retry_after_seconds(rate_limit_with_retry_after("soon")) is None
    assert retry_after_seconds(rate_limit_error()) is None  # no header: the gateway's own waits


async def test_tc_res_05_three_rate_limits_stop_with_a_clear_error(recorded_waits):
    planner_model = ScriptedModel(scripted_replies=[rate_limit_error()] * 3)

    with pytest.raises(ModelCallError, match="failed after 3 attempts"):
        await call_planner(gateway_with_real_waits(planner_model))

    assert len(recorded_waits) == 2  # no wait after the last attempt


class TimedScriptedModel(ScriptedModel):
    """A ScriptedModel that records when each request reaches it."""

    request_times: list[float] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.request_times.append(time.perf_counter())
        return super()._generate(messages, stop, run_manager, **kwargs)


async def test_calls_arriving_together_are_released_one_interval_apart():
    # Five requests a second: one every 0.2 s. Checked by the gaps between
    # requests, not by total time (parallel-fan-out, "What broke").
    planner_model = TimedScriptedModel(scripted_replies=[VALID_PLAN] * 4)
    gateway = ModelGateway(
        decision_model=planner_model,
        writer_model=ScriptedModel(),
        requests_per_second=5,
        retry_wait_seconds=0,
    )

    await asyncio.gather(*(call_planner(gateway) for _ in range(4)))

    request_times = sorted(planner_model.request_times)
    gaps = [later - earlier for earlier, later in pairwise(request_times)]
    assert len(gaps) == 3
    assert all(gap >= 0.15 for gap in gaps)


def gateway_limiters(monkeypatch, decision_model: str, writer_model: str):
    """The limiters of two gateways built with these model names, from an empty registry."""
    monkeypatch.setattr(graph.llm, "_shared_rate_limiters", {})
    monkeypatch.setattr(settings, "vre_chat_model", decision_model)
    monkeypatch.setattr(settings, "vre_writer_model", writer_model)
    first_gateway = ModelGateway(decision_model=ScriptedModel(), writer_model=ScriptedModel())
    second_gateway = ModelGateway(decision_model=ScriptedModel(), writer_model=ScriptedModel())
    return first_gateway.rate_limiters_by_tier, second_gateway.rate_limiters_by_tier


def test_tiers_on_different_models_have_their_own_limiters_shared_by_every_gateway(monkeypatch):
    first_limiters, second_limiters = gateway_limiters(monkeypatch, "model-a", "model-b")

    # Decision 1: two models, two limits, so one tier cannot slow the other.
    assert first_limiters["decision"] is not first_limiters["writer"]
    # Decision 2: a second gateway takes tokens from the same buckets.
    assert first_limiters["decision"] is second_limiters["decision"]
    assert first_limiters["writer"] is second_limiters["writer"]
    # Choice A: each at its tier's rate from settings (80% of the measured limit).
    assert first_limiters["decision"].requests_per_second == settings.vre_decision_rps
    assert first_limiters["writer"].requests_per_second == settings.vre_writer_rps


def test_tiers_on_the_same_model_share_its_one_limiter(monkeypatch):
    # D-55: both tiers run on one model, and the provider counts its limit
    # once, so both tiers must take from one bucket, not two at the full rate.
    first_limiters, second_limiters = gateway_limiters(
        monkeypatch, "voxtral-small-2507", "voxtral-small-2507"
    )

    assert first_limiters["decision"] is first_limiters["writer"]
    assert first_limiters["writer"] is second_limiters["decision"]


async def call_planner_in_a_run(gateway, budget: ModelCallBudget) -> PlannerOutput:
    """Call the gateway the way a node does: inside a runnable with the run's config."""

    async def node_like_step(_: object) -> PlannerOutput:
        return await call_planner(gateway)

    in_a_run = RunnableLambda(node_like_step)
    return await in_a_run.ainvoke(None, config={"configurable": {MODEL_CALL_BUDGET_KEY: budget}})


async def test_the_run_budget_counts_every_request_and_stops_the_run(make_gateway):
    # The first call needs its schema retry: two requests, the whole budget.
    planner_model = ScriptedModel(scripted_replies=[PLAN_WITH_BAD_ROUTE, VALID_PLAN, VALID_PLAN])
    gateway = make_gateway(decision_model=planner_model)
    budget = ModelCallBudget(limit=2)

    await call_planner_in_a_run(gateway, budget)
    assert budget.requests_sent == 2

    with pytest.raises(ModelCallBudgetExceeded, match="VRE_MAX_MODEL_CALLS_PER_RUN=2"):
        await call_planner_in_a_run(gateway, budget)
    assert planner_model.scripted_replies == [VALID_PLAN]  # the third request was never sent


async def test_a_transport_retry_counts_against_the_budget(make_gateway):
    planner_model = ScriptedModel(scripted_replies=[rate_limit_error(), VALID_PLAN])
    budget = ModelCallBudget(limit=1)

    with pytest.raises(ModelCallBudgetExceeded):
        await call_planner_in_a_run(make_gateway(decision_model=planner_model), budget)


def test_a_call_outside_a_run_has_no_budget():
    assert current_call_budget() is None
