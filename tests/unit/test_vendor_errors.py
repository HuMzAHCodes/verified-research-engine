"""Every vendor's refusals, read one way; a used-up daily quota is never retried.

CONCEPT: chat-provider-setting, decisions 3 and 4 (D-66, D-67). Each vendor's
client raises a different error for the same refusal: Mistral an
httpx.HTTPStatusError, Groq its own APIStatusError holding the httpx reply,
Google a LangChain error raised from Google's ClientError. The errors here are
built the way each client builds them (Google's through LangChain's own
conversion), so the tests see what a real refusal looks like.
"""

import logging

import groq
import httpx
import pytest
from google.genai import errors as google_errors
from langchain_core.exceptions import ModelTimeoutError
from langchain_google_genai.chat_models import _handle_client_error, _handle_server_error
from pydantic import BaseModel

import graph.llm
from graph.llm import (
    MAX_RETRY_WAIT_SECONDS,
    ModelCallError,
    ModelGateway,
    daily_quota_used_up,
    http_status_of,
    is_transient_error,
    retry_after_seconds,
)
from tests.conftest import ScriptedModel, http_error

GEMINI_MODEL = "gemini-3.5-flash-lite"
QUOTA_FAILURE = "type.googleapis.com/google.rpc.QuotaFailure"
RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"
PER_MINUTE_QUOTA = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY_QUOTA = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
FREE_TIER_REQUESTS_METRIC = "generativelanguage.googleapis.com/generate_content_free_tier_requests"


def google_details(quota_id: str, retry_delay: str) -> list[dict]:
    """Google's details on a 429: which quota was hit, and how long to wait."""
    return [
        {
            "@type": QUOTA_FAILURE,
            "violations": [
                {
                    "quotaMetric": FREE_TIER_REQUESTS_METRIC,
                    "quotaId": quota_id,
                    "quotaDimensions": {"location": "global", "model": GEMINI_MODEL},
                    "quotaValue": "500" if "PerDay" in quota_id else "15",
                }
            ],
        },
        {"@type": RETRY_INFO, "retryDelay": retry_delay},
    ]


def google_refusal(status_code: int, status_name: str, details: list[dict]) -> Exception:
    """The error LangChain's Google client raises for a refused request."""
    reply_json = {
        "error": {
            "code": status_code,
            "message": "You exceeded your current quota.",
            "status": status_name,
            "details": details,
        }
    }
    request = httpx.Request(
        "POST", f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
    )
    response = httpx.Response(status_code, json=reply_json, request=request)
    google_error_class = (
        google_errors.ServerError if status_code >= 500 else google_errors.ClientError
    )
    google_error = google_error_class(status_code, reply_json, response)
    # LangChain's own conversion, so the wrapping is exactly the real one.
    try:
        if status_code >= 500:
            _handle_server_error(google_error)
        else:
            _handle_client_error(google_error, {"model": GEMINI_MODEL})
    except Exception as converted_error:
        return converted_error
    raise AssertionError("LangChain's conversion raised nothing")


def gemini_per_minute(retry_delay: str = "39s") -> Exception:
    return google_refusal(429, "RESOURCE_EXHAUSTED", google_details(PER_MINUTE_QUOTA, retry_delay))


def gemini_per_day() -> Exception:
    return google_refusal(429, "RESOURCE_EXHAUSTED", google_details(PER_DAY_QUOTA, "35000s"))


def groq_refusal(limit_words: str, headers: dict[str, str] | None = None) -> groq.RateLimitError:
    """Groq's 429, worded as Groq words it, with its reply attached."""
    message = (
        "Rate limit reached for model `openai/gpt-oss-20b` in organization `org_x` "
        f"service tier `on_demand` on {limit_words}: Limit 1000, Used 1000, Requested 1."
    )
    body = {"error": {"message": message, "type": "requests", "code": "rate_limit_exceeded"}}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers=headers or {}, json=body, request=request)
    return groq.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)


def groq_dropped_connection() -> groq.APIConnectionError:
    """Groq's connection error, raised from httpx's, as its client does."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    connection_error = groq.APIConnectionError(request=request)
    connection_error.__cause__ = httpx.ConnectError("connection reset", request=request)
    return connection_error


@pytest.mark.parametrize(
    "refusal",
    [http_error(429), groq_refusal("requests per minute (RPM)"), gemini_per_minute()],
    ids=["mistral", "groq", "google"],
)
def test_the_status_is_found_for_every_vendor(refusal):
    assert http_status_of(refusal) == 429
    assert is_transient_error(refusal)


def test_an_error_with_no_status_has_none():
    assert http_status_of(ValueError("not from a vendor")) is None
    assert not is_transient_error(ValueError("not from a vendor"))


def test_geminis_per_minute_quota_is_retried_after_the_wait_it_names():
    refusal = gemini_per_minute("39s")

    assert daily_quota_used_up(refusal) is None
    assert retry_after_seconds(refusal) == 39.0


def test_geminis_daily_quota_is_named_and_not_retried():
    refusal = gemini_per_day()

    assert daily_quota_used_up(refusal) == PER_DAY_QUOTA
    assert not is_transient_error(refusal)


@pytest.mark.parametrize(
    ("limit_words", "daily"),
    [
        ("requests per day (RPD)", True),
        ("tokens per day (TPD)", True),
        ("requests per minute (RPM)", False),
        ("tokens per minute (TPM)", False),
    ],
)
def test_groqs_daily_limits_are_not_retried_and_its_minute_limits_are(limit_words, daily):
    refusal = groq_refusal(limit_words)

    assert (daily_quota_used_up(refusal) is not None) == daily
    assert is_transient_error(refusal) == (not daily)


def test_groqs_retry_after_header_is_read_from_its_reply():
    assert retry_after_seconds(groq_refusal("requests per minute (RPM)", {"retry-after": "7"})) == 7


@pytest.mark.parametrize(
    ("refusal", "transient"),
    [
        (google_refusal(400, "INVALID_ARGUMENT", []), False),
        (google_refusal(403, "PERMISSION_DENIED", []), False),
        (google_refusal(503, "UNAVAILABLE", []), True),
        (groq_dropped_connection(), True),
        (ModelTimeoutError("the request timed out"), True),
        # Mistral's refusal at the Phase 5 opening: an account without API access.
        (http_error(402), False),
    ],
    ids=["google-400", "google-403", "google-503", "groq-connection", "timeout", "mistral-402"],
)
def test_what_a_wait_can_fix_is_the_same_for_every_vendor(refusal, transient):
    assert is_transient_error(refusal) == transient


class CapitalAnswer(BaseModel):
    country: str


CAPITAL = {"country": "Pakistan"}


@pytest.fixture
def recorded_waits(monkeypatch) -> list[float]:
    """The waits the gateway asks for before retries, recorded instead of slept."""
    waits: list[float] = []

    async def record_wait(wait_seconds: float) -> None:
        waits.append(wait_seconds)

    monkeypatch.setattr(graph.llm, "wait_before_retry", record_wait)
    return waits


def gateway_answering_with(decision_model: ScriptedModel) -> ModelGateway:
    """A gateway with the production waits, so the waits asked for can be checked."""
    return ModelGateway(
        decision_model=decision_model,
        writer_model=ScriptedModel(),
        requests_per_second=1000,
        retry_wait_seconds=2.0,
    )


async def ask(gateway: ModelGateway) -> CapitalAnswer:
    return await gateway.structured(CapitalAnswer, [], prompt_id="P-TEST", prompt_version="v1")


async def test_the_gateway_waits_the_delay_gemini_names_then_retries(recorded_waits):
    decision_model = ScriptedModel(scripted_replies=[gemini_per_minute("39s"), CAPITAL])

    answer = await ask(gateway_answering_with(decision_model))

    assert answer.country == "Pakistan"
    # Gemini's own wait, exactly: no random spread is added to a wait the vendor names.
    assert recorded_waits == [39.0]


async def test_a_very_long_named_wait_is_capped(recorded_waits):
    decision_model = ScriptedModel(scripted_replies=[gemini_per_minute("300s"), CAPITAL])

    await ask(gateway_answering_with(decision_model))

    assert recorded_waits == [MAX_RETRY_WAIT_SECONDS]


async def test_a_used_up_daily_quota_stops_at_once_and_says_why(recorded_waits, caplog):
    caplog.set_level(logging.WARNING)
    decision_model = ScriptedModel(scripted_replies=[gemini_per_day(), CAPITAL])

    with pytest.raises(ModelCallError) as raised:
        await ask(gateway_answering_with(decision_model))

    message = str(raised.value)
    assert "the daily quota is used up" in message
    assert PER_DAY_QUOTA in message
    # It names the tier and model, and what to do about it.
    assert "(decision, " in message
    assert "VRE_CHAT_PROVIDER" in message
    # One request, no wait, no retry line in the log.
    assert len(decision_model.received_conversations) == 1
    assert recorded_waits == []
    assert "transient error" not in caplog.text


async def test_groqs_daily_limit_stops_at_once_too(recorded_waits):
    decision_model = ScriptedModel(scripted_replies=[groq_refusal("requests per day (RPD)")])

    with pytest.raises(ModelCallError, match="the daily quota is used up"):
        await ask(gateway_answering_with(decision_model))

    assert recorded_waits == []


async def test_mistrals_402_still_fails_at_once(recorded_waits):
    decision_model = ScriptedModel(scripted_replies=[http_error(402), CAPITAL])

    with pytest.raises(ModelCallError):
        await ask(gateway_answering_with(decision_model))

    assert len(decision_model.received_conversations) == 1
    assert recorded_waits == []
