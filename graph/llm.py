"""Model gateway: the only place that builds chat models and calls them.

CONCEPT: model-gateway. Every model decision in the graph goes through
ModelGateway.structured(), which applies, in this order:

1. the call budget           the run's ModelCallBudget: one more request, or stop
2. the throttle              the limiter for the tier's model, shared by the
                             whole process (shared-throttle)
3. the transport retry       429, 5xx, timeouts, dropped connections; 3 tries,
                             waiting as the provider asks or with a random spread
4. the schema retry          a reply that does not fit the schema gets one
                             more try, with the validation error shown
5. the trace tags            prompt id and version on every LangSmith run

Nodes never import ChatMistralAI. They receive a gateway, so tests can pass
one that wraps a scripted fake model instead of the real API.
"""

import asyncio
import logging
import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, TypeVar

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel

from settings import settings

logger = logging.getLogger(__name__)

# Which model a prompt uses. Small decision schemas run on the 8b model; the
# writer's long, structured answers run on the 14b model (D-24).
ModelTier = Literal["decision", "writer"]

# The schema class passed in is the type that comes back out.
OutputSchema = TypeVar("OutputSchema", bound=BaseModel)

TRANSPORT_ATTEMPTS = 3
SCHEMA_ATTEMPTS = 2
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REQUEST_TIMEOUT_SECONDS = 60
# The longest wait before a retry, whatever a Retry-After header says, so a
# wrong or very large value cannot stall a run (shared-throttle, choice B).
MAX_RETRY_WAIT_SECONDS = 30.0
# Where a run keeps its ModelCallBudget in its config (shared-throttle, choice C).
MODEL_CALL_BUDGET_KEY = "model_call_budget"


class ModelCallError(RuntimeError):
    """A model call failed for good: retries used up, or an error not worth retrying."""


class ModelCallBudgetExceeded(ModelCallError):
    """A run tried to send more model requests than its budget allows."""


class ModelCallBudget:
    """How many model requests one run may still send.

    CONCEPT: shared-throttle, choice C (D-43). The budget belongs to a run, so
    it travels in the run's config (research_run_config puts a fresh one there)
    rather than living in the gateway, which may serve many runs from Phase 9.
    LangGraph hands a run's config to every node, including every parallel
    branch, and they all see this same object, so the count covers the whole
    run. Every request counts, retries included, because the provider counts
    them all.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.requests_sent = 0

    def spend(self, prompt_id: str) -> None:
        """Count one more request, or stop the run if the budget is used up."""
        if self.requests_sent >= self.limit:
            raise ModelCallBudgetExceeded(
                f"{prompt_id}: this run has already sent {self.requests_sent} model requests, "
                f"the most it may send (VRE_MAX_MODEL_CALLS_PER_RUN={self.limit}); "
                "stopping instead of sending more"
            )
        self.requests_sent += 1


def current_call_budget() -> ModelCallBudget | None:
    """The budget of the run this call belongs to, or None outside a run.

    ensure_config() returns the config of the runnable currently executing:
    inside a graph run, the run's own config. A call made outside a run (a unit
    test of the gateway, a live check) has no budget and is not limited.
    """
    budget = ensure_config().get("configurable", {}).get(MODEL_CALL_BUDGET_KEY)
    return budget if isinstance(budget, ModelCallBudget) else None


# CONCEPT: shared-throttle, decision 2. One limiter per model name for the
# whole process. The provider's limit belongs to the account, so every
# gateway that calls the same model must take tokens from the same bucket;
# limiters inside each gateway would each believe they had the whole limit.
_shared_rate_limiters: dict[str, InMemoryRateLimiter] = {}


def build_rate_limiter(requests_per_second: float) -> InMemoryRateLimiter:
    """A token bucket that lets one request through every 1/rate seconds.

    max_bucket_size=1 means no bursts: tokens do not pile up while idle, so
    six calls arriving together are released one interval apart.
    """
    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def shared_rate_limiter(model_name: str, requests_per_second: float) -> InMemoryRateLimiter:
    """The process's limiter for this model, created at this rate the first time."""
    if model_name not in _shared_rate_limiters:
        _shared_rate_limiters[model_name] = build_rate_limiter(requests_per_second)
    return _shared_rate_limiters[model_name]


def is_transient_error(error: BaseException) -> bool:
    """True for failures that may succeed if the same request is sent again.

    A rate limit (429) or a server error (5xx) usually clears after a wait.
    A timeout or a dropped connection (WinError 10054) is a network blip.
    A 400 or 401 means a bad request or a bad key: sending it again cannot
    help, so those fail at once.
    """
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(error, (httpx.RequestError, ConnectionError))


def build_mistral_model(model_name: str) -> ChatMistralAI:
    """A Mistral chat model with project-wide settings.

    temperature=0 so the same prompt gives the same answer, which makes runs
    comparable across prompt versions. max_retries=1 switches off the client's
    own retry (its count is total attempts, so 1 means a single try): the
    gateway's retry is the only retry layer, so waits do not multiply and
    every retry is logged in one place.
    """
    return ChatMistralAI(
        model=model_name,
        temperature=0,
        api_key=settings.mistral_api_key,
        max_retries=1,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


class ModelGateway:
    """Owns the chat models and is the only way to call them."""

    def __init__(
        self,
        decision_model: BaseChatModel | None = None,
        writer_model: BaseChatModel | None = None,
        requests_per_second: float | None = None,
        retry_wait_seconds: float = 2.0,
    ) -> None:
        # Tests pass scripted models here; in normal use both are built from
        # settings.
        self.models_by_tier: dict[ModelTier, BaseChatModel] = {
            "decision": decision_model or build_mistral_model(settings.vre_chat_model),
            "writer": writer_model or build_mistral_model(settings.vre_writer_model),
        }
        self.model_names_by_tier: dict[ModelTier, str] = {
            "decision": settings.vre_chat_model,
            "writer": settings.vre_writer_model,
        }
        # CONCEPT: shared-throttle, decisions 1 and 2 (D-43). Each tier's
        # model has its own limiter, at that tier's rate, taken from the
        # process registry, so every gateway calling a model shares one
        # bucket. An explicit rate (the tests) gets private limiters instead,
        # so tests never wait for one another.
        if requests_per_second is not None:
            self.rate_limiters_by_tier: dict[ModelTier, InMemoryRateLimiter] = {
                "decision": build_rate_limiter(requests_per_second),
                "writer": build_rate_limiter(requests_per_second),
            }
        else:
            self.rate_limiters_by_tier = {
                "decision": shared_rate_limiter(settings.vre_chat_model, settings.vre_decision_rps),
                "writer": shared_rate_limiter(settings.vre_writer_model, settings.vre_writer_rps),
            }
        self.retry_wait_seconds = retry_wait_seconds

    async def structured(
        self,
        schema: type[OutputSchema],
        messages: list[BaseMessage],
        *,
        prompt_id: str,
        prompt_version: str,
        tier: ModelTier = "decision",
    ) -> OutputSchema:
        """Call the model and return a validated instance of `schema`.

        CONCEPT: structured-output-planning. include_raw=True makes LangChain
        return {"raw", "parsed", "parsing_error"} instead of raising when the
        reply does not fit, so the error text can be sent back for one retry.
        """
        structured_model = self.models_by_tier[tier].with_structured_output(
            schema, include_raw=True
        )
        # CONCEPT: tracing-from-day-one. The run name and metadata appear on
        # the LangSmith trace, so every score can be tied to a prompt version.
        run_config: RunnableConfig = {
            "run_name": prompt_id,
            "tags": [prompt_id, f"tier:{tier}"],
            "metadata": {"prompt_id": prompt_id, "prompt_version": prompt_version},
        }

        conversation = list(messages)
        for schema_attempt in range(1, SCHEMA_ATTEMPTS + 1):
            reply = await self._invoke_with_transport_retry(
                structured_model, conversation, run_config, prompt_id, tier
            )
            parsed_output = reply.get("parsed")
            if parsed_output is not None:
                return parsed_output

            failure_reason = describe_schema_failure(reply)
            logger.warning(
                "%s: reply did not fit %s (attempt %d of %d): %s",
                prompt_id,
                schema.__name__,
                schema_attempt,
                SCHEMA_ATTEMPTS,
                failure_reason,
            )
            if schema_attempt == SCHEMA_ATTEMPTS:
                raise ModelCallError(
                    f"{prompt_id}: reply did not fit {schema.__name__} after "
                    f"{SCHEMA_ATTEMPTS} attempts: {failure_reason}"
                )
            # The repair message is a new human turn rather than a copy of the
            # model's bad tool call: an assistant tool call must be followed
            # by a tool result, and there is no tool result to give.
            conversation = [*messages, HumanMessage(content=repair_instruction(schema, reply))]

        raise AssertionError("unreachable")  # the loop always returns or raises

    async def _invoke_with_transport_retry(
        self,
        structured_model: Any,
        conversation: list[BaseMessage],
        run_config: RunnableConfig,
        prompt_id: str,
        tier: ModelTier,
    ) -> dict[str, Any]:
        """Send one request, retrying only failures that a wait can fix.

        Every attempt first takes one request from the run's budget, then a
        token from its tier's limiter: a retry costs as much as a first try.
        """
        call_budget = current_call_budget()
        rate_limiter = self.rate_limiters_by_tier[tier]
        for transport_attempt in range(1, TRANSPORT_ATTEMPTS + 1):
            if call_budget is not None:
                call_budget.spend(prompt_id)
            await rate_limiter.aacquire()
            try:
                return await structured_model.ainvoke(conversation, config=run_config)
            except Exception as error:
                if not is_transient_error(error):
                    raise ModelCallError(f"{prompt_id}: {error}") from error
                if transport_attempt == TRANSPORT_ATTEMPTS:
                    raise ModelCallError(
                        f"{prompt_id}: failed after {TRANSPORT_ATTEMPTS} attempts: {error}"
                    ) from error
                wait_seconds = retry_wait_seconds(error, transport_attempt, self.retry_wait_seconds)
                # CONCEPT: shared-throttle, decision 4. The tier and the model
                # are named, so a run's retries show which limit was reached.
                logger.warning(
                    "%s (%s, %s): transient error (attempt %d of %d), retrying in %.1f s: %s",
                    prompt_id,
                    tier,
                    self.model_names_by_tier[tier],
                    transport_attempt,
                    TRANSPORT_ATTEMPTS,
                    wait_seconds,
                    error,
                )
                await wait_before_retry(wait_seconds)
        raise AssertionError("unreachable")


def retry_wait_seconds(error: BaseException, attempt: int, base_wait_seconds: float) -> float:
    """How long to wait before the next attempt (shared-throttle, choice B).

    If the provider said how long to wait (Retry-After), wait that long. If not,
    wait base, then twice base, each with up to half as much again added at
    random, so branches refused together do not all come back together. Either
    way, never more than MAX_RETRY_WAIT_SECONDS.
    """
    provider_wait = retry_after_seconds(error)
    if provider_wait is not None:
        return min(provider_wait, MAX_RETRY_WAIT_SECONDS)
    growing_wait = base_wait_seconds * 2 ** (attempt - 1)
    return min(growing_wait + random.uniform(0, growing_wait / 2), MAX_RETRY_WAIT_SECONDS)


def retry_after_seconds(error: BaseException) -> float | None:
    """The wait a Retry-After header asks for, in seconds, if the error has one.

    The header holds either a number of seconds ("2") or a date. Whether the
    provider sends it is not assumed: no header, or one that cannot be read,
    gives None, and the caller falls back to its own waits.
    """
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    header_value = error.response.headers.get("retry-after")
    if header_value is None:
        return None
    try:
        return max(0.0, float(header_value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


async def wait_before_retry(wait_seconds: float) -> None:
    """Sleep before a retry; a function of its own so tests can record waits."""
    await asyncio.sleep(wait_seconds)


def describe_schema_failure(reply: dict[str, Any]) -> str:
    """Why a structured reply has no parsed value, in one line."""
    parsing_error = reply.get("parsing_error")
    if parsing_error is not None:
        return str(parsing_error)
    return "the model answered in text instead of calling the output tool"


def repair_instruction(schema: type[BaseModel], reply: dict[str, Any]) -> str:
    """The extra message sent with the one schema retry."""
    raw_message = reply.get("raw")
    previous_answer = ""
    if raw_message is not None:
        tool_calls = getattr(raw_message, "tool_calls", None) or []
        previous_answer = str(tool_calls[0]["args"]) if tool_calls else str(raw_message.content)
    return (
        f"Your previous answer did not fit the required {schema.__name__} format.\n"
        f"Problem: {describe_schema_failure(reply)}\n"
        f"Previous answer: {previous_answer[:1000]}\n"
        f"Answer the original request again by calling the {schema.__name__} tool, "
        "with every required field filled in and only the allowed values."
    )
