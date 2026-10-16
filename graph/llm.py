"""Model gateway: the only place that builds chat models and calls them.

CONCEPT: model-gateway. Every model decision in the graph goes through
ModelGateway.structured(), which applies, in this order:

1. the shared throttle       one rate limiter for every call in the process
2. the transport retry       429, 5xx, timeouts, dropped connections; 3 tries
3. the schema retry          a reply that does not fit the schema gets one
                             more try, with the validation error shown
4. the trace tags            prompt id and version on every LangSmith run

Nodes never import ChatMistralAI. They receive a gateway, so tests can pass
one that wraps a scripted fake model instead of the real API.
"""

import asyncio
import logging
from typing import Any, Literal, TypeVar

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
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


class ModelCallError(RuntimeError):
    """A model call failed for good: retries used up, or an error not worth retrying."""


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
        # CONCEPT: shared throttle. One token bucket for the whole process.
        # Every attempt, by every node, takes a token before it is sent, so
        # parallel researchers (phase 3) share the account's rate limit
        # instead of each assuming it has the whole limit to itself.
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second or settings.vre_throttle_rps,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
        # Tests pass scripted models here; in normal use both are built from
        # settings.
        self.models_by_tier: dict[ModelTier, BaseChatModel] = {
            "decision": decision_model or build_mistral_model(settings.vre_chat_model),
            "writer": writer_model or build_mistral_model(settings.vre_writer_model),
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
                structured_model, conversation, run_config, prompt_id
            )
            parsed_output = reply.get("parsed")
            if parsed_output is not None:
                return parsed_output

            failure_reason = describe_schema_failure(reply)
            logger.warning(
                "%s: reply did not fit %s (attempt %d of %d): %s",
                prompt_id, schema.__name__, schema_attempt, SCHEMA_ATTEMPTS, failure_reason,
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
    ) -> dict[str, Any]:
        """Send one request, retrying only failures that a wait can fix."""
        for transport_attempt in range(1, TRANSPORT_ATTEMPTS + 1):
            await self.rate_limiter.aacquire()
            try:
                return await structured_model.ainvoke(conversation, config=run_config)
            except Exception as error:
                if not is_transient_error(error):
                    raise ModelCallError(f"{prompt_id}: {error}") from error
                if transport_attempt == TRANSPORT_ATTEMPTS:
                    raise ModelCallError(
                        f"{prompt_id}: failed after {TRANSPORT_ATTEMPTS} attempts: {error}"
                    ) from error
                wait_seconds = self.retry_wait_seconds * transport_attempt
                logger.warning(
                    "%s: transient error (attempt %d of %d), retrying in %.0f s: %s",
                    prompt_id, transport_attempt, TRANSPORT_ATTEMPTS, wait_seconds, error,
                )
                await asyncio.sleep(wait_seconds)
        raise AssertionError("unreachable")


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
