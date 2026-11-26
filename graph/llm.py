"""Model gateway: the only place that builds the models and calls them.

CONCEPT: model-gateway. Every model decision in the graph goes through
ModelGateway.structured(), which applies, in this order:

1. the call budget           the run's ModelCallBudget: one more request, or stop
2. the throttle              the limiter for the tier's model, shared by the
                             whole process (shared-throttle)
3. the transport retry       429, 5xx, timeouts, dropped connections; 3 tries,
                             waiting as the provider asks or with a random spread;
                             a 429 waits long enough to reach the next minute (D-55);
                             a used-up daily quota is never retried (D-67)
4. the schema retry          a reply that does not fit the schema gets one
                             more try, with the validation error shown
5. the trace tags            prompt id and version on every LangSmith run

CONCEPT: document-ingestion, decision 3 (D-48). Embeddings go through
ModelGateway.embed(), a third tier with the same budget, throttle, and
transport retry, and its own trace run per request. There is no schema retry:
an embedding reply is a list of numbers, not an answer that can misfit.

CONCEPT: chat-provider-setting (D-66). The chat vendor is a setting
(VRE_CHAT_PROVIDER): build_chat_model hands the vendor, the model name, and
that vendor's key to LangChain's init_chat_model, which returns the vendor's
own chat model (ChatGoogleGenerativeAI, ChatGroq, ChatMistralAI).

Nodes never import a vendor's model class. They receive a gateway, so tests
can pass one that wraps scripted fakes instead of the real API.
"""

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, TypeVar

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import ModelConnectionError, ModelTimeoutError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langchain_mistralai import MistralAIEmbeddings
from langsmith import traceable
from pydantic import BaseModel, SecretStr
from tokenizers import Tokenizer, models

from settings import ChatProvider, settings

logger = logging.getLogger(__name__)

# Which model a prompt uses. Small decision schemas run on the decision model;
# the writer's long, structured answers run on the larger writer model (D-24;
# the models themselves are set in settings.py, D-55).
ModelTier = Literal["decision", "writer"]
# Every tier with its own limiter: the two chat tiers and the embedding model.
LimitedTier = Literal["decision", "writer", "embed"]

# The schema class passed in is the type that comes back out.
OutputSchema = TypeVar("OutputSchema", bound=BaseModel)
# What one request returns: a structured reply, or a batch of vectors.
RequestResult = TypeVar("RequestResult")

TRANSPORT_ATTEMPTS = 3
SCHEMA_ATTEMPTS = 2
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REQUEST_TIMEOUT_SECONDS = 60
# The longest wait before a retry, whatever a Retry-After header says, so a
# wrong or very large value cannot stall a run (shared-throttle, choice B).
# 60 s, one whole minute, since D-55: Mistral's limits are counted per minute.
MAX_RETRY_WAIT_SECONDS = 60.0
# The first wait after a 429 that names no wait (D-55). Mistral sends no
# Retry-After, and its request and token limits are counted per minute, so
# the D-43 waits of 2 s and 4 s all fell inside the minute that was already
# used up (Phase 4, self-rag, step 4). 20 s, then 40 s, adds up to the whole
# minute before the third and last attempt.
RATE_LIMIT_WAIT_SECONDS = 20.0
# The headers in which Mistral and Groq report their limits and what is left
# of them (x-ratelimit-limit-req-minute, x-ratelimit-remaining-tokens-minute,
# ...). Gemini sends none: its refusal names the quota instead (D-67).
RATE_LIMIT_HEADER_PREFIX = "x-ratelimit-"
# How far down an error's chain of causes to look for the vendor's reply
# (chat-provider-setting, decision 3). Google's refusal arrives two deep:
# LangChain's error, raised from Google's, which holds the reply.
MAX_CAUSE_DEPTH = 5
# Google's structured error details (google.rpc): which quota was hit, and
# how long to wait before trying again.
GOOGLE_QUOTA_FAILURE_TYPE = "type.googleapis.com/google.rpc.QuotaFailure"
GOOGLE_RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"
# A daily quota as Google names it ("GenerateRequestsPerDayPerProjectPerModel-
# FreeTier") and as Groq words it ("... on requests per day (RPD) ...").
GOOGLE_DAILY_QUOTA_MARK = "PerDay"
DAILY_LIMIT_IN_MESSAGE = re.compile(r"(?:requests|tokens) per day", re.IGNORECASE)
# Where a run keeps its ModelCallBudget in its config (shared-throttle, choice C).
MODEL_CALL_BUDGET_KEY = "model_call_budget"
# Embedding batches that may wait for their replies at the same time (D-49).
# One reply takes about 2.5 s and the limiter releases a request every 1.25 s,
# so two in flight keep up with it; the third is room for a slow reply.
EMBED_REQUESTS_IN_FLIGHT = 3


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
                f"the most it may send ({self.limit}: sized to the run's plan, D-58, and never "
                "above VRE_MAX_MODEL_CALLS_PER_RUN); stopping instead of sending more"
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


def error_chain(error: BaseException) -> Iterator[BaseException]:
    """The error, then what caused it, and so on, at most MAX_CAUSE_DEPTH deep.

    CONCEPT: chat-provider-setting, decision 3 (D-66). Vendors wrap the
    reply's error in their own: LangChain's GoogleRateLimitError is raised
    from Google's ClientError, Groq's APIConnectionError from httpx's error.
    """
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and len(seen) < MAX_CAUSE_DEPTH and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def response_of(error: BaseException) -> httpx.Response | None:
    """The vendor's HTTP reply, if the error or one of its causes carries one.

    Mistral's error (httpx.HTTPStatusError), Groq's (APIStatusError), and
    Google's (ClientError, under LangChain's) all keep it as `.response`.
    """
    for linked_error in error_chain(error):
        response = getattr(linked_error, "response", None)
        if isinstance(response, httpx.Response):
            return response
    return None


def http_status_of(error: BaseException) -> int | None:
    """The HTTP status of a refused request, from any vendor, or None.

    CONCEPT: chat-provider-setting, decision 3 (D-66). One function for every
    vendor: the status of the reply in the error or one of its causes, else an
    integer `code` (Google's errors keep one even without the reply).
    """
    response = response_of(error)
    if response is not None:
        return response.status_code
    for linked_error in error_chain(error):
        code = getattr(linked_error, "code", None)
        if isinstance(code, int) and 100 <= code <= 599:
            return code
    return None


def google_error_details(error: BaseException) -> list[dict[str, Any]]:
    """Google's structured details of a refusal (google.rpc), or an empty list.

    Google's error keeps the reply's JSON as `.details`: either
    {"error": {"details": [...]}} or the inner part alone.
    """
    for linked_error in error_chain(error):
        reply_json = getattr(linked_error, "details", None)
        if not isinstance(reply_json, dict):
            continue
        error_body = reply_json.get("error", reply_json)
        detail_items = error_body.get("details") if isinstance(error_body, dict) else None
        if isinstance(detail_items, list):
            return [item for item in detail_items if isinstance(item, dict)]
    return []


def daily_quota_used_up(error: BaseException) -> str | None:
    """Which daily quota a 429 says is used up, or None for any other error.

    CONCEPT: chat-provider-setting, decision 4 (D-67). Google names the quota
    in a QuotaFailure (an id with "PerDay" in it); Groq says so in its message
    ("requests per day"). A per-minute 429 gives None, so it is retried.
    """
    if http_status_of(error) != 429:
        return None
    for detail in google_error_details(error):
        if detail.get("@type") != GOOGLE_QUOTA_FAILURE_TYPE:
            continue
        for violation in detail.get("violations") or []:
            quota_id = str(violation.get("quotaId", ""))
            if GOOGLE_DAILY_QUOTA_MARK in quota_id:
                return quota_id
    for linked_error in error_chain(error):
        daily_limit = DAILY_LIMIT_IN_MESSAGE.search(str(linked_error))
        if daily_limit is not None:
            return daily_limit.group(0)
    return None


def is_transient_error(error: BaseException) -> bool:
    """True for failures that may succeed if the same request is sent again.

    A rate limit (429) or a server error (5xx) usually clears after a wait.
    A timeout or a dropped connection (WinError 10054) is a network blip.
    A 400 or 401 means a bad request or a bad key, and a 402 an account
    without API access (Mistral, Phase 5): sending it again cannot help, so
    those fail at once. So does a used-up daily quota (D-67): it does not
    clear within a run. The status is read the same way for every vendor
    (http_status_of); a network failure may sit under a vendor's own error.
    """
    status = http_status_of(error)
    if status is not None:
        return status in RETRYABLE_STATUS_CODES and daily_quota_used_up(error) is None
    network_failures = (
        httpx.RequestError,
        ConnectionError,
        ModelConnectionError,
        ModelTimeoutError,
    )
    return any(isinstance(linked_error, network_failures) for linked_error in error_chain(error))


def build_one_token_per_text_tokenizer() -> Tokenizer:
    """A tokenizer that counts every text as exactly one token, built locally.

    CONCEPT: document-ingestion, decision 3, "What the experiment found".
    MistralAIEmbeddings counts tokens to split a request into pieces under
    16,000 tokens, and sends the pieces together. Without a tokenizer it
    downloads one from Hugging Face when created, and if that fails it counts
    characters, which splits a batch of 32 chunks of 600 characters into two
    requests sent at once: the second would bypass the limiter. This one maps
    a whole text to a single unknown token, with no pre-tokenizer to split it
    into words, so a batch of up to 128 texts counts 128 tokens and is never
    split: the batch size is the gateway's alone (VRE_EMBED_BATCH_SIZE). It
    downloads nothing. It counts nothing real, which is safe because the
    client uses the count only for that split.
    """
    return Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))


def build_mistral_embeddings(model_name: str) -> MistralAIEmbeddings:
    """The Mistral embeddings client, with its own batching and retry switched off.

    max_retries=None switches off its retry (tenacity, 5 attempts with fixed
    30 s waits), so the gateway's retry is the only retry layer, as for chat
    (build_chat_model). The local tokenizer stops it downloading one and
    re-splitting batches (build_one_token_per_text_tokenizer).

    Mistral refuses every request from Phase 5 (402, D-65); the embeddings
    stay here until concept local-embeddings moves them to this machine.
    With no Mistral key set, the client is built with an empty one and its
    first request is refused at once.
    """
    return MistralAIEmbeddings(
        model=model_name,
        api_key=settings.mistral_api_key or SecretStr(""),
        max_retries=None,
        timeout=REQUEST_TIMEOUT_SECONDS,
        tokenizer=build_one_token_per_text_tokenizer(),
    )


def quiet_embeddings_client_errors() -> None:
    """Stop the embeddings client logging a traceback for every failed request.

    The client logs "An error occurred with MistralAI" with a full traceback
    for any error, then raises it. A 429 that the gateway retries a moment
    later would fill the console with tracebacks for nothing. The gateway logs
    every failure itself (prompt id, tier, model, attempt) and raises
    ModelCallError with the cause when it gives up, so nothing is lost.
    """
    logging.getLogger("langchain_mistralai.embeddings").setLevel(logging.CRITICAL)


def summarise_embedding_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """What the trace shows of a request: the texts and the model, not the client.

    The client object would otherwise be written into the trace as text.
    """
    return {"model": inputs["model_name"], "texts": inputs["texts"]}


def summarise_embedding_outputs(vectors: Any) -> dict[str, Any]:
    """What the trace shows of a reply: how many vectors and their length.

    32 vectors of 1,024 numbers would make every trace large and say nothing
    a person can read.
    """
    if not isinstance(vectors, list):
        return {"output": str(vectors)}
    return {"vector_count": len(vectors), "dimensions": len(vectors[0]) if vectors else 0}


@traceable(
    run_type="embedding",
    name="embed",
    process_inputs=summarise_embedding_inputs,
    process_outputs=summarise_embedding_outputs,
)
async def send_embedding_request(
    embedding_model: Embeddings, model_name: str, texts: list[str]
) -> list[list[float]]:
    """One embedding request, as its own LangSmith run.

    CONCEPT: tracing-from-day-one. MistralAIEmbeddings is not a LangChain
    runnable, so unlike the chat models it is not traced by itself; @traceable
    gives every request a run of type "embedding", named by the caller.
    """
    return await embedding_model.aembed_documents(texts)


def build_chat_model(provider: ChatProvider, model_name: str, api_key: SecretStr) -> BaseChatModel:
    """A chat model from any configured vendor, with project-wide settings.

    CONCEPT: chat-provider-setting, decision 2 (D-66). init_chat_model is
    LangChain's own way to choose a model by vendor and name: it imports the
    vendor's package and returns that vendor's chat model. Building a client
    sends no request.

    temperature=0 so the same prompt gives the same answer, which makes runs
    comparable across prompt versions (Groq's client stores it as 1e-08, its
    own floor). max_retries=1 is a single attempt in every client, checked in
    each: Google's default is 6 attempts, Groq's 2, Mistral's counts total
    attempts too. The gateway's retry is the only retry layer, so waits do not
    multiply, every retry is logged in one place, and no retry bypasses the
    shared limiter. timeout is in seconds for every vendor (Google's client
    takes milliseconds underneath; LangChain converts).
    """
    return init_chat_model(
        model_name,
        model_provider=provider,
        temperature=0,
        max_retries=1,
        timeout=REQUEST_TIMEOUT_SECONDS,
        api_key=api_key,
    )


def build_configured_chat_model(model_name: str) -> BaseChatModel:
    """A chat model from the vendor and key in settings (VRE_CHAT_PROVIDER)."""
    return build_chat_model(settings.vre_chat_provider, model_name, settings.chat_api_key())


class ModelGateway:
    """Owns the chat and embedding models and is the only way to call them."""

    def __init__(
        self,
        decision_model: BaseChatModel | None = None,
        writer_model: BaseChatModel | None = None,
        requests_per_second: float | None = None,
        retry_wait_seconds: float = 2.0,
        embedding_model: Embeddings | None = None,
        embed_batch_size: int | None = None,
        rate_limit_wait_seconds: float = RATE_LIMIT_WAIT_SECONDS,
    ) -> None:
        # Tests pass scripted models here; in normal use all three are built
        # from settings.
        self.models_by_tier: dict[ModelTier, BaseChatModel] = {
            "decision": decision_model or build_configured_chat_model(settings.vre_chat_model),
            "writer": writer_model or build_configured_chat_model(settings.vre_writer_model),
        }
        if embedding_model is None:
            quiet_embeddings_client_errors()
            embedding_model = build_mistral_embeddings(settings.vre_embed_model)
        self.embedding_model = embedding_model
        # Chunks per embedding request (document-ingestion, choice B).
        self.embed_batch_size = embed_batch_size or settings.vre_embed_batch_size
        self.model_names_by_tier: dict[LimitedTier, str] = {
            "decision": settings.vre_chat_model,
            "writer": settings.vre_writer_model,
            "embed": settings.vre_embed_model,
        }
        # CONCEPT: shared-throttle, decisions 1 and 2 (D-43). Each tier's
        # model has its own limiter, at that tier's rate, taken from the
        # process registry, so every gateway calling a model shares one
        # bucket. An explicit rate (the tests) gets private limiters instead,
        # so tests never wait for one another. The embedding model joins on
        # the same terms (document-ingestion, decision 3): ingestion and a
        # run's question embeddings share its one bucket.
        if requests_per_second is not None:
            self.rate_limiters_by_tier: dict[LimitedTier, InMemoryRateLimiter] = {
                "decision": build_rate_limiter(requests_per_second),
                "writer": build_rate_limiter(requests_per_second),
                "embed": build_rate_limiter(requests_per_second),
            }
        else:
            self.rate_limiters_by_tier = {
                "decision": shared_rate_limiter(settings.vre_chat_model, settings.vre_decision_rps),
                "writer": shared_rate_limiter(settings.vre_writer_model, settings.vre_writer_rps),
                "embed": shared_rate_limiter(settings.vre_embed_model, settings.vre_embed_rps),
            }
        self.retry_wait_seconds = retry_wait_seconds
        self.rate_limit_wait_seconds = rate_limit_wait_seconds

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
            # The default argument binds this attempt's conversation now, so
            # every transport retry resends exactly what this attempt sent.
            async def send_structured_request(
                conversation_to_send: list[BaseMessage] = conversation,
            ) -> dict[str, Any]:
                return await structured_model.ainvoke(conversation_to_send, config=run_config)

            reply = await self._send_with_transport_retry(send_structured_request, prompt_id, tier)
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

    async def embed(self, texts: list[str], *, prompt_id: str) -> list[list[float]]:
        """Embed texts with mistral-embed and return one vector per text, in order.

        CONCEPT: document-ingestion, decision 3 and choice B. The texts are
        sent VRE_EMBED_BATCH_SIZE at a time, one request per batch, each
        through the embedding model's limiter, the transport retry, and the
        run's budget when there is one. prompt_id names the caller in logs,
        errors, and traces, as for chat calls ("INGEST:<document_id>").

        CONCEPT: document-ingestion, D-49. Up to EMBED_REQUESTS_IN_FLIGHT
        batches wait for their replies at the same time; the limiter still
        releases requests at VRE_EMBED_RPS, so the rate is unchanged and only
        the waiting overlaps. Sent one after another, each batch waited about
        2.5 s for its reply: 101 s for the annual report's 35 requests, where
        the limiter alone needs 44 s.

        All or nothing (decision 4): if one batch fails after its retries, the
        batches still in flight are cancelled, ModelCallError is raised, and
        no vectors are returned at all.
        """
        batches = [
            texts[batch_start : batch_start + self.embed_batch_size]
            for batch_start in range(0, len(texts), self.embed_batch_size)
        ]
        vectors_by_batch: list[list[list[float]]] = [[] for _ in batches]
        requests_in_flight = asyncio.Semaphore(EMBED_REQUESTS_IN_FLIGHT)
        finished_batches = 0

        async def embed_one_batch(batch_number: int, batch: list[str]) -> None:
            nonlocal finished_batches

            async def send_batch() -> list[list[float]]:
                return await send_embedding_request(
                    self.embedding_model,
                    self.model_names_by_tier["embed"],
                    batch,
                    langsmith_extra={
                        "name": prompt_id,
                        "tags": [prompt_id, "tier:embed"],
                        "metadata": {"prompt_id": prompt_id, "batch_number": batch_number + 1},
                    },
                )

            async with requests_in_flight:
                batch_vectors = await self._send_with_transport_retry(
                    send_batch, prompt_id, "embed"
                )
            # A reply with the wrong number of vectors would shift every
            # later chunk onto another chunk's vector; stop instead.
            if len(batch_vectors) != len(batch):
                raise ModelCallError(
                    f"{prompt_id}: asked for {len(batch)} vectors, got {len(batch_vectors)}"
                )
            # Kept by position, so replies that finish out of order still
            # come back in the order of the texts.
            vectors_by_batch[batch_number] = batch_vectors
            finished_batches += 1
            # A long ingestion shows it is moving (D-49): the annual report's
            # first live run was stopped by hand because it printed nothing.
            logger.info("%s: embedded batch %d of %d", prompt_id, finished_batches, len(batches))

        try:
            # A TaskGroup cancels every other task as soon as one fails.
            async with asyncio.TaskGroup() as task_group:
                for batch_number, batch in enumerate(batches):
                    task_group.create_task(embed_one_batch(batch_number, batch))
        except ExceptionGroup as failed_batches:
            # Callers expect the failure itself (ModelCallError), not a group;
            # it keeps its own cause (the HTTP error behind it).
            raise failed_batches.exceptions[0]  # noqa: B904
        return [vector for batch_vectors in vectors_by_batch for vector in batch_vectors]

    async def _send_with_transport_retry(
        self,
        send_request: Callable[[], Awaitable[RequestResult]],
        prompt_id: str,
        tier: LimitedTier,
    ) -> RequestResult:
        """Send one request, retrying only failures that a wait can fix.

        Every attempt first takes one request from the run's budget, then a
        token from its tier's limiter: a retry costs as much as a first try.
        send_request makes the actual call (a chat reply, or a batch of
        vectors), so the same retry serves every tier.
        """
        call_budget = current_call_budget()
        rate_limiter = self.rate_limiters_by_tier[tier]
        for transport_attempt in range(1, TRANSPORT_ATTEMPTS + 1):
            if call_budget is not None:
                call_budget.spend(prompt_id)
            await rate_limiter.aacquire()
            try:
                return await send_request()
            except Exception as error:
                # CONCEPT: chat-provider-setting, decision 4 (D-67). No retry
                # within a run can succeed before the daily reset, so say
                # which quota and what to do, instead of "failed after 3".
                used_up_quota = daily_quota_used_up(error)
                if used_up_quota is not None:
                    raise ModelCallError(
                        f"{prompt_id} ({tier}, {self.model_names_by_tier[tier]}): the daily "
                        f"quota is used up ({used_up_quota}). It resets once a day, so a "
                        "retry cannot help: wait for the reset, or set VRE_CHAT_PROVIDER "
                        "to the backup vendor in .env"
                    ) from error
                if not is_transient_error(error):
                    raise ModelCallError(f"{prompt_id}: {error}") from error
                limits_reported = rate_limit_report(error)
                if transport_attempt == TRANSPORT_ATTEMPTS:
                    raise ModelCallError(
                        f"{prompt_id}: failed after {TRANSPORT_ATTEMPTS} attempts: "
                        f"{error}{limits_reported}"
                    ) from error
                wait_seconds = retry_wait_seconds(
                    error,
                    transport_attempt,
                    self.retry_wait_seconds,
                    self.rate_limit_wait_seconds,
                )
                # CONCEPT: shared-throttle, decision 4. The tier and the model
                # are named, so a run's retries show which limit was reached;
                # since D-55 the provider's own limit headers are added, so a
                # changed limit shows in the log without a diagnostic.
                logger.warning(
                    "%s (%s, %s): transient error (attempt %d of %d), retrying in %.1f s: %s%s",
                    prompt_id,
                    tier,
                    self.model_names_by_tier[tier],
                    transport_attempt,
                    TRANSPORT_ATTEMPTS,
                    wait_seconds,
                    error,
                    limits_reported,
                )
                await wait_before_retry(wait_seconds)
        raise AssertionError("unreachable")


def is_rate_limit_error(error: BaseException) -> bool:
    """True for a 429, from any vendor: the provider refused because a limit was reached."""
    return http_status_of(error) == 429


def retry_wait_seconds(
    error: BaseException,
    attempt: int,
    base_wait_seconds: float,
    rate_limit_wait_seconds: float = RATE_LIMIT_WAIT_SECONDS,
) -> float:
    """How long to wait before the next attempt (shared-throttle, choice B; D-55).

    If the provider said how long to wait (Retry-After, or Gemini's
    retryDelay), wait that long. If not,
    a 429 starts from rate_limit_wait_seconds (20 s), because the limit it hit
    is counted per minute; any other transient failure (5xx, a timeout, a
    dropped connection) starts from base_wait_seconds (2 s). Either start
    doubles on the next attempt, with up to half as much again added at random,
    so branches refused together do not all come back together. Never more
    than MAX_RETRY_WAIT_SECONDS.
    """
    provider_wait = retry_after_seconds(error)
    if provider_wait is not None:
        return min(provider_wait, MAX_RETRY_WAIT_SECONDS)
    first_wait = rate_limit_wait_seconds if is_rate_limit_error(error) else base_wait_seconds
    growing_wait = first_wait * 2 ** (attempt - 1)
    return min(growing_wait + random.uniform(0, growing_wait / 2), MAX_RETRY_WAIT_SECONDS)


def rate_limit_report(error: BaseException) -> str:
    """The provider's limit headers on a refused request, for the log (D-55).

    For example " [limits: limit-req-minute=2, remaining-req-minute=0]". Empty
    when the error has no response or the response has no such headers
    (Gemini sends none; its quota is named in the error itself).
    """
    response = response_of(error)
    if response is None:
        return ""
    reported_limits = [
        f"{header_name.lower().removeprefix(RATE_LIMIT_HEADER_PREFIX)}={header_value}"
        for header_name, header_value in sorted(response.headers.items())
        if header_name.lower().startswith(RATE_LIMIT_HEADER_PREFIX)
    ]
    return f" [limits: {', '.join(reported_limits)}]" if reported_limits else ""


def google_retry_delay_seconds(error: BaseException) -> float | None:
    """The wait Gemini's RetryInfo asks for ("39s" or "1.5s"), or None.

    CONCEPT: chat-provider-setting, decision 4 (D-67). Gemini sends no
    Retry-After header; a per-minute refusal carries the wait in its details.
    """
    for detail in google_error_details(error):
        if detail.get("@type") != GOOGLE_RETRY_INFO_TYPE:
            continue
        delay_text = str(detail.get("retryDelay", "")).strip().removesuffix("s")
        try:
            return max(0.0, float(delay_text))
        except ValueError:
            return None
    return None


def retry_after_seconds(error: BaseException) -> float | None:
    """The wait the provider asks for, in seconds, if the error names one.

    A Retry-After header (Mistral, Groq) holds either a number of seconds
    ("2") or a date; Gemini puts the wait in its RetryInfo instead. Whether
    the provider sends either is not assumed: none, or one that cannot be
    read, gives None, and the caller falls back to its own waits.
    """
    response = response_of(error)
    header_value = response.headers.get("retry-after") if response is not None else None
    if header_value is None:
        return google_retry_delay_seconds(error)
    try:
        return max(0.0, float(header_value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        # "-0000" or no zone at all: HTTP dates are in GMT, so read it as UTC.
        # Subtracting a date with no zone from one with a zone raised TypeError
        # inside the retry (outside test review at the Phase 4 close, F-6).
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


async def wait_before_retry(wait_seconds: float) -> None:
    """Sleep before a retry; a function of its own so tests can record waits."""
    await asyncio.sleep(wait_seconds)


def describe_schema_failure(reply: dict[str, Any]) -> str:
    """Why a structured reply has no parsed value, in one line."""
    parsing_error = reply.get("parsing_error")
    if parsing_error is not None:
        return str(parsing_error)
    # CONCEPT: chat-provider-setting, decision 2. Vendors answer a structured
    # request differently (Gemini with JSON text, Groq and Mistral with a tool
    # call), so the message names neither.
    return "the reply held no structured answer to read"


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
        f"Answer the original request again in the {schema.__name__} format, "
        "with every required field filled in and only the allowed values."
    )
