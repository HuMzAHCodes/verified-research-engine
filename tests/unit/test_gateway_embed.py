"""Scripted tests of the gateway's embed tier: batching, retries, budget, no network.

CONCEPT: document-ingestion, decision 3 and choice B (D-48). gateway.embed()
sends VRE_EMBED_BATCH_SIZE texts per request through the embedding model's own
limiter, the same transport retry as chat calls, and the run's budget. A
failure after the retries returns no vectors at all (decision 4). The real
embeddings client is built so that it never retries or re-splits a batch by
itself ("What the experiment found").

CONCEPT: document-ingestion, D-49. Up to three batches wait for their replies
at once. Replies that finish out of order still come back in order, a batch
that fails for good cancels the others, and progress is logged per batch.
Tests of several batches never rely on which batch arrives first.
"""

import asyncio
import json
import logging

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

import graph.llm
from graph.llm import (
    EMBED_REQUESTS_IN_FLIGHT,
    MODEL_CALL_BUDGET_KEY,
    ModelCallBudget,
    ModelCallBudgetExceeded,
    ModelCallError,
    ModelGateway,
    build_mistral_embeddings,
    summarise_embedding_inputs,
    summarise_embedding_outputs,
)
from settings import settings
from tests.conftest import ScriptedEmbeddings, ScriptedModel, http_error, scripted_vector

PROMPT_ID = "INGEST:test-document"


def chunk_texts(count: int) -> list[str]:
    """Distinct chunk texts, so each has its own scripted vector."""
    return [f"Chunk number {index} of the test document." for index in range(count)]


@pytest.fixture
def recorded_waits(monkeypatch) -> list[float]:
    """The waits the gateway asks for before retries, recorded instead of slept."""
    waits: list[float] = []

    async def record_wait(wait_seconds: float) -> None:
        waits.append(wait_seconds)

    monkeypatch.setattr(graph.llm, "wait_before_retry", record_wait)
    return waits


async def test_texts_are_sent_one_batch_per_request_and_come_back_in_order(make_gateway):
    embedding_model = ScriptedEmbeddings()
    gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=32)
    texts = chunk_texts(70)

    vectors = await gateway.embed(texts, prompt_id=PROMPT_ID)

    # 70 chunks at 32 a request: 32, 32, and the 6 left over, in any order.
    assert sorted(len(batch) for batch in embedding_model.received_batches) == [6, 32, 32]
    # Every vector belongs to the chunk at the same position.
    assert vectors == [scripted_vector(text) for text in texts]


async def test_no_texts_send_no_request(make_gateway):
    embedding_model = ScriptedEmbeddings()

    vectors = await make_gateway(embedding_model=embedding_model).embed([], prompt_id=PROMPT_ID)

    assert vectors == []
    assert embedding_model.received_batches == []


async def test_the_batch_size_defaults_to_the_setting(make_gateway):
    assert make_gateway().embed_batch_size == settings.vre_embed_batch_size


async def test_a_rate_limit_is_retried_with_the_same_batch(make_gateway, recorded_waits):
    embedding_model = ScriptedEmbeddings(scripted_failures=[http_error(429)])
    texts = chunk_texts(3)

    vectors = await make_gateway(embedding_model=embedding_model).embed(texts, prompt_id=PROMPT_ID)

    assert vectors == [scripted_vector(text) for text in texts]
    # The retry resent exactly the batch that was refused, and nothing twice.
    assert embedding_model.received_batches == [texts, texts]
    assert len(recorded_waits) == 1


async def test_tc_doc_06_failure_after_the_retries_returns_no_vectors(
    make_gateway, recorded_waits, caplog
):
    caplog.set_level(logging.WARNING)
    texts = chunk_texts(4)
    # The second batch (texts 2 and 3) fails on every attempt; the first succeeds.
    embedding_model = ScriptedEmbeddings(failures_by_text={texts[2]: http_error(503)})
    gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=2)

    with pytest.raises(ModelCallError, match=f"{PROMPT_ID}: failed after 3 attempts"):
        await gateway.embed(texts, prompt_id=PROMPT_ID)

    # Nothing was returned, so the first batch's vectors cannot be stored on
    # their own: the caller gets all of a document's vectors or none.
    assert embedding_model.received_batches.count(texts[:2]) == 1
    assert embedding_model.received_batches.count(texts[2:]) == 3
    # Each retry names the embed tier and its model (shared-throttle, decision 4).
    assert f"{PROMPT_ID} (embed, {settings.vre_embed_model}): transient error" in caplog.text


async def test_a_bad_key_fails_at_once(make_gateway):
    embedding_model = ScriptedEmbeddings(scripted_failures=[http_error(401)])

    with pytest.raises(ModelCallError, match="401"):
        await make_gateway(embedding_model=embedding_model).embed(
            chunk_texts(2), prompt_id=PROMPT_ID
        )

    assert len(embedding_model.received_batches) == 1


class ShortReplyEmbeddings(ScriptedEmbeddings):
    """Returns one vector too few, as a broken reply might."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return (await super().aembed_documents(texts))[:-1]


async def test_a_reply_with_the_wrong_number_of_vectors_stops(make_gateway):
    gateway = make_gateway(embedding_model=ShortReplyEmbeddings())

    with pytest.raises(ModelCallError, match="asked for 3 vectors, got 2"):
        await gateway.embed(chunk_texts(3), prompt_id=PROMPT_ID)


async def embed_in_a_run(gateway, texts: list[str], budget: ModelCallBudget) -> list[list[float]]:
    """Embed the way a node does: inside a runnable with the run's config."""

    async def node_like_step(_: object) -> list[list[float]]:
        return await gateway.embed(texts, prompt_id=PROMPT_ID)

    in_a_run = RunnableLambda(node_like_step)
    return await in_a_run.ainvoke(None, config={"configurable": {MODEL_CALL_BUDGET_KEY: budget}})


async def test_inside_a_run_every_embedding_request_counts_against_the_budget(make_gateway):
    # One rate limit, then two batches: three requests in all.
    embedding_model = ScriptedEmbeddings(scripted_failures=[http_error(429)])
    gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=2)
    budget = ModelCallBudget(limit=3)

    await embed_in_a_run(gateway, chunk_texts(4), budget)
    assert budget.requests_sent == 3

    with pytest.raises(ModelCallBudgetExceeded, match=PROMPT_ID):
        await embed_in_a_run(gateway, chunk_texts(1), budget)


def test_the_embed_tier_has_its_own_limiter_shared_by_every_gateway():
    first_gateway = ModelGateway(
        decision_model=ScriptedModel(),
        writer_model=ScriptedModel(),
        embedding_model=ScriptedEmbeddings(),
    )
    second_gateway = ModelGateway(
        decision_model=ScriptedModel(),
        writer_model=ScriptedModel(),
        embedding_model=ScriptedEmbeddings(),
    )

    embed_limiter = first_gateway.rate_limiters_by_tier["embed"]
    # Its own bucket: embedding never slows the chat tiers, or the reverse.
    assert embed_limiter is not first_gateway.rate_limiters_by_tier["decision"]
    assert embed_limiter is not first_gateway.rate_limiters_by_tier["writer"]
    # Shared: ingestion and a run's question embeddings take from one bucket.
    assert embed_limiter is second_gateway.rate_limiters_by_tier["embed"]
    # At 80% of mistral-embed's 1 request a second (D-48).
    assert embed_limiter.requests_per_second == settings.vre_embed_rps


def test_the_real_client_never_retries_or_resplits_a_batch_by_itself():
    embeddings_client = build_mistral_embeddings(settings.vre_embed_model)

    # max_retries=None: the client's retry decorator returns the function
    # unchanged, so the gateway's retry is the only one.
    assert embeddings_client._retry(len) is len
    # The local tokenizer counts each text as one token, so even 128 chunks
    # of 600 characters stay one batch (the client splits above 16,000 tokens).
    long_chunks = ["revenue backlog " * 37] * 128
    assert len(list(embeddings_client._get_batches(long_chunks))) == 1


async def test_the_real_client_sends_one_http_request_per_batch(make_gateway):
    # The real client, with its network replaced by a recorder: the same
    # check as the experiment before this concept was written.
    request_sizes: list[int] = []

    def answer_embedding_request(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        request_sizes.append(len(texts))
        return httpx.Response(200, json={"data": [{"embedding": [0.5] * 4} for _ in texts]})

    embeddings_client = build_mistral_embeddings(settings.vre_embed_model)
    embeddings_client.async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(answer_embedding_request),
        base_url="https://api.mistral.ai/v1/",
    )
    gateway = make_gateway(embedding_model=embeddings_client, embed_batch_size=32)

    vectors = await gateway.embed(["word " * 120] * 64, prompt_id=PROMPT_ID)

    assert request_sizes == [32, 32]
    assert len(vectors) == 64


def test_a_trace_shows_the_texts_and_vector_counts_not_the_vectors():
    traced_inputs = summarise_embedding_inputs(
        {"embedding_model": object(), "model_name": "mistral-embed", "texts": ["a", "b"]}
    )
    traced_outputs = summarise_embedding_outputs([[0.1] * 1024] * 32)

    assert traced_inputs == {"model": "mistral-embed", "texts": ["a", "b"]}
    assert traced_outputs == {"vector_count": 32, "dimensions": 1024}


# --- D-49: batches in flight ---------------------------------------------


class SlowEmbeddings(ScriptedEmbeddings):
    """Answers each batch after a delay, and records how many wait at once."""

    def __init__(
        self,
        delay_seconds_by_first_text: dict[str, float],
        default_delay_seconds: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.delay_seconds_by_first_text = delay_seconds_by_first_text
        self.default_delay_seconds = default_delay_seconds
        self.batches_waiting = 0
        self.most_batches_waiting = 0
        self.completed_batches: list[list[str]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches_waiting += 1
        self.most_batches_waiting = max(self.most_batches_waiting, self.batches_waiting)
        try:
            await asyncio.sleep(
                self.delay_seconds_by_first_text.get(texts[0], self.default_delay_seconds)
            )
            vectors = await super().aembed_documents(texts)
            self.completed_batches.append(list(texts))
            return vectors
        finally:
            self.batches_waiting -= 1


async def test_replies_that_finish_out_of_order_come_back_in_order(make_gateway):
    texts = chunk_texts(6)
    # The first batch is the slowest, the last the fastest.
    embedding_model = SlowEmbeddings(
        delay_seconds_by_first_text={texts[0]: 0.3, texts[2]: 0.15, texts[4]: 0.0}
    )
    gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=2)

    vectors = await gateway.embed(texts, prompt_id=PROMPT_ID)

    assert embedding_model.completed_batches[0] == texts[4:]  # finished first
    assert vectors == [scripted_vector(text) for text in texts]  # still in order


async def test_no_more_than_three_batches_wait_at_once(make_gateway):
    # Each reply takes 0.5 s, far longer than the test limiter's polling
    # (0.1 s): with 0.05 s, a batch could finish before the third one was
    # released, and the test failed now and then in the full suite (What broke).
    embedding_model = SlowEmbeddings(delay_seconds_by_first_text={}, default_delay_seconds=0.5)
    gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=1)

    await gateway.embed(chunk_texts(5), prompt_id=PROMPT_ID)

    assert EMBED_REQUESTS_IN_FLIGHT == 3
    assert embedding_model.most_batches_waiting == EMBED_REQUESTS_IN_FLIGHT


async def test_a_batch_that_fails_for_good_cancels_the_batches_in_flight(make_gateway):
    texts = chunk_texts(3)
    # Batch 1 fails at once (a bad key is not retried); batches 2 and 3 are slow.
    embedding_model = SlowEmbeddings(
        delay_seconds_by_first_text={texts[0]: 0.0, texts[1]: 2.0, texts[2]: 2.0},
        failures_by_text={texts[0]: http_error(401)},
    )
    gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=1)
    started_at = asyncio.get_running_loop().time()

    with pytest.raises(ModelCallError, match="401"):
        await gateway.embed(texts, prompt_id=PROMPT_ID)

    # The slow batches were cancelled, not waited for.
    assert asyncio.get_running_loop().time() - started_at < 1.0
    assert embedding_model.completed_batches == []


async def test_progress_is_logged_for_every_batch(make_gateway, caplog):
    caplog.set_level(logging.INFO, logger="graph.llm")
    gateway = make_gateway(embed_batch_size=2)

    await gateway.embed(chunk_texts(6), prompt_id=PROMPT_ID)

    for finished_batches in (1, 2, 3):
        assert f"{PROMPT_ID}: embedded batch {finished_batches} of 3" in caplog.text
