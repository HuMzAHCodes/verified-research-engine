"""Scripted tests of the gateway's embed tier: batching, retries, budget, no network.

CONCEPT: document-ingestion, decision 3 and choice B (D-48). gateway.embed()
sends VRE_EMBED_BATCH_SIZE texts per request, one request after another,
through the embedding model's own limiter, the same transport retry as chat
calls, and the run's budget. A failure after the retries returns no vectors
at all (decision 4). The real embeddings client is built so that it never
retries or re-splits a batch by itself ("What the experiment found").
"""

import json
import logging

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

import graph.llm
from graph.llm import (
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

    # 70 chunks at 32 a request: 32, 32, then the 6 left over.
    assert [len(batch) for batch in embedding_model.received_batches] == [32, 32, 6]
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
    # The first batch succeeds; the second fails three times.
    embedding_model = ScriptedEmbeddings(
        scripted_failures=[None, http_error(503), http_error(503), http_error(503)]
    )
    gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=2)

    with pytest.raises(ModelCallError, match=f"{PROMPT_ID}: failed after 3 attempts"):
        await gateway.embed(chunk_texts(4), prompt_id=PROMPT_ID)

    # Nothing was returned, so the first batch's vectors cannot be stored on
    # their own: the caller gets all of a document's vectors or none.
    assert len(embedding_model.received_batches) == 4
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
