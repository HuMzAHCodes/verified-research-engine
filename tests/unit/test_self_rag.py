"""Tests of the Self-RAG subgraph: every path, every cap, and every call counted.

CONCEPT: self-rag. The real subgraph runs with a scripted chat model (one
reply per call, in call order: calls inside one subgraph are made one after
another), scripted query vectors, and a real store and hybrid retriever in a
temporary folder. Each test scripts exactly the calls its path should make;
ScriptedModel fails the test on any extra call, so a broken cap or an extra
loop cannot pass.

D-56: each pass searches with the question and a P-15 search passage (HyDE)
together, in one embedding request; the passage is never answered from.

TC-RES-10 (retrieval stays inside the run's documents) is checked here
through the whole subgraph as well as in test_retrieval.py.
"""

import pytest
from langchain_core.runnables import RunnableLambda

from graph.document_store import DocumentStore, StoredChunk, StoredDocument, chunk_id_for
from graph.llm import MODEL_CALL_BUDGET_KEY, ModelCallBudget
from graph.retrieval import HybridRetriever
from graph.self_rag.graph import build_self_rag_graph, run_self_rag
from graph.self_rag.nodes import checked_quotes
from tests.conftest import ScriptedEmbeddings, ScriptedModel

QUESTION = "How much of Contoso's revenue comes from Northwind?"
FIRST_PASSAGE = "Northwind accounted for approximately X% of our total revenue."
SECOND_PASSAGE = "Revenue from our largest customer, Northwind, was X% of the total."
# A passage with a figure the documents do not hold, to prove it goes nowhere.
INVENTED_PASSAGE = "INVENTED-PASSAGE Northwind accounted for 99% of our total revenue."

ANSWER_TEXT = "Northwind, our largest customer, accounted for 60% of our total revenue in 2025."
BACKGROUND_TEXT = "We signed a new supply agreement with Northwind for warehouse systems."
BOARD_TEXT = "The board of directors meets four times a year."

EAST = [1.0, 0.0, 0.0]
NORTH = [0.0, 1.0, 0.0]
UP = [0.0, 0.0, 1.0]


def save_document(store: DocumentStore, document_id: str, texts_and_vectors) -> None:
    chunks = [
        StoredChunk(
            chunk_id=chunk_id_for(document_id, chunk_index),
            document_id=document_id,
            page=chunk_index + 1,
            chunk_index=chunk_index,
            text=text,
        )
        for chunk_index, (text, _) in enumerate(texts_and_vectors)
    ]
    store.save(
        StoredDocument(
            document_id=document_id,
            file_name=f"{document_id}.pdf",
            fingerprint=f"fingerprint-{document_id}",
            page_count=len(chunks),
            chunk_count=len(chunks),
            embedding_model="mistral-embed",
        ),
        chunks,
        [vector for _, vector in texts_and_vectors],
    )


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    document_store = DocumentStore(tmp_path / "vectors")
    save_document(
        document_store,
        "report",
        [(ANSWER_TEXT, EAST), (BACKGROUND_TEXT, NORTH), (BOARD_TEXT, UP)],
    )
    return document_store


@pytest.fixture
def run_subgraph(make_gateway, store):
    """Run the real subgraph with these scripted replies; return the result and the model."""

    async def run(scripted_replies, document_ids=("report",)):
        chat_model = ScriptedModel(scripted_replies=list(scripted_replies))
        embedding_model = ScriptedEmbeddings(
            vectors_by_text={
                FIRST_PASSAGE: EAST,
                SECOND_PASSAGE: EAST,
                INVENTED_PASSAGE: EAST,
                QUESTION: EAST,
            }
        )
        gateway = make_gateway(decision_model=chat_model, embedding_model=embedding_model)
        self_rag_graph = build_self_rag_graph(gateway, HybridRetriever(store, gateway))
        result = await run_self_rag(self_rag_graph, QUESTION, list(document_ids))
        return result, chat_model, embedding_model

    return run


def search_passage(passage: str) -> dict:
    return {"passage": passage}


def relevant(*labels: str) -> dict:
    return {"relevant_labels": list(labels)}


def draft(answer: str) -> dict:
    return {"answer": answer}


def support(level: str, *evidence: str) -> dict:
    return {"evidence": list(evidence), "support": level}


def quotes(*quoted: str) -> dict:
    return {"quotes": list(quoted)}


def usefulness(verdict: str) -> dict:
    return {"reason": "One line.", "usefulness": verdict}


def prompt_texts(chat_model: ScriptedModel) -> list[str]:
    return [
        "\n".join(str(m.content) for m in conversation)
        for conversation in chat_model.received_conversations
    ]


async def test_the_best_path_answers_in_five_calls_and_one_embedding(run_subgraph):
    result, chat_model, embedding_model = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C1", "C2"),
            draft("Northwind accounted for 60% of Contoso's total revenue in 2025."),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )

    assert result.status == "answered"
    assert result.search_passages == [FIRST_PASSAGE]
    assert [chunk.text for chunk in result.evidence] == [ANSWER_TEXT, BACKGROUND_TEXT]
    assert len(chat_model.received_conversations) == 5
    # D-56: the question and the passage, embedded together in one request.
    assert embedding_model.received_batches == [[QUESTION, FIRST_PASSAGE]]


async def test_the_search_uses_the_question_and_the_passage_together(run_subgraph):
    # D-56: P-15 writes the passage from the question; both are searched.
    _, chat_model, embedding_model = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C1"),
            draft("x"),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )

    assert prompt_texts(chat_model)[0].endswith(f"Question: {QUESTION}")
    assert embedding_model.received_batches == [[QUESTION, FIRST_PASSAGE]]


async def test_a_partly_supported_answer_is_revised_into_checked_quotes(run_subgraph):
    result, chat_model, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C1", "C2"),
            draft("Northwind is a strong, strategic customer providing 60% of revenue."),
            support("partially_supported"),
            # One real quote, one the model made up: only the real one is kept.
            quotes(ANSWER_TEXT, "Northwind is our most strategic partner by far."),
            usefulness("useful"),
        ]
    )

    assert result.status == "answered_with_quotes"
    assert result.revised is True
    assert result.answer == ANSWER_TEXT
    # Only the chunk the kept quote came from stays as evidence.
    assert [chunk.text for chunk in result.evidence] == [ANSWER_TEXT]
    # No second IsSUP after the revision: six calls, the last one IsUSE.
    assert len(chat_model.received_conversations) == 6
    assert "useful: it directly answers" in prompt_texts(chat_model)[-1]


async def test_a_revision_with_no_real_quote_ends_in_not_found(run_subgraph):
    result, chat_model, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C1"),
            draft("Northwind is a key partner."),
            support("no_support"),
            quotes("Northwind is our most strategic partner by far."),
        ]
    )

    assert result.status == "not_found"
    assert (result.answer, result.evidence) == ("", [])
    assert len(chat_model.received_conversations) == 5  # no IsUSE call


async def test_nothing_relevant_searches_once_more_with_other_words(run_subgraph):
    result, chat_model, embedding_model = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant(),
            search_passage(SECOND_PASSAGE),
            relevant("C1"),
            draft("Northwind accounted for 60% of total revenue."),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )

    assert result.status == "answered"
    assert result.search_passages == [FIRST_PASSAGE, SECOND_PASSAGE]
    # The second passage is written knowing which one failed.
    assert f"did not find the answer:\n{FIRST_PASSAGE}" in prompt_texts(chat_model)[2]
    assert embedding_model.received_batches == [
        [QUESTION, FIRST_PASSAGE],
        [QUESTION, SECOND_PASSAGE],
    ]


async def test_nothing_relevant_twice_ends_in_not_found_after_four_calls(run_subgraph):
    result, chat_model, _ = await run_subgraph(
        [search_passage(FIRST_PASSAGE), relevant(), search_passage(SECOND_PASSAGE), relevant()]
    )

    assert result.status == "not_found"
    assert len(chat_model.received_conversations) == 4


async def test_a_not_useful_answer_searches_once_more(run_subgraph):
    result, chat_model, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C2"),
            draft("Contoso signed a supply agreement with Northwind."),
            support("fully_supported"),
            usefulness("not_useful"),
            search_passage(SECOND_PASSAGE),
            relevant("C1"),
            draft("Northwind accounted for 60% of total revenue."),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )

    assert result.status == "answered"
    assert result.search_passages == [FIRST_PASSAGE, SECOND_PASSAGE]
    # The second pass starts clean: the evidence is from the second pass only.
    assert [chunk.text for chunk in result.evidence] == [ANSWER_TEXT]
    assert len(chat_model.received_conversations) == 10


async def test_not_useful_twice_ends_in_not_found(run_subgraph):
    result, _, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C2"),
            draft("An agreement was signed."),
            support("fully_supported"),
            usefulness("not_useful"),
            search_passage(SECOND_PASSAGE),
            relevant("C2"),
            draft("An agreement was signed."),
            support("fully_supported"),
            usefulness("not_useful"),
        ]
    )

    assert result.status == "not_found"


async def test_the_one_revision_is_shared_by_both_passes(run_subgraph):
    # Pass 1 uses the revision; pass 2's unsupported answer cannot revise again.
    result, chat_model, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C2"),
            draft("A strategic agreement."),
            support("partially_supported"),
            quotes(BACKGROUND_TEXT),
            usefulness("not_useful"),
            search_passage(SECOND_PASSAGE),
            relevant("C1"),
            draft("A dominant customer."),
            support("partially_supported"),
        ]
    )

    assert result.status == "not_found"
    assert len(chat_model.received_conversations) == 10


async def test_no_documents_ends_at_once_without_a_call(run_subgraph):
    result, chat_model, embedding_model = await run_subgraph([], document_ids=())

    assert result.status == "not_found"
    assert chat_model.received_conversations == []
    assert embedding_model.received_batches == []


async def test_labels_the_model_was_not_given_are_dropped(run_subgraph):
    result, _, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C9", "c1"),  # C9 does not exist; c1 is C1 in another case
            draft("Northwind accounted for 60% of total revenue."),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )

    assert [chunk.text for chunk in result.evidence] == [ANSWER_TEXT]


async def test_an_empty_search_passage_falls_back_to_the_question(run_subgraph):
    result, _, embedding_model = await run_subgraph(
        [
            search_passage("   "),
            relevant("C1"),
            draft("x"),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )

    assert result.search_passages == [QUESTION]
    # The question is searched once, not twice.
    assert embedding_model.received_batches == [[QUESTION]]


async def test_tc_res_10_evidence_comes_only_from_the_documents_asked_for(run_subgraph, store):
    save_document(store, "other-run", [(ANSWER_TEXT.replace("60%", "99%"), EAST)])

    result, _, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C1", "C2", "C3"),
            draft("x"),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )

    assert {chunk.document_id for chunk in result.evidence} == {"report"}


async def test_every_request_counts_against_the_runs_budget(make_gateway, store):
    chat_model = ScriptedModel(
        scripted_replies=[
            search_passage(FIRST_PASSAGE),
            relevant("C1"),
            draft("x"),
            support("fully_supported"),
            usefulness("useful"),
        ]
    )
    gateway = make_gateway(
        decision_model=chat_model,
        embedding_model=ScriptedEmbeddings(vectors_by_text={FIRST_PASSAGE: EAST, QUESTION: EAST}),
    )
    self_rag_graph = build_self_rag_graph(gateway, HybridRetriever(store, gateway))
    budget = ModelCallBudget(limit=30)

    async def as_a_node_of_a_run(_: object):
        return await run_self_rag(self_rag_graph, QUESTION, ["report"])

    await RunnableLambda(as_a_node_of_a_run).ainvoke(
        None, config={"configurable": {MODEL_CALL_BUDGET_KEY: budget}}
    )

    # Five chat calls and one embedding request (question and passage together).
    assert budget.requests_sent == 6


async def test_the_made_up_passage_is_only_searched_with_never_answered_from(run_subgraph):
    # D-56 guard: the passage is invented by design. It may steer the search,
    # but no later prompt sees it, and it never becomes the answer or evidence.
    result, chat_model, _ = await run_subgraph(
        [
            search_passage(INVENTED_PASSAGE),
            relevant("C1"),
            draft("Northwind accounted for 60% of Contoso's total revenue in 2025."),
            support("partially_supported"),
            quotes(INVENTED_PASSAGE, ANSWER_TEXT),
            usefulness("useful"),
        ]
    )

    later_prompts = prompt_texts(chat_model)[1:]
    assert not any("INVENTED-PASSAGE" in prompt for prompt in later_prompts)
    # A quote copied from the passage is not in any chunk, so it is dropped.
    assert result.answer == ANSWER_TEXT
    assert not any("INVENTED-PASSAGE" in chunk.text for chunk in result.evidence)


def test_checked_quotes_keeps_only_real_quotes_and_their_chunks(store):
    chunks = store.search(["report"], EAST, k=3)

    kept, sources = checked_quotes(
        [
            "Northwind,  our largest customer, accounted for 60%",  # spacing differs: kept
            "Northwind",  # too short to prove anything
            "Northwind accounted for 99% of revenue.",  # not in any chunk
            "Northwind,  our largest customer, accounted for 60%",  # duplicate
        ],
        chunks,
    )

    assert kept == ["Northwind,  our largest customer, accounted for 60%"]
    assert [chunk.text for chunk in sources] == [ANSWER_TEXT]


async def test_an_answered_result_keeps_issups_quotes_for_citations(run_subgraph):
    # D-57, choice A: the wrapper cites only the chunks holding these quotes.
    supporting_quote = "Northwind, our largest customer, accounted for 60%"
    result, _, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C1", "C2"),
            draft("Northwind accounted for 60% of Contoso's total revenue in 2025."),
            support("fully_supported", supporting_quote),
            usefulness("useful"),
        ]
    )

    assert result.status == "answered"
    assert result.support_quotes == [supporting_quote]


async def test_quotes_from_a_failed_first_pass_do_not_reach_the_result(run_subgraph):
    # Pass 1's quotes back an answer that was not useful; pass 2 starts clean.
    result, _, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C2"),
            draft("Contoso signed a supply agreement with Northwind."),
            support("fully_supported", BACKGROUND_TEXT),
            usefulness("not_useful"),
            search_passage(SECOND_PASSAGE),
            relevant("C1"),
            draft("Northwind accounted for 60% of total revenue."),
            support("fully_supported", ANSWER_TEXT),
            usefulness("useful"),
        ]
    )

    assert result.support_quotes == [ANSWER_TEXT]


async def test_a_revised_or_not_found_result_has_no_issup_quotes(run_subgraph):
    revised, _, _ = await run_subgraph(
        [
            search_passage(FIRST_PASSAGE),
            relevant("C1"),
            draft("Northwind is a strong, strategic customer providing 60% of revenue."),
            support("partially_supported", "a quote IsSUP gave"),
            quotes(ANSWER_TEXT),
            usefulness("useful"),
        ]
    )
    not_found, _, _ = await run_subgraph(
        [search_passage(FIRST_PASSAGE), relevant(), search_passage(SECOND_PASSAGE), relevant()]
    )

    # A revised answer is made of checked quotes, which give its citations.
    assert (revised.status, revised.support_quotes) == ("answered_with_quotes", [])
    assert (not_found.status, not_found.support_quotes) == ("not_found", [])
