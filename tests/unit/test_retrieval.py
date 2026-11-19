"""Tests of hybrid retrieval: keyword and vector search merged by rank.

CONCEPT: self-rag, choice A (D-52) and decision 6. Each test stores small
documents in a temporary folder with vectors it chooses, so it decides which
chunks vector search finds; the query's vector comes from ScriptedEmbeddings
the same way. Keyword search then runs for real (BM25L) on the chunks' words.

TC-RES-10: retrieval stays inside the run's documents, for both searches.
"""

import pytest

from graph.document_store import DocumentStore, StoredChunk, StoredDocument, chunk_id_for
from graph.retrieval import HybridRetriever, keyword_tokens, reciprocal_rank_fusion
from tests.conftest import ScriptedEmbeddings

# Directions in a small space: a query along EAST is closest to EAST chunks.
EAST = [1.0, 0.0, 0.0]
NORTH = [0.0, 1.0, 0.0]
UP = [0.0, 0.0, 1.0]
NORTH_EAST = [0.7, 0.7, 0.0]


def save_document(
    store: DocumentStore, document_id: str, texts_and_vectors: list[tuple[str, list[float]]]
) -> None:
    """Store a document whose chunk i holds the i-th text, on page i + 1."""
    chunks = [
        StoredChunk(
            chunk_id=chunk_id_for(document_id, chunk_index),
            document_id=document_id,
            page=chunk_index + 1,
            chunk_index=chunk_index,
            text=chunk_text,
        )
        for chunk_index, (chunk_text, _) in enumerate(texts_and_vectors)
    ]
    document = StoredDocument(
        document_id=document_id,
        file_name=f"{document_id}.pdf",
        fingerprint=f"fingerprint-{document_id}",
        page_count=len(chunks),
        chunk_count=len(chunks),
        embedding_model="mistral-embed",
    )
    store.save(document, chunks, [vector for _, vector in texts_and_vectors])


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    return DocumentStore(tmp_path / "vectors")


@pytest.fixture
def make_retriever(make_gateway, store):
    def build_retriever(query_vectors: dict[str, list[float]], retrieved_chunks: int = 6):
        embedding_model = ScriptedEmbeddings(vectors_by_text=query_vectors)
        gateway = make_gateway(embedding_model=embedding_model)
        retriever = HybridRetriever(store, gateway, retrieved_chunks=retrieved_chunks)
        return retriever, embedding_model

    return build_retriever


def test_keyword_tokens_are_lower_case_words_and_numbers_without_function_words():
    assert keyword_tokens("What did Symbotic's revenue (fiscal 2025) grow by? 30%!") == [
        "symbotic",
        "revenue",
        "fiscal",
        "2025",
        "grow",
        "30",
    ]


def test_rank_fusion_prefers_a_chunk_both_searches_rank_well():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])

    # b: 2nd and 1st; a: 1st and 3rd; then d (2nd in one list) before c (3rd).
    assert [chunk_id for chunk_id, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rank_fusion_uses_ranks_only_and_breaks_ties_by_first_seen():
    # c and d are each ranked 1st by one list: equal scores, first seen first.
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion([["c"], ["d"]])] == ["c", "d"]


async def test_a_name_missed_by_vector_search_is_found_by_keyword_search(store, make_retriever):
    # The Walmart chunk points UP, far from the query's EAST: vector search
    # ranks it last. Only keyword search can find it.
    save_document(
        store,
        "annual-report",
        [
            ("Revenue grew in fiscal 2025 across all regions.", EAST),
            ("Our systems automate warehouse storage.", NORTH_EAST),
            ("Gross margin improved on systems deployments.", NORTH),
            ("Walmart accounted for most of our revenue this year.", UP),
        ],
    )
    question = "How much revenue comes from Walmart?"
    retriever, _ = make_retriever({question: EAST}, retrieved_chunks=2)

    vector_only = store.search(["annual-report"], EAST, k=2)
    retrieved = await retriever.retrieve(question, ["annual-report"], prompt_id="Q1-RETRIEVE")

    assert all("Walmart" not in chunk.text for chunk in vector_only)
    assert any("Walmart" in chunk.text for chunk in retrieved)


async def test_a_paraphrase_with_no_shared_word_is_still_found_by_meaning(store, make_retriever):
    save_document(
        store,
        "annual-report",
        [
            ("Robots move cases around distribution centres.", EAST),
            ("The board meets four times a year.", UP),
        ],
    )
    # No word in common with chunk 0; "the" is shared with chunk 1 but is a
    # function word, so it must not make chunk 1 a keyword match (STOP_WORDS).
    question = "What does the company automate?"
    retriever, _ = make_retriever({question: EAST}, retrieved_chunks=1)

    (retrieved_chunk,) = await retriever.retrieve(question, ["annual-report"], prompt_id="Q1-R")

    assert retriever.keyword_search(["annual-report"], question) == []
    assert retrieved_chunk.text.startswith("Robots move cases")


async def test_tc_res_10_retrieval_never_leaves_the_documents_asked_for(store, make_retriever):
    # The other document holds the best match by both meaning and keyword.
    save_document(store, "this-run", [("Warehouse robots and pallets.", NORTH)])
    save_document(store, "other-run", [("Walmart revenue Walmart revenue Walmart.", EAST)])
    question = "Walmart revenue"
    retriever, _ = make_retriever({question: EAST})

    retrieved = await retriever.retrieve(question, ["this-run"], prompt_id="Q1-RETRIEVE")

    assert {chunk.document_id for chunk in retrieved} == {"this-run"}


async def test_each_chunk_appears_once_and_at_most_the_limit_come_back(store, make_retriever):
    texts_and_vectors = [(f"Revenue line {index} for fiscal 2025.", EAST) for index in range(10)]
    save_document(store, "annual-report", texts_and_vectors)
    question = "revenue fiscal 2025"
    retriever, _ = make_retriever({question: EAST}, retrieved_chunks=6)

    retrieved = await retriever.retrieve(question, ["annual-report"], prompt_id="Q1-RETRIEVE")

    # Every chunk is found by both searches, and still appears only once.
    assert len(retrieved) == 6
    assert len({chunk.chunk_id for chunk in retrieved}) == 6
    scores = [chunk.score for chunk in retrieved]
    assert scores == sorted(scores, reverse=True)


async def test_a_query_with_no_word_in_the_documents_falls_back_to_meaning(store, make_retriever):
    save_document(store, "annual-report", [("Revenue grew.", EAST), ("Margins fell.", NORTH)])
    question = "zzzz qqqq"
    retriever, _ = make_retriever({question: NORTH})

    assert retriever.keyword_search(["annual-report"], question) == []
    retrieved = await retriever.retrieve(question, ["annual-report"], prompt_id="Q1-RETRIEVE")
    assert retrieved[0].text == "Margins fell."


async def test_one_embedding_request_per_retrieval_and_none_without_documents(
    store, make_retriever
):
    save_document(store, "annual-report", [("Revenue grew.", EAST)])
    question = "revenue"
    retriever, embedding_model = make_retriever({question: EAST})

    await retriever.retrieve(question, ["annual-report"], prompt_id="Q1-RETRIEVE")
    assert embedding_model.received_batches == [[question]]

    assert await retriever.retrieve(question, [], prompt_id="Q1-RETRIEVE") == []
    assert len(embedding_model.received_batches) == 1


def test_a_name_in_half_the_chunks_of_a_short_document_still_counts(store, make_retriever):
    # Why not BM25Okapi (D-53): in a four-chunk document where two chunks name
    # Walmart, Okapi gives the word no weight. BM25Plus finds exactly the two
    # chunks that contain it, and no chunk without it.
    save_document(
        store,
        "short-report",
        [
            ("Walmart revenue grew.", EAST),
            ("Warehouse robots.", EAST),
            ("Revenue growth overall.", EAST),
            ("Walmart contract renewed with Walmart.", EAST),
        ],
    )
    retriever, _ = make_retriever({})

    found = retriever.keyword_search(["short-report"], "Walmart")

    assert {chunk.chunk_index for chunk in found} == {0, 3}


def test_a_word_repeated_many_times_does_not_beat_a_chunk_matching_more_words(
    store, make_retriever
):
    # Why not BM25L (D-53): on the annual report its top chunk said "Walmart"
    # 10 times and the answer, saying it twice, ranked 108th. BM25Plus limits
    # the reward for repetition, so the chunk that matches more of the
    # query's words ranks first.
    filler = [(f"Filler line {index} about warehouses and storage.", NORTH) for index in range(40)]
    save_document(
        store,
        "annual-report",
        [
            ("Walmart " * 10 + "agreement to purchase systems.", NORTH),
            ("Walmart, our largest customer, accounted for 85% of our total revenue.", NORTH),
            *filler,
        ],
    )
    retriever, _ = make_retriever({})

    found = retriever.keyword_search(
        ["annual-report"], "Walmart share of total revenue largest customer"
    )

    assert found[0].chunk_index == 1
    # Chunks holding no query word are left out, though BM25Plus scores them.
    assert all(chunk.chunk_index in (0, 1) for chunk in found)


async def test_a_second_text_lifts_the_chunk_it_finds_in_one_request(store, make_retriever):
    # D-56: the question and a search passage, merged. The question alone
    # ranks the Northwind chunk last; the passage ranks it first by meaning
    # and by keyword, so merged it moves up, while the question's own best
    # chunk keeps first place (a passage cannot take over the search alone).
    save_document(
        store,
        "report",
        [
            ("Northwind accounted for 60% of our total revenue.", EAST),
            ("The board of directors meets four times a year.", NORTH),
            ("Our largest customer agreement was renewed.", NORTH_EAST),
        ],
    )
    question = "How often does the board meet?"
    passage = "Northwind accounted for X% of our total revenue."
    retriever, embedding_model = make_retriever({question: NORTH_EAST, passage: EAST})

    merged = await retriever.retrieve_for_texts([question, passage], ["report"], prompt_id="T")
    only_question = await retriever.retrieve(question, ["report"], prompt_id="T")

    # Both texts in one request, in the order given.
    assert embedding_model.received_batches[0] == [question, passage]
    # Pages: 1 Northwind, 2 board, 3 agreement.
    assert [chunk.page for chunk in only_question] == [2, 3, 1]
    assert [chunk.page for chunk in merged] == [2, 1, 3]


async def test_one_text_gives_exactly_what_retrieve_gives(store, make_retriever):
    save_document(
        store,
        "report",
        [("Northwind accounted for 60% of revenue.", EAST), ("The board meets.", NORTH)],
    )
    retriever, _ = make_retriever({"Northwind revenue": EAST})

    single = await retriever.retrieve("Northwind revenue", ["report"], prompt_id="T")
    as_texts = await retriever.retrieve_for_texts(
        ["Northwind revenue", "Northwind revenue", "   "], ["report"], prompt_id="T"
    )

    # Repeated and empty texts are dropped, so the scores are the same too.
    assert [(c.chunk_id, c.score) for c in as_texts] == [(c.chunk_id, c.score) for c in single]


async def test_no_texts_or_no_documents_means_no_request(store, make_retriever):
    save_document(store, "report", [("Northwind accounted for 60% of revenue.", EAST)])
    retriever, embedding_model = make_retriever({})

    assert await retriever.retrieve_for_texts(["  "], ["report"], prompt_id="T") == []
    assert await retriever.retrieve_for_texts(["Northwind"], [], prompt_id="T") == []
    assert embedding_model.received_batches == []
