"""Live check of hybrid retrieval on the stored annual report: the Walmart question.

CONCEPT: self-rag, choice A (D-52) and D-53, the acceptance check. Runs only
on request (`uv run pytest -m live`), never in CI. Needs the annual report
ingested (tests/live/test_ingestion_live.py) and no server.

The question has one chunk that answers it: page 23, "Walmart, our largest
customer, accounted for approximately 85% of our total revenue in the fiscal
year ended September 27, 2025". It is found by its words, not by a chunk
number, so the check survives a change to chunking.

How this check got here (Phase 4, concept self-rag):
- First it passed if any top-6 chunk mentioned Walmart; vector search alone
  passed that too, so it proved nothing.
- Then it measured the answer chunk's rank: 10th by vector search, 19th by
  hybrid retrieval with BM25L. The diagnosis (D-53) found that the question's
  wording hurts both searches, and that BM25L over-rewards a repeated word.
- Now the subgraph rewords each sub-question into search words (P-15) before
  retrieving, and keyword search uses BM25Plus. The check asserts the answer
  is in the top 6 for a search query as P-15 is asked to write it, and prints
  the rank for the question as asked, so the gap stays visible.

Three embedding requests: vector search as asked, hybrid as asked, hybrid
for the search query.
"""

from pathlib import Path

import pytest

from app.ingestion import file_fingerprint
from graph.document_store import DocumentStore
from graph.llm import ModelGateway
from graph.retrieval import RETRIEVED_CHUNKS, HybridRetriever

pytestmark = pytest.mark.live

ANNUAL_REPORT_PATH = Path("data/uploads/symbotic-annual-report.pdf")
WALMART_QUESTION = "How much of Symbotic's revenue comes from Walmart?"
# A search query of the kind P-15 is asked to write: 6 to 16 words, names
# kept, likely report words added. The P-15 rewording itself is checked by
# the subgraph's own live check, once P-15 is built.
WALMART_SEARCH_QUERY = "Walmart share of total revenue largest customer"
# Words only the answer chunk holds (checked by searching chunks.jsonl).
ANSWER_WORDS = ("walmart", "% of our total revenue")
# How far down each ranking to look for the answer chunk.
RANKING_DEPTH = 40


def is_answer(chunk) -> bool:
    text = " ".join(chunk.text.lower().split())
    return all(word in text for word in ANSWER_WORDS)


def answer_rank(ranked_chunks) -> int | None:
    """The answer chunk's place in a ranking, counted from 1, or None if absent."""
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if is_answer(chunk):
            return rank
    return None


def describe(rank: int, chunk) -> str:
    marker = "ANSWER " if is_answer(chunk) else "       "
    preview = " ".join(chunk.text.split())[:80]
    return f"  {rank:>2}. page {chunk.page:>3} {marker} {preview}"


def shown_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else f"below {RANKING_DEPTH}"


async def test_hybrid_retrieval_puts_the_answer_chunk_in_the_top_6():
    if not ANNUAL_REPORT_PATH.is_file():
        pytest.skip(f"put the annual report at {ANNUAL_REPORT_PATH} first (D-45)")
    store = DocumentStore()
    stored_document = store.find_by_fingerprint(file_fingerprint(ANNUAL_REPORT_PATH))
    if stored_document is None:
        pytest.skip("ingest the annual report first: tests/live/test_ingestion_live.py")
    document_ids = [stored_document.document_id]
    stored_chunks = store.chunks(stored_document.document_id)
    answer_chunks = [chunk for chunk in stored_chunks if is_answer(chunk)]
    assert answer_chunks, "the answer chunk was not found in the stored report"

    gateway = ModelGateway()
    (question_vector,) = await gateway.embed([WALMART_QUESTION], prompt_id="LIVE-VECTOR-ONLY")
    by_meaning_only = store.search(document_ids, question_vector, k=RANKING_DEPTH)
    # A deeper list than the 6 a run uses, only to see where the answer ranks.
    deep_retriever = HybridRetriever(store, gateway, retrieved_chunks=RANKING_DEPTH)
    hybrid_as_asked = await deep_retriever.retrieve(
        WALMART_QUESTION, document_ids, prompt_id="LIVE-HYBRID-QUESTION"
    )
    hybrid_search_query = await deep_retriever.retrieve(
        WALMART_SEARCH_QUERY, document_ids, prompt_id="LIVE-HYBRID-SEARCH-QUERY"
    )

    print(f"\nanswer chunk: page {answer_chunks[0].page}, chunk {answer_chunks[0].chunk_index}")
    print(f"hybrid retrieval for the search query {WALMART_SEARCH_QUERY!r}, top 6:")
    for rank, chunk in enumerate(hybrid_search_query[:RETRIEVED_CHUNKS], start=1):
        print(describe(rank, chunk))
    print("answer chunk's rank:")
    print(f"  vector search only, question as asked: {shown_rank(answer_rank(by_meaning_only))}")
    print(f"  hybrid, question as asked:             {shown_rank(answer_rank(hybrid_as_asked))}")
    search_query_rank = answer_rank(hybrid_search_query)
    print(f"  hybrid, search query (what P-15 sends): {shown_rank(search_query_rank)}")
    assert search_query_rank is not None and search_query_rank <= RETRIEVED_CHUNKS, (
        "hybrid retrieval must put the answer chunk in the top 6 for the search query (D-53)"
    )
