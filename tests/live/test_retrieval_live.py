"""Live check of hybrid retrieval on the stored annual report: the Walmart question.

CONCEPT: self-rag, choice A (D-52), its acceptance check. Runs only on request
(`uv run pytest -m live`), never in CI. Needs the annual report ingested
(tests/live/test_ingestion_live.py) and no server: retrieval reads the store.

Concept 2's live check found that vector search alone returns no chunk
mentioning Walmart for this question, though 94 chunks on 30 pages do. This
check asks it again, through hybrid retrieval, and prints vector search's
answer beside it. Two embedding requests: one for each search shown.
"""

from pathlib import Path

import pytest

from app.ingestion import file_fingerprint
from graph.document_store import DocumentStore
from graph.llm import ModelGateway
from graph.retrieval import HybridRetriever

pytestmark = pytest.mark.live

ANNUAL_REPORT_PATH = Path("data/uploads/symbotic-annual-report.pdf")
WALMART_QUESTION = "How much of Symbotic's revenue comes from Walmart?"


def describe(chunk) -> str:
    preview = " ".join(chunk.text.split())[:90]
    mentions = "Walmart" if "walmart" in chunk.text.lower() else "-------"
    return f"page {chunk.page:>3}  {mentions}  {preview}"


async def test_hybrid_retrieval_finds_walmart_chunks_for_the_walmart_question():
    if not ANNUAL_REPORT_PATH.is_file():
        pytest.skip(f"put the annual report at {ANNUAL_REPORT_PATH} first (D-45)")
    store = DocumentStore()
    stored_document = store.find_by_fingerprint(file_fingerprint(ANNUAL_REPORT_PATH))
    if stored_document is None:
        pytest.skip("ingest the annual report first: tests/live/test_ingestion_live.py")
    document_ids = [stored_document.document_id]
    gateway = ModelGateway()
    retriever = HybridRetriever(store, gateway)

    (question_vector,) = await gateway.embed([WALMART_QUESTION], prompt_id="LIVE-VECTOR-ONLY")
    by_meaning_only = store.search(document_ids, question_vector, k=6)
    hybrid = await retriever.retrieve(WALMART_QUESTION, document_ids, prompt_id="LIVE-HYBRID")

    print(f"\nquestion: {WALMART_QUESTION}")
    print("vector search only:")
    for chunk in by_meaning_only:
        print("  " + describe(chunk))
    print("hybrid retrieval (D-52):")
    for chunk in hybrid:
        print("  " + describe(chunk))

    walmart_chunks = [chunk for chunk in hybrid if "walmart" in chunk.text.lower()]
    print(
        f"Walmart chunks in the top 6: {len(walmart_chunks)} (vector only: "
        f"{sum('walmart' in chunk.text.lower() for chunk in by_meaning_only)})"
    )
    assert walmart_chunks, "hybrid retrieval must find chunks mentioning Walmart (D-52)"
