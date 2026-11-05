"""Live check: the annual report through the whole ingestion pipeline.

CONCEPT: document-ingestion, "What we measured". Runs only on request
(`uv run pytest -m live`), never in CI. Needs pdf-parser-mcp running
(scripts/start_mcp_servers.ps1) and the report in data/uploads/ (D-45).

It ingests into the real store, data/vectors/, so the document is there for
concept 3 and the Phase 4 notebook. The first run embeds the report (about 34
requests); every later run finds the same fingerprint and embeds nothing.
Either way it then asks one question (one more request) and prints the three
closest chunks with their pages, as a first look at retrieval quality.
"""

import math
import time
from pathlib import Path

import pytest

from app.ingestion import CHUNK_MAX_CHARS, IngestionService
from graph.document_store import DocumentStore
from graph.llm import ModelGateway
from settings import settings

pytestmark = pytest.mark.live

ANNUAL_REPORT_PATH = Path("data/uploads/symbotic-annual-report.pdf")
RETRIEVAL_QUESTION = "How much of Symbotic's revenue comes from Walmart?"


async def test_the_annual_report_is_ingested_and_searchable():
    if not ANNUAL_REPORT_PATH.is_file():
        pytest.skip(f"put the annual report at {ANNUAL_REPORT_PATH} first (D-45)")
    gateway = ModelGateway()
    store = DocumentStore()

    result = await IngestionService(gateway, store).ingest(ANNUAL_REPORT_PATH)

    document = result.document
    all_chunks = store.search([document.document_id], [1.0] + [0.0] * 1023, k=document.chunk_count)
    chunk_lengths = sorted(len(chunk.text) for chunk in all_chunks)
    if result.already_stored:
        print(f"\nalready stored as {document.document_id}: nothing embedded")
    else:
        embedding_requests = math.ceil(document.chunk_count / settings.vre_embed_batch_size)
        print(
            f"\ningested {document.file_name}: {document.page_count} pages, "
            f"{document.chunk_count} chunks, {embedding_requests} embedding requests, "
            f"{result.seconds:.1f} s"
        )
    print(
        f"chunk length: shortest {chunk_lengths[0]}, median "
        f"{chunk_lengths[len(chunk_lengths) // 2]}, longest {chunk_lengths[-1]}; "
        f"pages with chunks: {len({chunk.page for chunk in all_chunks})}"
    )
    assert chunk_lengths[-1] <= CHUNK_MAX_CHARS
    assert len(all_chunks) == document.chunk_count

    started_at = time.perf_counter()
    (question_vector,) = await gateway.embed([RETRIEVAL_QUESTION], prompt_id="LIVE-RETRIEVE")
    closest_chunks = store.search([document.document_id], question_vector, k=3)
    print(f"retrieval for {RETRIEVAL_QUESTION!r} in {time.perf_counter() - started_at:.1f} s:")
    for closest_chunk in closest_chunks:
        preview = " ".join(closest_chunk.text.split())[:110]
        print(f"  page {closest_chunk.page}, score {closest_chunk.score:.3f}: {preview}")
    assert len(closest_chunks) == 3
