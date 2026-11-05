"""Tests of IngestionService: the whole pipeline, with no network and no real PDF.

CONCEPT: document-ingestion. Each test runs the real pieces together: the real
pdf-parser-mcp in memory (its reading function replaced by a dictionary of
pages), the real gateway around ScriptedEmbeddings, and a real DocumentStore
in a temporary folder. Only pypdf and Mistral are stood in for.

TC-DOC-03: chunks are at most 600 characters and embedding requests carry at
most 32 of them. TC-DOC-06: an embedding that fails after its retries stores
nothing, and the error names the document.
"""

import pytest

import graph.llm
from app.ingestion import (
    CHUNK_MAX_CHARS,
    IngestionError,
    IngestionService,
    ScannedPdfRefused,
    build_chunks,
    file_fingerprint,
)
from graph.document_store import DocumentStore
from mcp_servers.pdf_parser_mcp.page_store import DocumentPageStore
from mcp_servers.pdf_parser_mcp.reading import PdfReadError
from mcp_servers.pdf_parser_mcp.server import create_server
from tests.conftest import ScriptedEmbeddings, http_error, scripted_vector

# About six chunks a page: long enough to cross chunk boundaries.
REPORT_PAGES = [
    " ".join(f"Page {page_number} sentence {index} about warehouse revenue." for index in range(60))
    for page_number in range(1, 4)
] + ["", "Short last page."]


@pytest.fixture
def uploads_folder(tmp_path):
    folder = tmp_path / "uploads"
    folder.mkdir()
    return folder


def write_upload(uploads_folder, name: str, content: bytes):
    """A file whose bytes give it a fingerprint; its pages come from pages_by_path."""
    pdf_path = uploads_folder / name
    pdf_path.write_bytes(content)
    return pdf_path


@pytest.fixture
def pages_by_path() -> dict[str, list[str]]:
    return {}


@pytest.fixture
def pdf_server(pages_by_path):
    def read_pages_from_dict(path: str) -> list[str]:
        if path not in pages_by_path:
            raise PdfReadError("the path is outside the folder this server may read")
        return pages_by_path[path]

    return create_server(read_pages_from_dict, DocumentPageStore(max_documents=10))


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    return DocumentStore(tmp_path / "vectors")


@pytest.fixture
def make_service(make_gateway, store, pdf_server):
    def build_service(embedding_model: ScriptedEmbeddings, embed_batch_size: int = 32):
        gateway = make_gateway(embedding_model=embedding_model, embed_batch_size=embed_batch_size)
        return IngestionService(gateway, store, pdf_parser_server=pdf_server)

    return build_service


async def test_tc_doc_03_a_pdf_is_chunked_embedded_in_batches_and_stored(
    make_service, store, uploads_folder, pages_by_path
):
    pdf_path = write_upload(uploads_folder, "report.pdf", b"%PDF- report bytes")
    pages_by_path[str(pdf_path)] = REPORT_PAGES
    embedding_model = ScriptedEmbeddings()

    result = await make_service(embedding_model, embed_batch_size=8).ingest(pdf_path)

    document = result.document
    assert result.already_stored is False
    assert (document.file_name, document.page_count) == ("report.pdf", 5)
    # Every request carried at most the batch size, and every chunk was sent once.
    batch_sizes = [len(batch) for batch in embedding_model.received_batches]
    assert max(batch_sizes) <= 8
    assert sum(batch_sizes) == document.chunk_count
    # The stored chunks: within the size, on real pages, the empty page skipped.
    best_match = store.search([document.document_id], scripted_vector("Short last page."), k=1)
    assert best_match[0].text == "Short last page."
    assert best_match[0].page == 5
    all_chunks = store.search([document.document_id], scripted_vector("x"), k=1000)
    assert len(all_chunks) == document.chunk_count
    assert all(len(chunk.text) <= CHUNK_MAX_CHARS for chunk in all_chunks)
    assert {chunk.page for chunk in all_chunks} == {1, 2, 3, 5}


async def test_the_same_file_is_never_embedded_twice(make_service, uploads_folder, pages_by_path):
    pdf_path = write_upload(uploads_folder, "report.pdf", b"%PDF- report bytes")
    pages_by_path[str(pdf_path)] = REPORT_PAGES
    embedding_model = ScriptedEmbeddings()
    ingestion_service = make_service(embedding_model)

    first_result = await ingestion_service.ingest(pdf_path)
    requests_after_first = len(embedding_model.received_batches)
    # The same bytes under another name: still the same document.
    renamed_path = write_upload(uploads_folder, "renamed.pdf", b"%PDF- report bytes")
    second_result = await ingestion_service.ingest(renamed_path)

    assert second_result.already_stored is True
    assert second_result.document == first_result.document
    assert len(embedding_model.received_batches) == requests_after_first


async def test_tc_doc_06_an_embedding_failure_stores_nothing_and_names_the_document(
    make_service, store, uploads_folder, pages_by_path, monkeypatch
):
    async def no_wait(wait_seconds: float) -> None:
        return None

    monkeypatch.setattr(graph.llm, "wait_before_retry", no_wait)
    pdf_path = write_upload(uploads_folder, "report.pdf", b"%PDF- report bytes")
    pages_by_path[str(pdf_path)] = REPORT_PAGES
    # The batch holding the sixth chunk (the second batch of four) fails on
    # every attempt; the other batches are embedded. Tied to a chunk's text,
    # not to arrival order, because batches run in parallel (D-49).
    sixth_chunk_text = build_chunks("any-id", REPORT_PAGES)[5].text
    embedding_model = ScriptedEmbeddings(failures_by_text={sixth_chunk_text: http_error(503)})

    with pytest.raises(
        IngestionError, match="report.pdf: ingestion stopped and nothing was stored"
    ):
        await make_service(embedding_model, embed_batch_size=4).ingest(pdf_path)

    # Nothing in the store: no document folder, no unfinished save, and the
    # file is not known, so the next try embeds it from the start.
    assert not store.vectors_dir.exists() or list(store.vectors_dir.iterdir()) == []
    assert store.find_by_fingerprint(file_fingerprint(pdf_path)) is None


async def test_a_scanned_pdf_is_refused_before_anything_is_embedded(
    make_service, store, uploads_folder, pages_by_path
):
    pdf_path = write_upload(uploads_folder, "scanned.pdf", b"%PDF- scanned bytes")
    pages_by_path[str(pdf_path)] = ["", "  ", "\n"]
    embedding_model = ScriptedEmbeddings()

    with pytest.raises(ScannedPdfRefused, match="no text on any of its 3 pages"):
        await make_service(embedding_model).ingest(pdf_path)

    assert embedding_model.received_batches == []
    assert not store.vectors_dir.exists() or list(store.vectors_dir.iterdir()) == []


async def test_a_file_the_server_refuses_stops_ingestion_with_its_reason(
    make_service, uploads_folder
):
    # Not in pages_by_path: the server refuses it, as it would a file outside uploads.
    pdf_path = write_upload(uploads_folder, "elsewhere.pdf", b"%PDF- other bytes")

    with pytest.raises(IngestionError, match="elsewhere.pdf: .*outside the folder"):
        await make_service(ScriptedEmbeddings()).ingest(pdf_path)


async def test_a_missing_file_is_reported_before_the_server_is_asked(make_service, uploads_folder):
    with pytest.raises(IngestionError, match="no such file"):
        await make_service(ScriptedEmbeddings()).ingest(uploads_folder / "missing.pdf")


async def test_two_different_files_become_two_documents_with_their_own_ids(
    make_service, store, uploads_folder, pages_by_path
):
    first_path = write_upload(uploads_folder, "first.pdf", b"%PDF- first")
    second_path = write_upload(uploads_folder, "second.pdf", b"%PDF- second")
    pages_by_path[str(first_path)] = ["Alpha text."]
    pages_by_path[str(second_path)] = ["Beta text."]
    ingestion_service = make_service(ScriptedEmbeddings())

    first_document = (await ingestion_service.ingest(first_path)).document
    second_document = (await ingestion_service.ingest(second_path)).document

    assert first_document.document_id != second_document.document_id
    # Each document's search finds only its own text.
    (first_hit,) = store.search([first_document.document_id], scripted_vector("Beta text."), k=5)
    assert first_hit.text == "Alpha text."
