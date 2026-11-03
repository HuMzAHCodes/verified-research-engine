"""Tests of PdfParserClient against the real pdf-parser-mcp, in memory.

CONCEPT: document-ingestion, decision 7. The client talks to the server built
by create_server(), reached through FastMCP's in-memory transport, so the
whole MCP path runs (tool call, resource read, error envelopes) with no port
and no PDF file: the reading function is replaced by a dictionary of pages.
"""

import pytest
from fastmcp import FastMCP

from app.pdf_parser_client import (
    IngestedPdf,
    PdfParserClient,
    PdfParserError,
    PdfParserUnavailable,
)
from mcp_servers.pdf_parser_mcp.page_store import DocumentPageStore
from mcp_servers.pdf_parser_mcp.reading import PdfReadError
from mcp_servers.pdf_parser_mcp.server import create_server

PAGES_BY_PATH = {
    "data/uploads/report.pdf": ["Page one text.", "", "Page three text."],
    "data/uploads/scanned.pdf": ["", " "],
}


def read_pages_from_dict(path: str) -> list[str]:
    """Stands in for pypdf: the pages of a known path, or the server's refusal."""
    if path not in PAGES_BY_PATH:
        raise PdfReadError("the path is outside the folder this server may read")
    return PAGES_BY_PATH[path]


@pytest.fixture
def pdf_server() -> FastMCP:
    return create_server(read_pages_from_dict, DocumentPageStore(max_documents=10))


async def test_a_pdf_is_ingested_and_every_page_read_in_order(pdf_server):
    async with PdfParserClient.connect(pdf_server) as pdf_parser:
        ingested_pdf = await pdf_parser.ingest_pdf("data/uploads/report.pdf", "doc-1")
        page_texts = await pdf_parser.read_all_pages(ingested_pdf)

    assert ingested_pdf == IngestedPdf(document_id="doc-1", page_count=3, has_text=True)
    # Page 2 has no text and comes back empty, keeping later pages in place.
    assert page_texts == ["Page one text.", "", "Page three text."]


async def test_a_pdf_with_no_text_is_reported_as_such(pdf_server):
    async with PdfParserClient.connect(pdf_server) as pdf_parser:
        ingested_pdf = await pdf_parser.ingest_pdf("data/uploads/scanned.pdf", "doc-2")

    assert ingested_pdf.has_text is False


async def test_a_refused_file_raises_the_servers_reason(pdf_server):
    async with PdfParserClient.connect(pdf_server) as pdf_parser:
        with pytest.raises(PdfParserError, match="outside the folder") as raised:
            await pdf_parser.ingest_pdf("C:/secret.pdf", "doc-3")

    # The server answered, so this is a refusal, not an unavailable server.
    assert not isinstance(raised.value, PdfParserUnavailable)


async def test_a_page_that_does_not_exist_raises_the_servers_reason(pdf_server):
    async with PdfParserClient.connect(pdf_server) as pdf_parser:
        await pdf_parser.ingest_pdf("data/uploads/report.pdf", "doc-1")
        with pytest.raises(PdfParserError, match="no page 9"):
            await pdf_parser.read_page("doc-1", 9)
        with pytest.raises(PdfParserError, match="call ingest_pdf first"):
            await pdf_parser.read_page("never-ingested", 1)


async def test_no_server_at_the_address_says_how_to_start_it():
    # Port 9 on this machine: nothing listens there.
    with pytest.raises(PdfParserUnavailable, match="start it with: uv run python -m"):
        async with PdfParserClient.connect(
            "http://127.0.0.1:9/mcp", call_timeout_seconds=2
        ) as pdf_parser:
            await pdf_parser.read_page("doc-1", 1)


async def test_a_server_without_the_page_resource_is_refused_at_connection():
    wrong_server = FastMCP(name="wrong-server")

    @wrong_server.tool
    def ingest_pdf(path: str, document_id: str) -> dict:
        return {}

    with pytest.raises(PdfParserUnavailable, match="missing the resource"):
        async with PdfParserClient.connect(wrong_server):
            pass


async def test_an_error_inside_the_block_is_not_mistaken_for_a_connection_failure(pdf_server):
    with pytest.raises(ValueError, match="the caller's own bug"):
        async with PdfParserClient.connect(pdf_server):
            raise ValueError("the caller's own bug")


class StoppedServerClient:
    """A FastMCP client whose connection has dropped."""

    async def call_tool(self, *args, **kwargs):
        raise ConnectionResetError("connection reset by peer")

    async def read_resource(self, *args, **kwargs):
        raise ConnectionResetError("connection reset by peer")


async def test_a_server_that_stops_answering_is_unavailable():
    pdf_parser = PdfParserClient(StoppedServerClient(), "test server")

    with pytest.raises(PdfParserUnavailable, match="stopped answering during ingest_pdf"):
        await pdf_parser.ingest_pdf("data/uploads/report.pdf", "doc-1")
    with pytest.raises(PdfParserUnavailable, match="stopped answering during pdf://doc-1/page/1"):
        await pdf_parser.read_page("doc-1", 1)
