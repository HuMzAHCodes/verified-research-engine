"""TC-MCP-02: pdf-parser-mcp's tool and resource, called through the MCP protocol.

CONCEPT: pdf-parser-mcp. The real server is built around a temporary uploads
folder and called through FastMCP's in-memory client, so argument checks,
serialisation, tool errors, and resource errors are all real, with no port and
no files outside the test's folder. The PDFs come from tests/pdf_fixtures.py.

A failed tool call reaches the client as ToolError; a failed resource read as
MCPError, carrying the sentence the server's ResourceError gave.
"""

from functools import partial

import pytest
from fastmcp import Client
from fastmcp.exceptions import MCPError, ToolError

from mcp_servers.pdf_parser_mcp.page_store import DocumentPageStore
from mcp_servers.pdf_parser_mcp.reading import PdfLimits, read_pdf_pages
from mcp_servers.pdf_parser_mcp.server import create_server
from tests.pdf_fixtures import write_pdf

PAGE_TEMPLATE = "pdf://{document_id}/page/{page_number}"


@pytest.fixture
def uploads_folder(tmp_path):
    folder = tmp_path / "uploads"
    folder.mkdir()
    return folder


def build_server(uploads_folder, max_documents: int = 10):
    limits = PdfLimits(allowed_folder=uploads_folder, max_bytes=1_048_576)
    return create_server(
        read_pages=partial(read_pdf_pages, limits=limits),
        page_store=DocumentPageStore(max_documents=max_documents),
    )


async def ingest(client: Client, path, document_id: str) -> dict:
    tool_result = await client.call_tool(
        "ingest_pdf", {"path": str(path), "document_id": document_id}
    )
    return tool_result.structured_content


async def read_page(client: Client, document_id: str, page_number: int) -> str:
    resource_contents = await client.read_resource(f"pdf://{document_id}/page/{page_number}")
    return resource_contents[0].text


async def test_tc_mcp_02_the_server_offers_the_tool_and_the_page_resource(uploads_folder):
    async with Client(build_server(uploads_folder)) as client:
        tool_names = [tool.name for tool in await client.list_tools()]
        resource_templates = await client.list_resource_templates()

    assert tool_names == ["ingest_pdf"]
    assert [template.uri_template for template in resource_templates] == [PAGE_TEMPLATE]


async def test_tc_doc_02_each_page_is_read_at_its_own_number(uploads_folder):
    pdf_path = write_pdf(uploads_folder, "report.pdf", ["alpha", "", "gamma"])

    async with Client(build_server(uploads_folder)) as client:
        ingested = await ingest(client, pdf_path, "doc-1")
        page_texts = [await read_page(client, "doc-1", number) for number in (1, 2, 3)]

    assert ingested == {"document_id": "doc-1", "page_count": 3, "has_text": True}
    assert page_texts == ["alpha", "", "gamma"]


async def test_a_pdf_with_no_text_is_reported_by_has_text(uploads_folder):
    pdf_path = write_pdf(uploads_folder, "scanned.pdf", ["", ""])

    async with Client(build_server(uploads_folder)) as client:
        ingested = await ingest(client, pdf_path, "doc-1")

    assert ingested["has_text"] is False
    assert ingested["page_count"] == 2


async def test_a_refused_file_is_a_tool_error_with_its_reason(uploads_folder):
    outside_path = write_pdf(uploads_folder.parent, "outside.pdf", ["secret"])

    async with Client(build_server(uploads_folder)) as client:
        with pytest.raises(ToolError, match="outside the folder this server may read"):
            await ingest(client, outside_path, "doc-1")


@pytest.mark.parametrize("bad_document_id", ["../doc", "doc 1", "doc/1", ""])
async def test_a_document_id_that_could_change_the_address_is_refused(
    uploads_folder, bad_document_id
):
    pdf_path = write_pdf(uploads_folder, "report.pdf", ["alpha"])

    async with Client(build_server(uploads_folder)) as client:
        with pytest.raises(ToolError):
            await ingest(client, pdf_path, bad_document_id)


async def test_reading_a_document_not_held_says_to_ingest_it_first(uploads_folder):
    async with Client(build_server(uploads_folder)) as client:
        with pytest.raises(MCPError, match="call ingest_pdf first"):
            await read_page(client, "doc-9", 1)


async def test_reading_a_page_past_the_end_says_which_pages_exist(uploads_folder):
    pdf_path = write_pdf(uploads_folder, "report.pdf", ["alpha", "beta"])

    async with Client(build_server(uploads_folder)) as client:
        await ingest(client, pdf_path, "doc-1")
        with pytest.raises(MCPError, match="has pages 1 to 2"):
            await read_page(client, "doc-1", 3)


async def test_the_oldest_document_is_forgotten_when_the_store_is_full(uploads_folder):
    first_pdf = write_pdf(uploads_folder, "first.pdf", ["one"])
    second_pdf = write_pdf(uploads_folder, "second.pdf", ["two"])

    async with Client(build_server(uploads_folder, max_documents=1)) as client:
        await ingest(client, first_pdf, "doc-1")
        await ingest(client, second_pdf, "doc-2")

        assert await read_page(client, "doc-2", 1) == "two"
        with pytest.raises(MCPError, match="call ingest_pdf first"):
            await read_page(client, "doc-1", 1)
