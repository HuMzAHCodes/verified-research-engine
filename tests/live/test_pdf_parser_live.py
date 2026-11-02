"""Live check of pdf-parser-mcp: a real PDF through the running server.

CONCEPT: pdf-parser-mcp. Runs only on request (`uv run pytest -m live`), with
the server running (scripts/start_mcp_servers.ps1) and a PDF in data/uploads/
(D-45: Symbotic's annual report). Skipped, with what to do, if either is
missing. Costs nothing: no key, no credits, no model call.

Prints the numbers for "What we measured": pages, pages with text, characters,
the time to ingest and to read every page, and a few lines of one page, so the
quality of pypdf's text (open choice A) can be judged by eye.
"""

import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastmcp import Client

from mcp_servers.pdf_parser_mcp.config import PdfParserConfig
from tests.live.test_mcp_client_live import server_is_listening

pytestmark = pytest.mark.live


async def test_the_running_server_reads_a_real_pdf_page_by_page():
    server_config = PdfParserConfig()
    server_url = (
        f"http://{server_config.pdf_parser_mcp_host}:{server_config.pdf_parser_mcp_port}/mcp"
    )
    if not server_is_listening(server_url):
        pytest.skip(
            f"pdf-parser-mcp is not running at {server_url}; start it with: "
            r".\scripts\start_mcp_servers.ps1"
        )
    uploaded_pdfs = sorted(Path(server_config.pdf_uploads_dir).glob("*.pdf"))
    if not uploaded_pdfs:
        pytest.skip(f"no PDF in {server_config.pdf_uploads_dir}; put the annual report there")
    pdf_path = uploaded_pdfs[0]
    document_id = f"live-{uuid4().hex[:8]}"

    async with Client(server_url) as client:
        started_at = time.perf_counter()
        tool_result = await client.call_tool(
            "ingest_pdf", {"path": str(pdf_path), "document_id": document_id}
        )
        ingest_seconds = time.perf_counter() - started_at
        ingested = tool_result.structured_content

        started_at = time.perf_counter()
        page_texts = []
        for page_number in range(1, ingested["page_count"] + 1):
            page_contents = await client.read_resource(f"pdf://{document_id}/page/{page_number}")
            page_texts.append(page_contents[0].text)
        read_seconds = time.perf_counter() - started_at

    pages_with_text = sum(1 for page_text in page_texts if page_text.strip())
    sample_page_number = max(1, ingested["page_count"] // 2)
    sample_lines = page_texts[sample_page_number - 1].strip().splitlines()[:8]

    print(f"\nfile: {pdf_path.name} ({pdf_path.stat().st_size / 1_048_576:.1f} MB)")
    print(f"pages: {ingested['page_count']}, with text: {pages_with_text}")
    print(f"characters: {sum(len(page_text) for page_text in page_texts):,}")
    print(f"ingest: {ingest_seconds:.1f} s; reading every page: {read_seconds:.1f} s")
    print(f"--- page {sample_page_number}, first lines ---")
    for line in sample_lines:
        print(f"  {line}")

    assert ingested["document_id"] == document_id
    assert ingested["has_text"] is True
    assert len(page_texts) == ingested["page_count"]
