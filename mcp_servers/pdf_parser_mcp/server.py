"""pdf-parser-mcp: a PDF's text, page by page, as an MCP tool and resource.

CONCEPT: pdf-parser-mcp. The server's main file: it defines the tool and the
resource and wires the parts together. The checks and the reading are in
reading.py, the held pages in page_store.py.

- Tool ingest_pdf(path, document_id): an action. Reads the PDF once and holds
  its pages; returns document_id, page_count, and has_text.
- Resource pdf://{document_id}/page/{page_number}: data at an address. The
  text of one page, counted from 1. Reading it changes nothing.

create_server() takes the reading function and the store as arguments, as
search-mcp does, so tests build the same server around a temporary folder and
call it through FastMCP's in-memory client.

Run it, and leave it running, in its own terminal (D-46), or start both
servers with scripts/start_mcp_servers.ps1:
    uv run python -m mcp_servers.pdf_parser_mcp.server
It listens on http://127.0.0.1:8766/mcp, on this machine only.
"""

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from pydantic import BaseModel, Field

from mcp_servers.pdf_parser_mcp.config import load_pdf_parser_config
from mcp_servers.pdf_parser_mcp.page_store import (
    DocumentNotHeld,
    DocumentPageStore,
    PageOutOfRange,
)
from mcp_servers.pdf_parser_mcp.reading import PdfLimits, PdfReadError, read_pdf_pages

SERVER_INSTRUCTIONS = (
    "Call ingest_pdf with the path of a PDF in the uploads folder and a document id, "
    "then read each page from the resource pdf://{document_id}/page/{page_number}, "
    "counted from 1."
)

# A document id appears inside the resource address, so it is limited to
# letters, digits, "-" and "_": no "/", "..", or spaces that could change the
# address's meaning. Ingestion assigns UUIDs, which fit.
DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"

# Reads a PDF at a path and returns its page texts, page 1 first.
ReadPagesFunction = Callable[[str], list[str]]


class IngestedPdf(BaseModel):
    """What ingest_pdf returns: enough to read every page next."""

    document_id: str
    page_count: int
    has_text: bool  # False when no page has any text: a scanned PDF (decision 3)


def create_server(read_pages: ReadPagesFunction, page_store: DocumentPageStore) -> FastMCP:
    """Build pdf-parser-mcp around the given reading function and page store."""
    pdf_server = FastMCP(name="pdf-parser-mcp", instructions=SERVER_INSTRUCTIONS)

    @pdf_server.tool
    async def ingest_pdf(
        path: Annotated[
            str, Field(min_length=1, description="Path of the PDF, inside the uploads folder.")
        ],
        document_id: Annotated[
            str,
            Field(
                pattern=DOCUMENT_ID_PATTERN,
                description="An id for the document: letters, digits, '-' and '_'.",
            ),
        ],
    ) -> IngestedPdf:
        """Read a PDF once and hold its pages. Returns its page count and whether it has text."""
        try:
            # Reading a long PDF takes seconds; a worker thread keeps the
            # server answering other requests meanwhile.
            page_texts = await asyncio.to_thread(read_pages, path)
        except PdfReadError as error:
            # A refused or unreadable file is a tool error with its reason.
            raise ToolError(str(error)) from error
        if not page_texts:
            raise ToolError("the PDF has no pages")
        page_store.add(document_id, page_texts)
        return IngestedPdf(
            document_id=document_id,
            page_count=len(page_texts),
            has_text=any(page_text.strip() for page_text in page_texts),
        )

    # CONCEPT: pdf-parser-mcp. A resource template: FastMCP matches an address
    # such as pdf://doc-1/page/3 and passes document_id="doc-1" and
    # page_number=3, converted to an int from the type hint.
    @pdf_server.resource("pdf://{document_id}/page/{page_number}", mime_type="text/plain")
    def pdf_page(document_id: str, page_number: int) -> str:
        """The text of one page of an ingested PDF, counted from 1; empty if the page has none."""
        try:
            return page_store.page_text(document_id, page_number)
        except (DocumentNotHeld, PageOutOfRange) as error:
            # Found by experiment: any other exception reaches the client as
            # a bare message (such as "'doc9'") with a traceback in the log.
            raise ResourceError(str(error)) from error

    return pdf_server


def main() -> None:
    server_config = load_pdf_parser_config()
    pdf_limits = PdfLimits(
        allowed_folder=server_config.pdf_uploads_dir, max_bytes=server_config.pdf_max_bytes
    )
    pdf_server = create_server(
        read_pages=partial(read_pdf_pages, limits=pdf_limits),
        page_store=DocumentPageStore(max_documents=server_config.pdf_max_documents_held),
    )
    pdf_server.run(
        transport="http",
        host=server_config.pdf_parser_mcp_host,
        port=server_config.pdf_parser_mcp_port,
        path="/mcp",
    )


if __name__ == "__main__":
    main()
