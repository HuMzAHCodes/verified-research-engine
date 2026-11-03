"""Ingestion's side of pdf-parser-mcp: read a PDF's pages over MCP.

CONCEPT: document-ingestion, decision 7 (D-48). Ingestion never imports pypdf
or anything from mcp_servers/; it reaches the PDF server only through this
client, so the server can move (Phase 11) by changing VRE_PDF_PARSER_MCP_URL.

PdfParserClient wraps FastMCP's own Client, as McpToolbox does for search-mcp
(D-34). One connection serves one ingestion:

    async with PdfParserClient.connect(settings.vre_pdf_parser_mcp_url) as pdf_parser:
        ingested_pdf = await pdf_parser.ingest_pdf(path, document_id)
        page_texts = await pdf_parser.read_all_pages(ingested_pdf)

Failures come as one exception family, PdfParserError, in two kinds:

- PdfParserError itself: the server answered and refused this PDF or page
  (outside the uploads folder, not a PDF, encrypted, no such page).
- PdfParserUnavailable, a subclass: the server cannot be used at all (not
  running, stopped answering, missing its tool or resource, wrong shape).

A caller that only needs "ingestion failed" catches PdfParserError; one that
must tell "fix the file" from "start the server" can check the subclass.
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.exceptions import MCPError, ToolError
from langsmith import traceable
from pydantic import BaseModel, ValidationError

INGEST_TOOL_NAME = "ingest_pdf"
PAGE_RESOURCE_TEMPLATE = "pdf://{document_id}/page/{page_number}"
# Reading a 124-page PDF took 6.0 s (pdf-parser-mcp, "What we measured"); a
# much longer one still fits in a minute. Only a server that has stopped
# answering reaches this limit.
CALL_TIMEOUT_SECONDS = 60.0
START_SERVER_COMMAND = "uv run python -m mcp_servers.pdf_parser_mcp.server"


class IngestedPdf(BaseModel):
    """What ingest_pdf returns. The application's own copy of the server's shape:
    app/ does not import mcp_servers/, so the protocol is the contract."""

    document_id: str
    page_count: int
    has_text: bool


class PdfParserError(RuntimeError):
    """pdf-parser-mcp refused this PDF or page; the message is the server's reason."""


class PdfParserUnavailable(PdfParserError):
    """pdf-parser-mcp cannot be used; the message says why and what to do."""


def without_self(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the client object itself out of the trace; show only the arguments."""
    return {name: value for name, value in inputs.items() if name != "self"}


def summarise_pages(page_texts: list[str]) -> dict[str, Any]:
    """What the trace shows of a whole document: counts, not every page's text."""
    return {
        "page_count": len(page_texts),
        "pages_with_text": sum(1 for page_text in page_texts if page_text.strip()),
        "characters": sum(len(page_text) for page_text in page_texts),
    }


class PdfParserClient:
    """One open MCP connection to pdf-parser-mcp."""

    def __init__(self, mcp_client: Client, server_label: str) -> None:
        self._mcp_client = mcp_client
        self._server_label = server_label

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        server: str | FastMCP,
        *,
        call_timeout_seconds: float = CALL_TIMEOUT_SECONDS,
    ) -> AsyncIterator["PdfParserClient"]:
        """Open the connection, check the tool and the resource, close it at the end.

        `server` is the server's address, or in tests the server object itself,
        which FastMCP's client then reaches in memory.
        """
        server_label = server if isinstance(server, str) else f"in-memory server {server.name}"
        mcp_client = Client(server, timeout=call_timeout_seconds)
        # As in McpToolbox: a failure to connect is told apart from an error
        # raised later by the caller's own code inside the `async with` block.
        async with AsyncExitStack() as exit_stack:
            try:
                await exit_stack.enter_async_context(mcp_client)
            except Exception as error:
                raise PdfParserUnavailable(
                    f"cannot connect to pdf-parser-mcp at {server_label}; "
                    f"start it with: {START_SERVER_COMMAND}"
                ) from error
            pdf_parser = cls(mcp_client, server_label)
            await pdf_parser.check_server_offers_what_ingestion_needs()
            yield pdf_parser

    async def check_server_offers_what_ingestion_needs(self) -> None:
        """Fail before any work: the tool and the page resource must both exist."""
        try:
            offered_tools = await self._mcp_client.list_tools()
            offered_templates = await self._mcp_client.list_resource_templates()
        except Exception as error:
            raise PdfParserUnavailable(
                f"pdf-parser-mcp at {self._server_label} did not list what it offers: "
                f"{type(error).__name__}"
            ) from error
        missing_parts = []
        if INGEST_TOOL_NAME not in {tool.name for tool in offered_tools}:
            missing_parts.append(f"the tool {INGEST_TOOL_NAME}")
        if PAGE_RESOURCE_TEMPLATE not in {template.uri_template for template in offered_templates}:
            missing_parts.append(f"the resource {PAGE_RESOURCE_TEMPLATE}")
        if missing_parts:
            raise PdfParserUnavailable(
                f"pdf-parser-mcp at {self._server_label} is missing "
                f"{' and '.join(missing_parts)}; is it the right server?"
            )

    @traceable(run_type="tool", name="ingest_pdf", process_inputs=without_self)
    async def ingest_pdf(self, path: str, document_id: str) -> IngestedPdf:
        """Ask the server to read the PDF once and hold its pages."""
        try:
            call_result = await self._mcp_client.call_tool(
                INGEST_TOOL_NAME, {"path": path, "document_id": document_id}
            )
        except ToolError as error:
            # The server answered with a reason: this file was refused.
            raise PdfParserError(str(error)) from error
        except Exception as error:
            raise self.stopped_answering(INGEST_TOOL_NAME, error) from error
        try:
            return IngestedPdf.model_validate(call_result.structured_content)
        except ValidationError as error:
            raise PdfParserUnavailable(
                f"pdf-parser-mcp at {self._server_label} returned an unexpected result "
                f"for {INGEST_TOOL_NAME}: {error.error_count()} problems"
            ) from error

    async def read_page(self, document_id: str, page_number: int) -> str:
        """The text of one page, counted from 1; "" for a page with no text."""
        page_address = PAGE_RESOURCE_TEMPLATE.format(
            document_id=document_id, page_number=page_number
        )
        try:
            page_contents = await self._mcp_client.read_resource(page_address)
        except MCPError as error:
            # The server answered with a reason: no such document or page.
            raise PdfParserError(str(error)) from error
        except Exception as error:
            raise self.stopped_answering(page_address, error) from error
        page_text = getattr(page_contents[0], "text", None) if page_contents else None
        if not isinstance(page_text, str):
            raise PdfParserUnavailable(
                f"pdf-parser-mcp at {self._server_label} returned no text for {page_address}"
            )
        return page_text

    # One trace run for the whole document instead of one per page: 124 page
    # runs for the annual report would bury the ingestion in its trace.
    @traceable(
        run_type="tool",
        name="read_pdf_pages",
        process_inputs=without_self,
        process_outputs=summarise_pages,
    )
    async def read_all_pages(self, ingested_pdf: IngestedPdf) -> list[str]:
        """Every page's text, page 1 first, read one page at a time."""
        return [
            await self.read_page(ingested_pdf.document_id, page_number)
            for page_number in range(1, ingested_pdf.page_count + 1)
        ]

    def stopped_answering(self, what_was_asked: str, error: Exception) -> PdfParserUnavailable:
        """No answer at all: a lost connection or the call's time limit."""
        return PdfParserUnavailable(
            f"pdf-parser-mcp at {self._server_label} stopped answering during "
            f"{what_was_asked}: {type(error).__name__}"
        )
