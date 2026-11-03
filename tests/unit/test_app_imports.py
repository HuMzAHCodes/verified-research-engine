"""The application reads PDFs only through pdf-parser-mcp.

CONCEPT: document-ingestion, decision 7 (D-48). Ingestion lives in app/ and
must reach PDFs through the MCP server, never by importing a PDF library or
the server's own code. Checked the same way as the graph (TC-MCP-04): every
.py file under app/ is read with the ast module, without running it.
"""

from pathlib import Path

import pytest

from tests.unit.test_graph_imports import imported_top_level_modules

APP_FOLDER = Path(__file__).resolve().parents[2] / "app"

FORBIDDEN_MODULES = {
    "pypdf": "PDF reading goes through pdf-parser-mcp",
    "fitz": "PDF reading goes through pdf-parser-mcp",
    "pdfplumber": "PDF reading goes through pdf-parser-mcp",
    "mcp_servers": "the application and the MCP servers are separate programs (D-31)",
}


def app_source_files() -> list[Path]:
    return sorted(APP_FOLDER.rglob("*.py"))


def test_the_check_reads_the_app_folder():
    # Guards the guard: a wrong path would make the other test pass vacuously.
    assert "pdf_parser_client.py" in {source_file.name for source_file in app_source_files()}


@pytest.mark.parametrize(
    "source_file", app_source_files(), ids=lambda path: path.relative_to(APP_FOLDER).as_posix()
)
def test_no_pdf_library_or_server_code_is_imported_by_the_app(source_file):
    forbidden_imports = {
        module_name: FORBIDDEN_MODULES[module_name]
        for module_name in imported_top_level_modules(source_file) & FORBIDDEN_MODULES.keys()
    }
    assert not forbidden_imports, f"{source_file.name} imports {forbidden_imports}"
