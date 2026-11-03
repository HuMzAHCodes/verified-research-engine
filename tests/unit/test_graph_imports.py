"""TC-MCP-04: no tool library is imported anywhere in graph/.

CONCEPT: tools-through-mcp, decision 7 (D-37). The graph reaches tools only
through MCP servers. This test reads every .py file under graph/ with Python's
ast module, without running it, and fails if any import names a library that
would let the graph search, read pages, or read PDFs by itself, or anything
from mcp_servers/ (the graph and the servers are separate programs, D-31).

CONCEPT: document-ingestion, choice C (D-48). It also fails if the graph
imports app/: the application uses the graph (ingestion writes through
graph/document_store.py), never the other way round.

Static on purpose: it checks every file, including code no test happens to
run, and needs no network. Extend FORBIDDEN_MODULES when a new tool library
joins the project.
"""

import ast
from pathlib import Path

import pytest

GRAPH_FOLDER = Path(__file__).resolve().parents[2] / "graph"

FORBIDDEN_MODULES = {
    "tavily": "search goes through search-mcp",
    "trafilatura": "page reading goes through search-mcp",
    "bs4": "page reading goes through search-mcp",
    "requests": "page reading goes through search-mcp",
    "pypdf": "PDF reading goes through pdf-parser-mcp (Phase 4)",
    "fitz": "PDF reading goes through pdf-parser-mcp (Phase 4)",
    "pdfplumber": "PDF reading goes through pdf-parser-mcp (Phase 4)",
    "mcp_servers": "the graph and the MCP servers are separate programs (D-31)",
    "app": "the application uses the graph, never the reverse (D-48, choice C)",
}


def imported_top_level_modules(source_file: Path) -> set[str]:
    """The first part of every module the file imports ("tavily" for "tavily.client")."""
    syntax_tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    module_names: set[str] = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            module_names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            module_names.add(node.module.split(".")[0])
    return module_names


def graph_source_files() -> list[Path]:
    return sorted(GRAPH_FOLDER.rglob("*.py"))


def test_the_check_reads_the_graph_folder():
    # Guards the guard: a wrong path would make every other test pass vacuously.
    source_names = {source_file.name for source_file in graph_source_files()}
    assert {"supervisor.py", "researcher.py", "mcp_client.py", "document_store.py"} <= source_names


@pytest.mark.parametrize(
    "source_file", graph_source_files(), ids=lambda path: path.relative_to(GRAPH_FOLDER).as_posix()
)
def test_tc_mcp_04_no_tool_library_is_imported_by_the_graph(source_file):
    forbidden_imports = {
        module_name: FORBIDDEN_MODULES[module_name]
        for module_name in imported_top_level_modules(source_file) & FORBIDDEN_MODULES.keys()
    }
    assert not forbidden_imports, f"{source_file.name} imports {forbidden_imports}"


def test_the_check_catches_a_forbidden_import(tmp_path):
    # The detector itself, on a file written to break the rule both ways.
    offending_file = tmp_path / "offending.py"
    offending_file.write_text(
        "import os\n"
        "from tavily import AsyncTavilyClient\n"
        "import mcp_servers.search_mcp.fetch\n"
        "from app.ingestion import IngestionService\n",
        encoding="utf-8",
    )

    assert imported_top_level_modules(offending_file) & FORBIDDEN_MODULES.keys() == {
        "tavily",
        "mcp_servers",
        "app",
    }
