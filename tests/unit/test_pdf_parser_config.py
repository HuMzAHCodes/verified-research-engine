"""pdf-parser-mcp's settings: defaults, and values that are refused at start-up.

CONCEPT: pdf-parser-mcp, decision 5. The server's own settings hold no key,
so the defaults alone are enough to start it.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_servers.pdf_parser_mcp.config import DEFAULT_MAX_PDF_BYTES, PdfParserConfig

PDF_PARSER_VARIABLES = (
    "PDF_PARSER_MCP_HOST",
    "PDF_PARSER_MCP_PORT",
    "PDF_UPLOADS_DIR",
    "PDF_MAX_BYTES",
    "PDF_MAX_DOCUMENTS_HELD",
)


def test_defaults_need_no_key_and_listen_beside_search_mcp(monkeypatch):
    for variable in PDF_PARSER_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    defaults = PdfParserConfig(_env_file=None)

    assert (defaults.pdf_parser_mcp_host, defaults.pdf_parser_mcp_port) == ("127.0.0.1", 8766)
    assert defaults.pdf_uploads_dir == Path("data/uploads")
    assert defaults.pdf_max_bytes == DEFAULT_MAX_PDF_BYTES == 50 * 1024 * 1024
    assert defaults.pdf_max_documents_held == 10


@pytest.mark.parametrize(
    ("variable", "bad_value"),
    [
        ("PDF_PARSER_MCP_PORT", "80"),
        ("PDF_MAX_BYTES", "10"),
        ("PDF_MAX_DOCUMENTS_HELD", "0"),
        ("PDF_MAX_DOCUMENTS_HELD", "1000"),
    ],
)
def test_a_value_out_of_range_stops_the_server(monkeypatch, variable, bad_value):
    monkeypatch.setenv(variable, bad_value)
    with pytest.raises(ValidationError, match=variable.lower()):
        PdfParserConfig(_env_file=None)
