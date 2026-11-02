"""pdf-parser-mcp's own settings.

CONCEPT: pdf-parser-mcp, decision 5 (D-31). The server runs on its own and
reads only what it needs: where to listen, which folder it may read, and its
limits. It never imports the project's settings.py, and it needs no API key,
so it can never hold one.

Read when the server starts (load_pdf_parser_config); tests build their own
config with a temporary folder instead.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 50 MB: annual reports are usually a few MB; the limit bounds the memory one
# request can use (decision 2).
DEFAULT_MAX_PDF_BYTES = 50 * 1024 * 1024


class PdfParserConfig(BaseSettings):
    """Settings for pdf-parser-mcp, from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Listen on this machine only, on the port after search-mcp's (D-46).
    pdf_parser_mcp_host: str = "127.0.0.1"
    pdf_parser_mcp_port: int = Field(default=8766, ge=1024, le=65535)

    # Decision 1: the only folder whose files may be read. Phase 8's upload
    # endpoint saves files here; data/ is ignored by git.
    pdf_uploads_dir: Path = Path("data/uploads")

    # Decision 2: larger files are refused before they are read.
    pdf_max_bytes: int = Field(default=DEFAULT_MAX_PDF_BYTES, ge=1_000)

    # Decision 4: how many documents' pages the server keeps in memory.
    pdf_max_documents_held: int = Field(default=10, ge=1, le=100)


def load_pdf_parser_config() -> PdfParserConfig:
    return PdfParserConfig()
