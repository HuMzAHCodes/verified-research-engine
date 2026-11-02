"""Checking a PDF and reading its text, page by page.

CONCEPT: pdf-parser-mcp. Every check runs before the library reads anything,
cheapest first, so a bad request fails with a sentence that says what to do
instead of an error from inside the library:

1. the path lies inside the allowed folder        (decision 1)
2. the file exists and is at most the size limit  (decision 2)
3. its first bytes are %PDF-                      (decision 2)
4. it is not password-protected                   (decision 2)

Then every page's text is read with pypdf (open choice A), in order, counted
from 1 by the caller; a page with no text becomes empty text, so page numbers
never shift (decision 3). pypdf is imported in this file only: switching to
another library changes this file and nothing else.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PageObject, PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"


class PdfReadError(RuntimeError):
    """The file was refused or could not be read; the message says why."""


@dataclass(frozen=True)
class PdfLimits:
    """Where PDFs may be read from, and how large they may be."""

    allowed_folder: Path
    max_bytes: int


def resolve_allowed_path(path_text: str, allowed_folder: Path) -> Path:
    """The file's real, absolute path, if it lies inside the allowed folder.

    CONCEPT: pdf-parser-mcp, decision 1. resolve() follows ".." and links to
    where the path really points, so "data/uploads/../../secret.txt" and a
    link that leads elsewhere are both caught. A relative path is taken from
    the folder the server was started in, the project root.
    """
    allowed_root = allowed_folder.resolve()
    requested_path = Path(path_text).resolve()
    if not requested_path.is_relative_to(allowed_root):
        raise PdfReadError(
            f"the path is outside the folder this server may read ({allowed_folder}); "
            "copy the PDF into that folder first"
        )
    return requested_path


def check_pdf_file(pdf_path: Path, max_bytes: int) -> None:
    """Refuse a missing file, a file over the size limit, or a file that is not a PDF."""
    if not pdf_path.is_file():
        raise PdfReadError(f"no file named {pdf_path.name} in the allowed folder")
    size_in_bytes = pdf_path.stat().st_size
    if size_in_bytes > max_bytes:
        raise PdfReadError(
            f"{pdf_path.name} is {size_in_bytes / 1_048_576:.1f} MB, larger than the "
            f"{max_bytes / 1_048_576:.0f} MB this server reads"
        )
    with pdf_path.open("rb") as pdf_file:
        first_bytes = pdf_file.read(len(PDF_SIGNATURE))
    if first_bytes != PDF_SIGNATURE:
        raise PdfReadError(f"{pdf_path.name} is not a PDF (it does not start with %PDF-)")


def read_pdf_pages(path_text: str, limits: PdfLimits) -> list[str]:
    """Every page's text, in order; a page with no text is an empty string.

    The list's first item is page 1. Raises PdfReadError, with the reason, for
    anything the checks refuse or the library cannot read.
    """
    pdf_path = resolve_allowed_path(path_text, limits.allowed_folder)
    check_pdf_file(pdf_path, limits.max_bytes)
    try:
        pdf_reader = PdfReader(pdf_path)
        if pdf_reader.is_encrypted:
            raise PdfReadError(
                f"{pdf_path.name} is password-protected; remove the password and upload it again"
            )
        page_texts: list[str] = []
        for page_index, pdf_page in enumerate(pdf_reader.pages):
            page_texts.append(read_page_text(pdf_path.name, page_index + 1, pdf_page))
    except PyPdfError as error:
        raise PdfReadError(f"could not read {pdf_path.name} as a PDF: {error}") from error
    return page_texts


def read_page_text(file_name: str, page_number: int, pdf_page: PageObject) -> str:
    """One page's text; a page the library cannot read counts as a page with no text.

    CONCEPT: pdf-parser-mcp, decision 3. Keeping the page, empty, keeps every
    later page's number right; the reason is logged so it is not lost.
    """
    try:
        return pdf_page.extract_text() or ""
    except Exception as error:  # a single damaged page must not lose the document
        logger.warning("%s, page %d: no text read (%s)", file_name, page_number, error)
        return ""
