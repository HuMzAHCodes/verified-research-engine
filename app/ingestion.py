"""Document ingestion: an uploaded PDF becomes stored, searchable chunks.

CONCEPT: document-ingestion (D-44, D-47, D-48). One call does the whole job:

    fingerprint the file    already stored?  return that document, embed nothing   (decision 5)
    read it over MCP        ingest_pdf, then every page, through pdf-parser-mcp     (decision 7)
    refuse a scanned PDF    no text on any page: nothing to search                  (decision 6)
    cut each page           at most 600 characters, 150 overlapping, inside a page  (decisions 1, 2)
    embed the chunks        through the gateway's embed tier, 32 per request        (decision 3)
    store it                all or nothing, in one step                             (decision 4)

Phase 8 puts an upload endpoint in front of IngestionService; until then the
notebook and the live check call it directly (D-44). The service uses the
graph's gateway and store, never the reverse (choice C).
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastmcp import FastMCP
from langsmith import traceable

from app.pdf_parser_client import PdfParserClient, PdfParserError
from graph.document_store import (
    DocumentStore,
    DocumentStoreError,
    StoredChunk,
    StoredDocument,
    chunk_id_for,
)
from graph.llm import ModelCallError, ModelGateway
from settings import settings

logger = logging.getLogger(__name__)

# FR-DOC-03 and DATABASE_SCHEMA.md: chunks of at most 600 characters, each
# starting at least 150 characters before the previous one ended.
CHUNK_MAX_CHARS = 600
CHUNK_OVERLAP_CHARS = 150
FINGERPRINT_READ_BYTES = 1024 * 1024


@dataclass(frozen=True)
class PageSlice:
    """Where a chunk sits in its page: page_text[start:end] is the chunk's text."""

    start: int
    end: int


def is_word_start(page_text: str, position: int) -> bool:
    """True when a word begins here: a non-space character after a space or the start."""
    return not page_text[position].isspace() and (
        position == 0 or page_text[position - 1].isspace()
    )


def last_word_start_between(page_text: str, lowest: int, highest: int) -> int | None:
    """The last position in [lowest, highest] where a word begins, if there is one."""
    for position in range(highest, lowest - 1, -1):
        if is_word_start(page_text, position):
            return position
    return None


def first_word_start_from(page_text: str, position: int) -> int | None:
    """The first position at or after `position` where a word begins, if there is one."""
    for candidate in range(position, len(page_text)):
        if is_word_start(page_text, candidate):
            return candidate
    return None


def last_space_between(page_text: str, lowest: int, highest: int) -> int | None:
    """The last position in [lowest, highest] holding whitespace, if there is one."""
    for position in range(min(highest, len(page_text) - 1), lowest - 1, -1):
        if page_text[position].isspace():
            return position
    return None


def without_trailing_spaces(page_text: str, start: int, end: int) -> int:
    """`end` moved back over spaces, so a chunk never ends with whitespace."""
    while end > start and page_text[end - 1].isspace():
        end -= 1
    return end


def chunk_page(
    page_text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[PageSlice]:
    """Cut one page into overlapping windows that end between words.

    CONCEPT: document-ingestion, decisions 1 and 2. The rules, each checked by
    the property tests in tests/unit/test_chunking.py:

    - every chunk is an exact slice of the page, never reworded or cleaned;
    - every chunk is at most max_chars long and neither starts nor ends with
      whitespace;
    - a chunk ends just before a space, so no word is cut, unless one word is
      longer than max_chars (then it is cut at the limit: there is no space,
      and the next chunk continues inside that word);
    - each chunk starts at the start of a word at least overlap_chars before
      the previous chunk ended, so a sentence near a boundary is whole in one
      of the two; the overlap is shorter only when the previous chunk is too
      short to give it (a word longer than a chunk right after its start);
    - every non-space character of the page is in at least one chunk.

    Not LangChain's RecursiveCharacterTextSplitter: tried first, it kept no
    overlap at 82% of chunk boundaries, because it overlaps only inside the
    level it splits at (document-ingestion notes, decision 2).
    """
    first_word = first_word_start_from(page_text, 0)
    if first_word is None:
        return []  # a page with no text gives no chunks

    page_slices: list[PageSlice] = []
    start = first_word
    while True:
        end_of_text = without_trailing_spaces(page_text, start, len(page_text))
        if end_of_text - start <= max_chars:
            page_slices.append(PageSlice(start, end_of_text))
            return page_slices

        # End just before the last space inside the window, so no word is cut.
        last_space = last_space_between(page_text, start + 1, start + max_chars)
        if last_space is not None:
            end = without_trailing_spaces(page_text, start, last_space)
        else:
            end = start + max_chars  # one word longer than a chunk: cut it at the limit
        page_slices.append(PageSlice(start, end))

        # The next chunk starts at a word beginning at least overlap_chars
        # before this one ended, but always after this one's start.
        next_start = last_word_start_between(page_text, start + 1, end - overlap_chars)
        if next_start is None:
            # This chunk is too short to give that overlap. Take the next word,
            # unless that would skip text: then this chunk was cut inside a
            # word longer than a chunk, and the next one continues that word.
            following_word = first_word_start_from(page_text, start + 1)
            skipped_until = following_word if following_word is not None else len(page_text)
            if page_text[end:skipped_until].strip():
                next_start = max(start + 1, end - overlap_chars)
            else:
                next_start = following_word
        if next_start is None or next_start >= end_of_text:
            return page_slices
        start = next_start


def file_fingerprint(pdf_path: Path) -> str:
    """The SHA-256 of the file's bytes, read a megabyte at a time (decision 5)."""
    digest = hashlib.sha256()
    with pdf_path.open("rb") as pdf_file:
        while block := pdf_file.read(FINGERPRINT_READ_BYTES):
            digest.update(block)
    return digest.hexdigest()


def build_chunks(document_id: str, page_texts: list[str]) -> list[StoredChunk]:
    """Every page's chunks, numbered across the document, each with its page."""
    stored_chunks: list[StoredChunk] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        for page_slice in chunk_page(page_text):
            chunk_index = len(stored_chunks)
            stored_chunks.append(
                StoredChunk(
                    chunk_id=chunk_id_for(document_id, chunk_index),
                    document_id=document_id,
                    page=page_number,
                    chunk_index=chunk_index,
                    text=page_text[page_slice.start : page_slice.end],
                )
            )
    return stored_chunks


class IngestionError(RuntimeError):
    """Ingestion stopped and stored nothing; the message names the file and the cause."""


class ScannedPdfRefused(IngestionError):
    """The PDF has no text on any page, probably scanned; there is no OCR (decision 6)."""


@dataclass(frozen=True)
class IngestionResult:
    """What one call of ingest() did."""

    document: StoredDocument
    already_stored: bool  # True: the same file was stored before; nothing was embedded
    seconds: float


class IngestionService:
    """Turns a PDF in the uploads folder into a stored document."""

    def __init__(
        self,
        gateway: ModelGateway,
        store: DocumentStore,
        pdf_parser_server: str | FastMCP | None = None,
    ) -> None:
        self.gateway = gateway
        self.store = store
        # The server's address, or in tests the server object itself.
        self.pdf_parser_server = pdf_parser_server or settings.vre_pdf_parser_mcp_url

    @traceable(
        run_type="chain",
        name="ingest_document",
        process_inputs=lambda inputs: {"pdf_path": str(inputs["pdf_path"])},
    )
    async def ingest(self, pdf_path: Path | str) -> IngestionResult:
        """Ingest one PDF, or return the stored document if this file is already in.

        Raises ScannedPdfRefused for a PDF with no text, and IngestionError for
        anything else that stops it: the file refused or the server down, an
        embedding that fails after its retries (TC-DOC-06), or the store
        refusing the write. In every case nothing of the document is stored.
        """
        started_at = time.perf_counter()
        pdf_path = Path(pdf_path)
        self.store.remove_unfinished_saves()
        if not pdf_path.is_file():
            raise IngestionError(f"{pdf_path}: no such file")

        fingerprint = file_fingerprint(pdf_path)
        stored_document = self.store.find_by_fingerprint(fingerprint)
        if stored_document is not None:
            logger.info(
                "%s: already stored as %s; nothing embedded",
                pdf_path.name,
                stored_document.document_id,
            )
            return IngestionResult(stored_document, True, time.perf_counter() - started_at)

        # A new id for every new document, never derived from the file name,
        # so two files with the same name never collide.
        document_id = uuid.uuid4().hex
        try:
            page_texts = await self.read_pages(pdf_path, document_id)
            stored_chunks = build_chunks(document_id, page_texts)
            vectors = await self.gateway.embed(
                [stored_chunk.text for stored_chunk in stored_chunks],
                prompt_id=f"INGEST:{document_id}",
            )
            stored_document = StoredDocument(
                document_id=document_id,
                file_name=pdf_path.name,
                fingerprint=fingerprint,
                page_count=len(page_texts),
                chunk_count=len(stored_chunks),
                embedding_model=settings.vre_embed_model,
            )
            self.store.save(stored_document, stored_chunks, vectors)
        except ScannedPdfRefused:
            raise
        except (PdfParserError, ModelCallError, DocumentStoreError) as error:
            raise IngestionError(
                f"{pdf_path.name}: ingestion stopped and nothing was stored: {error}"
            ) from error

        seconds = time.perf_counter() - started_at
        logger.info(
            "%s: stored as %s: %d pages, %d chunks, in %.1f s",
            pdf_path.name,
            document_id,
            stored_document.page_count,
            stored_document.chunk_count,
            seconds,
        )
        return IngestionResult(stored_document, False, seconds)

    async def read_pages(self, pdf_path: Path, document_id: str) -> list[str]:
        """Every page's text through pdf-parser-mcp; refuse a PDF with none (decision 6)."""
        async with PdfParserClient.connect(self.pdf_parser_server) as pdf_parser:
            ingested_pdf = await pdf_parser.ingest_pdf(str(pdf_path), document_id)
            if not ingested_pdf.has_text:
                raise ScannedPdfRefused(
                    f"{pdf_path.name}: this PDF has no text on any of its "
                    f"{ingested_pdf.page_count} pages (it may be scanned); "
                    "OCR is not supported"
                )
            return await pdf_parser.read_all_pages(ingested_pdf)
