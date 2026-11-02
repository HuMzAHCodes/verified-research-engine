"""The pages of the most recently ingested documents, held in memory.

CONCEPT: pdf-parser-mcp, decision 4. The server is started by hand and left
running, so a store without a limit would grow with every PDF ever ingested.
It holds at most a fixed number of documents and forgets the one ingested
longest ago when a new one arrives. Ingestion reads a document's pages right
after ingesting it, so a few recent documents are all it needs.
"""

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class DocumentNotHeld(LookupError):
    """The server does not hold this document: never ingested, or forgotten."""


class PageOutOfRange(LookupError):
    """The document has no page with this number."""


class DocumentPageStore:
    """Pages by document id, oldest forgotten first.

    An OrderedDict keeps the order documents were added in; adding a document
    again moves it to the newest end, so the oldest is always first.
    """

    def __init__(self, max_documents: int) -> None:
        self.max_documents = max_documents
        self.pages_by_document: OrderedDict[str, list[str]] = OrderedDict()

    def add(self, document_id: str, page_texts: list[str]) -> list[str]:
        """Hold a document's pages; return the ids of any documents forgotten to make room."""
        self.pages_by_document[document_id] = list(page_texts)
        self.pages_by_document.move_to_end(document_id)
        forgotten_ids: list[str] = []
        while len(self.pages_by_document) > self.max_documents:
            oldest_id, _ = self.pages_by_document.popitem(last=False)
            forgotten_ids.append(oldest_id)
            logger.info("forgot document %s to hold at most %d", oldest_id, self.max_documents)
        return forgotten_ids

    def page_text(self, document_id: str, page_number: int) -> str:
        """The text of one page, counted from 1 (decision 3)."""
        page_texts = self.pages_by_document.get(document_id)
        if page_texts is None:
            raise DocumentNotHeld(
                f"document {document_id} is not held by this server; call ingest_pdf first"
            )
        if not 1 <= page_number <= len(page_texts):
            raise PageOutOfRange(
                f"document {document_id} has pages 1 to {len(page_texts)}; "
                f"there is no page {page_number}"
            )
        return page_texts[page_number - 1]

    @property
    def held_document_ids(self) -> list[str]:
        """Held documents, oldest first."""
        return list(self.pages_by_document)
