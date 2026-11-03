"""Document store: the embedded chunks of every ingested document, one folder each.

CONCEPT: document-ingestion, choice A (D-47) and decision 4 (D-48). Each
document lives in data/vectors/<document_id>/:

    document.json   file name, SHA-256 fingerprint, page count, chunk count, model
    chunks.jsonl    one chunk per line, in index order: id, page, position, text
    index.faiss     the chunks' vectors, normalised, in an exact inner-product index

FAISS is an index, not a database: it stores vectors and answers with their
positions and scores. Line i of chunks.jsonl is the chunk behind vector i, so
this module turns positions back into chunks; that is the one part of a vector
database this project needs.

Ingestion writes through save(); the document researcher reads through
search(). The store sits in graph/ because runs read it, and ingestion in app/
uses it from outside (choice C): imports point from app/ to graph/, never back.
"""

import json
import logging
import re
import shutil
import uuid
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel, Field

from settings import settings

logger = logging.getLogger(__name__)

DOCUMENT_FILE_NAME = "document.json"
CHUNKS_FILE_NAME = "chunks.jsonl"
INDEX_FILE_NAME = "index.faiss"
# A folder being written starts with this, so readers can tell it is unfinished.
UNFINISHED_SAVE_PREFIX = ".unfinished-"
# The same rule as pdf-parser-mcp's document ids (D-46): letters, digits, '-'
# and '_', so an id can never point outside the store ("../other"). Repeated
# here because graph/ does not import from mcp_servers/.
DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


class DocumentStoreError(RuntimeError):
    """The store cannot do what was asked: bad input, or a document already there."""


class DocumentNotStored(LookupError):
    """A document id with no complete folder in the store."""


class StoredDocument(BaseModel):
    """What document.json holds: the facts about one stored document."""

    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    file_name: str
    fingerprint: str = Field(description="SHA-256 of the file's bytes, in hex")
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    embedding_model: str


class StoredChunk(BaseModel):
    """One line of chunks.jsonl: a chunk and where it comes from."""

    chunk_id: str
    document_id: str
    page: int = Field(ge=1, description="The page the chunk comes from, counted from 1")
    chunk_index: int = Field(ge=0, description="Position in the document, counted from 0")
    text: str


class RetrievedChunk(StoredChunk):
    """A chunk found by a search, with its cosine similarity to the question."""

    score: float


def chunk_id_for(document_id: str, chunk_index: int) -> str:
    """The id of a chunk: "<document_id>:<chunk_index>" (DATABASE_SCHEMA.md)."""
    return f"{document_id}:{chunk_index}"


def default_vectors_dir() -> Path:
    """Where the store lives unless told otherwise: data/vectors/, ignored by git."""
    return settings.vre_data_dir / "vectors"


def normalised_vectors(vectors: list[list[float]]) -> np.ndarray:
    """Vectors as a float32 array, each scaled to length 1.

    With every vector of length 1, the inner product FAISS computes is the
    cosine similarity: 1.0 for the same direction, lower for less alike.
    """
    vector_array = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vector_array)
    return vector_array


class DocumentStore:
    """Writes documents in one step and searches only the documents asked for."""

    def __init__(self, vectors_dir: Path | None = None) -> None:
        self.vectors_dir = vectors_dir or default_vectors_dir()
        # Loaded documents, kept in memory: a document never changes once its
        # folder exists, so an index read once stays correct.
        self._loaded_documents: dict[str, tuple[faiss.Index, list[StoredChunk]]] = {}

    def document_dir(self, document_id: str) -> Path:
        """The folder of a document, after checking the id cannot leave the store."""
        if re.fullmatch(DOCUMENT_ID_PATTERN, document_id) is None:
            raise DocumentStoreError(
                f"document id {document_id!r} must be letters, digits, '-' or '_' "
                "(at most 64 characters)"
            )
        return self.vectors_dir / document_id

    def has_document(self, document_id: str) -> bool:
        """True when the document's folder is complete (document.json is in it)."""
        return (self.document_dir(document_id) / DOCUMENT_FILE_NAME).is_file()

    def save(
        self,
        document: StoredDocument,
        chunks: list[StoredChunk],
        vectors: list[list[float]],
    ) -> None:
        """Write a document, all of it or nothing (decision 4, TC-DOC-06).

        Everything is written into a temporary folder beside the store, then
        that folder is renamed to the document's id. A rename within one disk
        is a single step, so a reader sees either no folder or a complete one.
        If anything fails before the rename, the temporary folder is removed
        and the error is raised: nothing of the document is left behind.
        """
        check_document_matches_chunks(document, chunks, vectors)
        final_dir = self.document_dir(document.document_id)
        if final_dir.exists():
            raise DocumentStoreError(f"document {document.document_id!r} is already stored")

        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        unfinished_dir = self.vectors_dir / (
            f"{UNFINISHED_SAVE_PREFIX}{document.document_id}-{uuid.uuid4().hex[:8]}"
        )
        unfinished_dir.mkdir()
        try:
            vector_array = normalised_vectors(vectors)
            vector_index = faiss.IndexFlatIP(vector_array.shape[1])
            vector_index.add(vector_array)
            faiss.write_index(vector_index, str(unfinished_dir / INDEX_FILE_NAME))
            chunk_lines = "".join(chunk.model_dump_json() + "\n" for chunk in chunks)
            (unfinished_dir / CHUNKS_FILE_NAME).write_text(chunk_lines, encoding="utf-8")
            # Written last: has_document() treats a folder with document.json
            # as complete, so it must never be there before the rest.
            (unfinished_dir / DOCUMENT_FILE_NAME).write_text(
                document.model_dump_json(indent=2), encoding="utf-8"
            )
            unfinished_dir.rename(final_dir)
        except Exception:
            shutil.rmtree(unfinished_dir, ignore_errors=True)
            raise
        logger.info(
            "%s: stored %d chunks from %d pages",
            document.document_id,
            document.chunk_count,
            document.page_count,
        )

    def load_document(self, document_id: str) -> StoredDocument:
        """The facts about a stored document, from its document.json."""
        document_file = self.document_dir(document_id) / DOCUMENT_FILE_NAME
        if not document_file.is_file():
            raise DocumentNotStored(f"document {document_id!r} is not in the store")
        return StoredDocument.model_validate_json(document_file.read_text(encoding="utf-8"))

    def find_by_fingerprint(self, fingerprint: str) -> StoredDocument | None:
        """The stored document whose file had these bytes, if there is one (decision 5).

        Unfinished saves are skipped: they are not documents yet.
        """
        if not self.vectors_dir.is_dir():
            return None
        for document_file in sorted(self.vectors_dir.glob(f"*/{DOCUMENT_FILE_NAME}")):
            if document_file.parent.name.startswith(UNFINISHED_SAVE_PREFIX):
                continue
            stored_document = StoredDocument.model_validate_json(
                document_file.read_text(encoding="utf-8")
            )
            if stored_document.fingerprint == fingerprint:
                return stored_document
        return None

    def remove_unfinished_saves(self) -> list[str]:
        """Delete temporary folders left by a save that was cut off, and name them.

        A save that raises cleans up after itself; only a process killed in the
        middle of one (closed window, power cut) leaves a folder behind.
        Ingestion calls this before it starts.
        """
        if not self.vectors_dir.is_dir():
            return []
        removed_names = []
        for leftover_dir in sorted(self.vectors_dir.glob(f"{UNFINISHED_SAVE_PREFIX}*")):
            shutil.rmtree(leftover_dir, ignore_errors=True)
            removed_names.append(leftover_dir.name)
            logger.warning("removed an unfinished save: %s", leftover_dir.name)
        return removed_names

    def search(
        self,
        document_ids: list[str],
        query_vector: list[float],
        k: int,
    ) -> list[RetrievedChunk]:
        """The k chunks closest to the question, from these documents only, best first.

        CONCEPT: document-ingestion, choice A. Only the indexes of document_ids
        are opened, so a run can never retrieve another run's documents: the
        rule (TC-RES-10) holds by what is loaded, not by a filter. Each index
        is searched exactly, the results are merged, and the best k are kept.
        """
        if k < 1:
            raise DocumentStoreError(f"k must be at least 1, not {k}")
        if not document_ids:
            return []
        query_array = normalised_vectors([query_vector])

        retrieved_chunks: list[RetrievedChunk] = []
        for document_id in dict.fromkeys(document_ids):  # each document once, in order
            vector_index, stored_chunks = self._load(document_id)
            if vector_index.d != query_array.shape[1]:
                raise DocumentStoreError(
                    f"the question's vector has {query_array.shape[1]} numbers, but "
                    f"{document_id!r} was embedded with {vector_index.d}"
                )
            scores, positions = vector_index.search(query_array, min(k, vector_index.ntotal))
            for score, position in zip(scores[0], positions[0], strict=True):
                retrieved_chunks.append(
                    RetrievedChunk(**stored_chunks[position].model_dump(), score=float(score))
                )

        retrieved_chunks.sort(key=lambda retrieved_chunk: retrieved_chunk.score, reverse=True)
        return retrieved_chunks[:k]

    def _load(self, document_id: str) -> tuple[faiss.Index, list[StoredChunk]]:
        """A document's index and chunks, read from disk the first time only."""
        if document_id not in self._loaded_documents:
            stored_document = self.load_document(document_id)
            document_dir = self.document_dir(document_id)
            vector_index = faiss.read_index(str(document_dir / INDEX_FILE_NAME))
            chunk_lines = (document_dir / CHUNKS_FILE_NAME).read_text(encoding="utf-8")
            stored_chunks = [
                StoredChunk.model_validate(json.loads(line)) for line in chunk_lines.splitlines()
            ]
            if not len(stored_chunks) == vector_index.ntotal == stored_document.chunk_count:
                raise DocumentStoreError(
                    f"{document_id!r} is damaged: {len(stored_chunks)} chunks, "
                    f"{vector_index.ntotal} vectors, {stored_document.chunk_count} expected"
                )
            self._loaded_documents[document_id] = (vector_index, stored_chunks)
        return self._loaded_documents[document_id]


def check_document_matches_chunks(
    document: StoredDocument,
    chunks: list[StoredChunk],
    vectors: list[list[float]],
) -> None:
    """Refuse to save a document whose parts do not line up.

    Line i of chunks.jsonl must be the chunk behind vector i; if the counts or
    positions were off, every search would return the wrong text for a vector.
    """
    if not len(chunks) == len(vectors) == document.chunk_count:
        raise DocumentStoreError(
            f"{document.document_id!r}: {len(chunks)} chunks, {len(vectors)} vectors, "
            f"document.json says {document.chunk_count}"
        )
    for position, chunk in enumerate(chunks):
        expected_id = chunk_id_for(document.document_id, position)
        if chunk.chunk_index != position or chunk.chunk_id != expected_id:
            raise DocumentStoreError(
                f"chunk at position {position} is {chunk.chunk_id!r} "
                f"(index {chunk.chunk_index}); expected {expected_id!r}"
            )
        if chunk.document_id != document.document_id:
            raise DocumentStoreError(
                f"chunk {chunk.chunk_id!r} belongs to {chunk.document_id!r}, "
                f"not {document.document_id!r}"
            )
        if chunk.page > document.page_count:
            raise DocumentStoreError(
                f"chunk {chunk.chunk_id!r} is on page {chunk.page}, "
                f"but the document has {document.page_count} pages"
            )
    if len({len(vector) for vector in vectors}) != 1:
        raise DocumentStoreError(f"{document.document_id!r}: vectors of different lengths")
