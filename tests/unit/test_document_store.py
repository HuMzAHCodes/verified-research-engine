"""Tests of the document store: one folder per document, written in one step.

CONCEPT: document-ingestion, choice A (D-47) and decision 4. A save writes a
complete folder or nothing (TC-DOC-06, the store's part); a search opens only
the documents asked for (the ground TC-RES-10 stands on), returns the best
chunks first with their pages, and scores by cosine similarity. Every test
uses its own temporary folder, so nothing touches data/.
"""

import json

import faiss
import pytest

import graph.document_store
from graph.document_store import (
    CHUNKS_FILE_NAME,
    DOCUMENT_FILE_NAME,
    INDEX_FILE_NAME,
    UNFINISHED_SAVE_PREFIX,
    DocumentNotStored,
    DocumentStore,
    DocumentStoreError,
    StoredChunk,
    StoredDocument,
    chunk_id_for,
)

# Three directions in a small space, so the closest chunk is obvious.
EAST = [1.0, 0.0, 0.0]
NORTH = [0.0, 1.0, 0.0]
UP = [0.0, 0.0, 1.0]


def make_document(
    document_id: str, chunk_texts: list[str], fingerprint: str = "fingerprint-1"
) -> tuple[StoredDocument, list[StoredChunk]]:
    """A document whose chunk i is on page i + 1."""
    chunks = [
        StoredChunk(
            chunk_id=chunk_id_for(document_id, chunk_index),
            document_id=document_id,
            page=chunk_index + 1,
            chunk_index=chunk_index,
            text=chunk_text,
        )
        for chunk_index, chunk_text in enumerate(chunk_texts)
    ]
    document = StoredDocument(
        document_id=document_id,
        file_name=f"{document_id}.pdf",
        fingerprint=fingerprint,
        page_count=len(chunk_texts),
        chunk_count=len(chunk_texts),
        embedding_model="mistral-embed",
    )
    return document, chunks


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    return DocumentStore(tmp_path / "vectors")


def test_a_saved_document_has_its_three_files_and_reads_back(store):
    document, chunks = make_document("annual-report", ["Revenue grew.", "Risks remain."])

    store.save(document, chunks, [EAST, NORTH])

    document_dir = store.vectors_dir / "annual-report"
    assert sorted(path.name for path in document_dir.iterdir()) == sorted(
        [DOCUMENT_FILE_NAME, CHUNKS_FILE_NAME, INDEX_FILE_NAME]
    )
    assert store.has_document("annual-report")
    assert store.load_document("annual-report") == document
    chunk_lines = (document_dir / CHUNKS_FILE_NAME).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in chunk_lines] == ["Revenue grew.", "Risks remain."]


def test_search_returns_the_closest_chunks_first_with_their_pages(store):
    document, chunks = make_document("annual-report", ["east text", "north text", "up text"])
    store.save(document, chunks, [EAST, NORTH, UP])

    # A question pointing mostly north, a little east.
    retrieved = store.search(["annual-report"], [0.3, 0.9, 0.0], k=2)

    assert [chunk.text for chunk in retrieved] == ["north text", "east text"]
    assert [chunk.page for chunk in retrieved] == [2, 1]
    assert retrieved[0].chunk_id == "annual-report:1"
    assert retrieved[0].score > retrieved[1].score


def test_scores_are_cosine_similarity_whatever_the_vector_lengths(store):
    # The stored vector is long, the question short: same direction, score 1.
    document, chunks = make_document("annual-report", ["east text"])
    store.save(document, chunks, [[40.0, 0.0, 0.0]])

    (retrieved_chunk,) = store.search(["annual-report"], [0.5, 0.0, 0.0], k=1)

    assert retrieved_chunk.score == pytest.approx(1.0)


def test_search_never_returns_chunks_from_a_document_not_asked_for(store):
    first_document, first_chunks = make_document("report-a", ["a east", "a north"], "fp-a")
    second_document, second_chunks = make_document("report-b", ["b up exact match"], "fp-b")
    store.save(first_document, first_chunks, [EAST, NORTH])
    store.save(second_document, second_chunks, [UP])

    # The question points straight at report-b's chunk, but only report-a is asked for.
    retrieved = store.search(["report-a"], UP, k=5)

    assert {chunk.document_id for chunk in retrieved} == {"report-a"}
    assert len(retrieved) == 2  # k larger than the document: every chunk, no padding


def test_search_across_documents_merges_and_keeps_the_best_k(store):
    first_document, first_chunks = make_document("report-a", ["a east", "a north"], "fp-a")
    second_document, second_chunks = make_document("report-b", ["b up", "b east"], "fp-b")
    store.save(first_document, first_chunks, [EAST, NORTH])
    store.save(second_document, second_chunks, [UP, [0.9, 0.1, 0.0]])

    retrieved = store.search(["report-a", "report-b"], EAST, k=2)

    assert [chunk.text for chunk in retrieved] == ["a east", "b east"]


def test_no_documents_asked_for_means_no_results(store):
    assert store.search([], EAST, k=3) == []


def test_an_unknown_document_is_refused(store):
    with pytest.raises(DocumentNotStored, match="missing-report"):
        store.search(["missing-report"], EAST, k=3)


@pytest.mark.parametrize("bad_document_id", ["../outside", "with space", "", "a" * 65])
def test_a_document_id_that_could_leave_the_store_is_refused(store, bad_document_id):
    with pytest.raises(DocumentStoreError, match="must be letters"):
        store.search([bad_document_id], EAST, k=3)


def test_a_question_vector_of_the_wrong_length_is_refused(store):
    document, chunks = make_document("annual-report", ["east text"])
    store.save(document, chunks, [EAST])

    with pytest.raises(DocumentStoreError, match="embedded with 3"):
        store.search(["annual-report"], [1.0, 0.0], k=1)


def test_find_by_fingerprint_returns_the_stored_document_or_none(store):
    document, chunks = make_document("annual-report", ["east text"], fingerprint="abc123")
    store.save(document, chunks, [EAST])

    assert store.find_by_fingerprint("abc123") == document
    assert store.find_by_fingerprint("another-file") is None
    assert DocumentStore(store.vectors_dir / "not-created").find_by_fingerprint("abc123") is None


def test_a_document_cannot_be_saved_twice(store):
    document, chunks = make_document("annual-report", ["east text"])
    store.save(document, chunks, [EAST])

    with pytest.raises(DocumentStoreError, match="already stored"):
        store.save(document, chunks, [EAST])


def test_tc_doc_06_a_save_that_fails_leaves_nothing_behind(store, monkeypatch):
    document, chunks = make_document("annual-report", ["east text", "north text"])

    def failing_write_index(vector_index, path):
        raise OSError("disk full")

    monkeypatch.setattr(graph.document_store.faiss, "write_index", failing_write_index)

    with pytest.raises(OSError, match="disk full"):
        store.save(document, chunks, [EAST, NORTH])

    # No document folder, and no unfinished folder either.
    assert not store.has_document("annual-report")
    assert list(store.vectors_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("chunk_count_change", "vector_count", "expected_message"),
    [
        (0, 1, "2 chunks, 1 vectors"),
        (1, 2, "document.json says 3"),
    ],
)
def test_parts_that_do_not_line_up_are_refused(
    store, chunk_count_change, vector_count, expected_message
):
    document, chunks = make_document("annual-report", ["east text", "north text"])
    document = document.model_copy(update={"chunk_count": 2 + chunk_count_change})

    with pytest.raises(DocumentStoreError, match=expected_message):
        store.save(document, chunks, [EAST, NORTH][:vector_count])

    assert not store.vectors_dir.exists() or list(store.vectors_dir.iterdir()) == []


def test_chunks_out_of_order_are_refused(store):
    document, chunks = make_document("annual-report", ["east text", "north text"])

    with pytest.raises(DocumentStoreError, match="expected 'annual-report:0'"):
        store.save(document, list(reversed(chunks)), [EAST, NORTH])


def test_an_unfinished_save_is_not_a_document_and_can_be_removed(store):
    document, chunks = make_document("annual-report", ["east text"], fingerprint="abc123")
    leftover_dir = store.vectors_dir / f"{UNFINISHED_SAVE_PREFIX}annual-report-1234abcd"
    leftover_dir.mkdir(parents=True)
    # A process killed after writing document.json but before the rename.
    (leftover_dir / DOCUMENT_FILE_NAME).write_text(document.model_dump_json(), encoding="utf-8")

    assert store.find_by_fingerprint("abc123") is None
    assert store.remove_unfinished_saves() == [leftover_dir.name]
    assert not leftover_dir.exists()

    # The real save then works as normal.
    store.save(document, chunks, [EAST])
    assert store.find_by_fingerprint("abc123") == document


def test_a_damaged_document_is_reported_not_searched(store):
    document, chunks = make_document("annual-report", ["east text", "north text"])
    store.save(document, chunks, [EAST, NORTH])
    # Replace the index with one holding a single vector.
    short_index = faiss.IndexFlatIP(3)
    short_index.add(graph.document_store.normalised_vectors([EAST]))
    faiss.write_index(short_index, str(store.vectors_dir / "annual-report" / INDEX_FILE_NAME))

    with pytest.raises(DocumentStoreError, match="damaged"):
        DocumentStore(store.vectors_dir).search(["annual-report"], EAST, k=1)
