"""The page store: pages by document, counted from 1, oldest forgotten first.

CONCEPT: pdf-parser-mcp, decisions 3 and 4.
"""

import pytest

from mcp_servers.pdf_parser_mcp.page_store import (
    DocumentNotHeld,
    DocumentPageStore,
    PageOutOfRange,
)


def test_pages_are_counted_from_one():
    page_store = DocumentPageStore(max_documents=3)
    page_store.add("doc-1", ["first", "", "third"])

    assert [page_store.page_text("doc-1", number) for number in (1, 2, 3)] == [
        "first",
        "",
        "third",
    ]


@pytest.mark.parametrize("page_number", [0, 4, -1])
def test_a_page_number_outside_the_document_says_which_pages_exist(page_number):
    page_store = DocumentPageStore(max_documents=3)
    page_store.add("doc-1", ["first", "second", "third"])

    with pytest.raises(PageOutOfRange, match="has pages 1 to 3"):
        page_store.page_text("doc-1", page_number)


def test_a_document_never_ingested_says_to_ingest_it():
    with pytest.raises(DocumentNotHeld, match="call ingest_pdf first"):
        DocumentPageStore(max_documents=3).page_text("doc-9", 1)


def test_the_oldest_document_is_forgotten_to_make_room():
    page_store = DocumentPageStore(max_documents=2)
    page_store.add("doc-1", ["one"])
    page_store.add("doc-2", ["two"])

    forgotten_ids = page_store.add("doc-3", ["three"])

    assert forgotten_ids == ["doc-1"]
    assert page_store.held_document_ids == ["doc-2", "doc-3"]
    with pytest.raises(DocumentNotHeld):
        page_store.page_text("doc-1", 1)


def test_ingesting_a_document_again_replaces_its_pages_and_makes_it_newest():
    page_store = DocumentPageStore(max_documents=2)
    page_store.add("doc-1", ["old text"])
    page_store.add("doc-2", ["two"])

    page_store.add("doc-1", ["new text"])
    page_store.add("doc-3", ["three"])  # doc-2 is now the oldest

    assert page_store.page_text("doc-1", 1) == "new text"
    assert page_store.held_document_ids == ["doc-1", "doc-3"]
