"""TC-DOC-03, the chunk rules: hand-written examples and property tests.

CONCEPT: document-ingestion, decisions 1 and 2. chunk_page() promises rules
that must hold for every page, not only the pages someone thought of. The
property tests use Hypothesis, which generates hundreds of pages per test
(ordinary prose, and also text with no spaces, odd whitespace, very long
words), checks every rule on each, and, when one fails, shrinks the page to
the smallest one that still breaks it.

Small chunk sizes (60 characters, 15 overlapping) are used in the generated
tests so that short pages already cross many chunk boundaries; the rules do
not depend on the numbers. The examples use the real 600 and 150.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ingestion import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    PageSlice,
    build_chunks,
    chunk_page,
)

SMALL_MAX_CHARS = 60
SMALL_OVERLAP_CHARS = 15
LONGEST_PROSE_WORD = 12

# Ordinary prose: words of up to 12 letters, between single or double spaces
# and line breaks, as PDF text comes out.
prose_words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz.,", min_size=1, max_size=12)
prose_separators = st.sampled_from([" ", " ", " ", "  ", "\n", "\n\n", " \n"])
prose_pages = st.lists(st.tuples(prose_words, prose_separators), min_size=1, max_size=120).map(
    lambda pairs: "".join(word + separator for word, separator in pairs)
)

# Anything at all: every kind of character and whitespace (including a
# no-break space and an em space), pages that are only spaces.
unicode_pages = st.text(
    alphabet=st.one_of(
        st.characters(codec="utf-8"),
        st.sampled_from([" ", "\n", "\t", " ", " "]),
    ),
    max_size=400,
)
# Words longer than a chunk (a URL, a table row with no spaces), which random
# text above rarely produces. Added after the random text alone missed a
# planted bug that lost the end of such a word (document-ingestion notes).
over_long_words = st.text(alphabet="xyz0123", min_size=SMALL_MAX_CHARS + 1, max_size=150)
long_word_pages = st.lists(
    st.tuples(st.one_of(prose_words, over_long_words), prose_separators),
    min_size=1,
    max_size=8,
).map(lambda pairs: "".join(word + separator for word, separator in pairs))
any_pages = st.one_of(unicode_pages, long_word_pages)


def chunk_texts(page_text: str, page_slices: list[PageSlice]) -> list[str]:
    return [page_text[page_slice.start : page_slice.end] for page_slice in page_slices]


def check_rules_that_hold_for_every_page(page_text: str, page_slices: list[PageSlice]) -> None:
    """The rules no page may break, whatever its text."""
    for page_slice in page_slices:
        chunk_text = page_text[page_slice.start : page_slice.end]
        assert 0 < len(chunk_text) <= SMALL_MAX_CHARS
        assert chunk_text == chunk_text.strip()  # no whitespace at either edge

    # In order, each chunk starting after the one before.
    starts = [page_slice.start for page_slice in page_slices]
    assert starts == sorted(set(starts))

    # No text lost: every non-space character is inside some chunk.
    covered_positions = set()
    for page_slice in page_slices:
        covered_positions.update(range(page_slice.start, page_slice.end))
    for position, character in enumerate(page_text):
        if not character.isspace():
            assert position in covered_positions, f"character {position} is in no chunk"


@settings(max_examples=300)
@given(page_text=any_pages)
def test_every_page_gives_exact_bounded_chunks_that_lose_no_text(page_text):
    page_slices = chunk_page(page_text, SMALL_MAX_CHARS, SMALL_OVERLAP_CHARS)

    check_rules_that_hold_for_every_page(page_text, page_slices)
    if not page_text.strip():
        assert page_slices == []


@settings(max_examples=300)
@given(page_text=prose_pages)
def test_prose_chunks_end_between_words_and_overlap_by_at_least_the_overlap(page_text):
    page_slices = chunk_page(page_text, SMALL_MAX_CHARS, SMALL_OVERLAP_CHARS)

    check_rules_that_hold_for_every_page(page_text, page_slices)
    for page_slice in page_slices:
        # Starts at the start of a word and ends at the end of one: no word is cut.
        assert page_slice.start == 0 or page_text[page_slice.start - 1].isspace()
        assert page_slice.end == len(page_text) or page_text[page_slice.end].isspace()
    for earlier_slice, later_slice in zip(page_slices, page_slices[1:], strict=False):
        overlap = earlier_slice.end - later_slice.start
        # At least the overlap, and at most one word and its spaces more.
        assert SMALL_OVERLAP_CHARS <= overlap <= SMALL_OVERLAP_CHARS + LONGEST_PROSE_WORD + 2


def test_a_short_page_is_one_chunk_without_its_edge_spaces():
    page_text = "  \nRevenue grew 30% in fiscal 2025.\n\n"

    assert chunk_texts(page_text, chunk_page(page_text)) == ["Revenue grew 30% in fiscal 2025."]


def test_an_empty_or_blank_page_gives_no_chunks():
    assert chunk_page("") == []
    assert chunk_page(" \n\t \n") == []


def test_a_real_sized_page_is_cut_at_600_with_150_overlapping():
    # About one page of the annual report: 4,000 characters of prose.
    page_text = " ".join(f"word{index % 97:02d}" for index in range(700))

    page_slices = chunk_page(page_text)

    assert all(page_slice.end - page_slice.start <= CHUNK_MAX_CHARS for page_slice in page_slices)
    overlaps = [
        earlier.end - later.start
        for earlier, later in zip(page_slices, page_slices[1:], strict=False)
    ]
    assert all(overlap >= CHUNK_OVERLAP_CHARS for overlap in overlaps)
    # 4,899 characters: the first chunk takes 594, each later one 441 new
    # ones (its 594 less the 153 it shares with the one before), so 11 chunks.
    assert page_slices[0] == PageSlice(0, 594)
    assert len(page_slices) == 11


def test_a_word_longer_than_a_chunk_is_cut_and_continued():
    long_word = "x" * 130
    page_text = f"start {long_word} end"

    page_texts = chunk_texts(page_text, chunk_page(page_text, 60, 15))

    # Cut at the limit, with nothing of the word lost between chunks.
    assert all(len(chunk_text) <= 60 for chunk_text in page_texts)
    assert page_texts[-1].endswith(" end")
    assert sum(chunk_text.count("x") for chunk_text in page_texts) >= len(long_word)


def test_chunks_are_numbered_across_pages_and_never_cross_one():
    page_texts = ["First page text.", "", "Third page text."]

    stored_chunks = build_chunks("doc-1", page_texts)

    assert [(chunk.chunk_id, chunk.page, chunk.text) for chunk in stored_chunks] == [
        ("doc-1:0", 1, "First page text."),
        ("doc-1:1", 3, "Third page text."),
    ]


@pytest.mark.parametrize("page_number", [1, 2])
def test_every_chunk_is_an_exact_slice_of_its_own_page(page_number):
    page_texts = ["Alpha  beta\ngamma. " * 60, "Delta\n\nepsilon zeta. " * 60]

    stored_chunks = build_chunks("doc-1", page_texts)

    for stored_chunk in stored_chunks:
        if stored_chunk.page == page_number:
            assert stored_chunk.text in page_texts[page_number - 1]
