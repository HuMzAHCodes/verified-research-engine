"""Checking a PDF and reading its pages: pdf-parser-mcp's reading.py.

CONCEPT: pdf-parser-mcp. Every refusal is tested with the reason it gives,
because the reason is what the user reads. The PDFs are written by
tests/pdf_fixtures.py into a temporary allowed folder.
"""

import pytest
from pypdf import PdfReader, PdfWriter

from mcp_servers.pdf_parser_mcp.reading import PdfLimits, PdfReadError, read_pdf_pages
from tests.pdf_fixtures import make_pdf, write_pdf

ONE_MEGABYTE = 1_048_576


@pytest.fixture
def uploads_folder(tmp_path):
    folder = tmp_path / "uploads"
    folder.mkdir()
    return folder


@pytest.fixture
def limits(uploads_folder):
    return PdfLimits(allowed_folder=uploads_folder, max_bytes=ONE_MEGABYTE)


def test_tc_doc_02_pages_come_back_in_order_with_an_empty_page_kept(uploads_folder, limits):
    pdf_path = write_pdf(uploads_folder, "report.pdf", ["alpha", "", "gamma (draft)"])

    page_texts = read_pdf_pages(str(pdf_path), limits)

    # Item 0 is page 1; the empty page keeps its place, so page 3 stays page 3.
    assert page_texts == ["alpha", "", "gamma (draft)"]


def test_a_relative_path_inside_the_folder_is_accepted(tmp_path, uploads_folder, monkeypatch):
    write_pdf(uploads_folder, "report.pdf", ["alpha"])
    monkeypatch.chdir(tmp_path)  # the server runs from the project root

    page_texts = read_pdf_pages("uploads/report.pdf", PdfLimits(uploads_folder, ONE_MEGABYTE))

    assert page_texts == ["alpha"]


@pytest.mark.parametrize("escape", ["../outside.pdf", "../../outside.pdf"])
def test_a_path_that_climbs_out_of_the_folder_is_refused(uploads_folder, limits, escape):
    write_pdf(uploads_folder.parent, "outside.pdf", ["secret"])

    with pytest.raises(PdfReadError, match="outside the folder this server may read"):
        read_pdf_pages(str(uploads_folder / escape), limits)


def test_an_absolute_path_elsewhere_is_refused(tmp_path, limits):
    elsewhere = write_pdf(tmp_path / "elsewhere", "report.pdf", ["secret"])

    with pytest.raises(PdfReadError, match="outside the folder"):
        read_pdf_pages(str(elsewhere), limits)


def test_a_missing_file_is_refused(uploads_folder, limits):
    with pytest.raises(PdfReadError, match="no file named missing.pdf"):
        read_pdf_pages(str(uploads_folder / "missing.pdf"), limits)


def test_a_file_that_is_not_a_pdf_is_refused(uploads_folder, limits):
    text_file = uploads_folder / "notes.pdf"
    text_file.write_text("Just some notes, saved with a .pdf name.")

    with pytest.raises(PdfReadError, match="not a PDF"):
        read_pdf_pages(str(text_file), limits)


def test_a_file_over_the_size_limit_is_refused_before_it_is_read(uploads_folder):
    pdf_path = write_pdf(uploads_folder, "large.pdf", ["alpha"])
    tiny_limits = PdfLimits(allowed_folder=uploads_folder, max_bytes=100)

    with pytest.raises(PdfReadError, match="larger than the"):
        read_pdf_pages(str(pdf_path), tiny_limits)


def test_a_password_protected_pdf_is_refused(uploads_folder, limits):
    pdf_writer = PdfWriter()
    pdf_writer.append(PdfReader(write_pdf(uploads_folder, "plain.pdf", ["alpha"])))
    pdf_writer.encrypt("a-password")
    locked_path = uploads_folder / "locked.pdf"
    with locked_path.open("wb") as locked_file:
        pdf_writer.write(locked_file)

    with pytest.raises(PdfReadError, match="password-protected"):
        read_pdf_pages(str(locked_path), limits)


def test_a_damaged_pdf_is_refused_with_its_name(uploads_folder, limits):
    damaged_path = uploads_folder / "damaged.pdf"
    damaged_path.write_bytes(make_pdf(["alpha"])[:60])  # starts with %PDF-, then cut off

    with pytest.raises(PdfReadError, match="could not read damaged.pdf as a PDF"):
        read_pdf_pages(str(damaged_path), limits)


def test_a_pdf_with_no_text_gives_only_empty_pages(uploads_folder, limits):
    pdf_path = write_pdf(uploads_folder, "scanned.pdf", ["", ""])

    assert read_pdf_pages(str(pdf_path), limits) == ["", ""]
