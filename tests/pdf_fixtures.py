"""Small text PDFs for tests, written byte by byte.

CONCEPT: pdf-parser-mcp, decision 6. A test says in one line what its PDF
holds, make_pdf(["alpha", "", "gamma"]): one line of text per page, an empty
string for a page with no text. No PDF files are kept in the repository and
no PDF-writing library is needed. pypdf reads these files back exactly
(checked by experiment before the concept was written).
"""

from io import BytesIO
from pathlib import Path


def escape_pdf_text(text: str) -> str:
    """Backslashes and brackets have a meaning inside a PDF string; escape them."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(page_texts: list[str]) -> bytes:
    """A valid PDF with one page per entry, each showing its text in Helvetica.

    The objects are numbered in the order they are added: the font, then a
    content stream and a page for each entry, then the page tree, then the
    catalog. The cross-reference table at the end records where each object
    starts, which is what a PDF reader uses to find them.
    """
    pdf_objects: list[bytes] = []

    def add_object(body: bytes) -> int:
        pdf_objects.append(body)
        return len(pdf_objects)

    font_number = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_tree_number = len(pdf_objects) + 2 * len(page_texts) + 1
    page_numbers: list[int] = []
    for page_text in page_texts:
        drawing = (
            f"BT /F1 12 Tf 72 720 Td ({escape_pdf_text(page_text)}) Tj ET".encode("latin-1")
            if page_text
            else b""
        )
        content_number = add_object(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(drawing), drawing)
        )
        page_numbers.append(
            add_object(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (page_tree_number, font_number, content_number)
            )
        )
    page_references = b" ".join(b"%d 0 R" % number for number in page_numbers)
    add_object(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (page_references, len(page_numbers)))
    catalog_number = add_object(b"<< /Type /Catalog /Pages %d 0 R >>" % page_tree_number)

    pdf_file = BytesIO()
    pdf_file.write(b"%PDF-1.4\n")
    object_offsets: list[int] = []
    for object_number, body in enumerate(pdf_objects, start=1):
        object_offsets.append(pdf_file.tell())
        pdf_file.write(b"%d 0 obj\n%s\nendobj\n" % (object_number, body))
    cross_reference_offset = pdf_file.tell()
    pdf_file.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(pdf_objects) + 1))
    for offset in object_offsets:
        pdf_file.write(b"%010d 00000 n \n" % offset)
    pdf_file.write(
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(pdf_objects) + 1, catalog_number, cross_reference_offset)
    )
    return pdf_file.getvalue()


def write_pdf(folder: Path, file_name: str, page_texts: list[str]) -> Path:
    """Write make_pdf(page_texts) to folder/file_name and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / file_name
    pdf_path.write_bytes(make_pdf(page_texts))
    return pdf_path
