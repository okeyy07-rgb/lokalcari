from pathlib import Path

import pypdf

from lokalcari.parsers.cleaner import clean_text
from lokalcari.parsers.models import ParseError, Segment


def parse_pdf(path: Path) -> list[Segment]:
    """Parse file PDF menggunakan pypdf: satu segment per halaman."""
    try:
        if path.stat().st_size == 0:
            raise ParseError(path, f"File PDF kosong (0 byte): '{path.name}'")
    except OSError as e:
        raise ParseError(path, f"Gagal mengakses file '{path.name}': {e}") from e

    try:
        reader = pypdf.PdfReader(str(path))
        pages = reader.pages
    except Exception as e:
        raise ParseError(path, f"File PDF korup atau tidak valid '{path.name}': {e}") from e

    segments: list[Segment] = []
    for page_idx, page in enumerate(pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except (pypdf.errors.PyPdfError, ValueError, TypeError, KeyError):
            # Halaman tanpa teks atau gagal diekstrak dilewati tanpa error
            raw_text = ""

        cleaned = clean_text(raw_text)
        if cleaned:
            segments.append(Segment(text=cleaned, page=page_idx))

    return segments
