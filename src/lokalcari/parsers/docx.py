from pathlib import Path

import docx
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from lokalcari.parsers.cleaner import clean_text
from lokalcari.parsers.models import ParseError, Segment


def parse_docx(path: Path) -> list[Segment]:
    """Parse file DOCX: ambil teks paragraf dan isi tabel sesuai urutan dokumen, satu segment."""
    try:
        if path.stat().st_size == 0:
            raise ParseError(path, f"File DOCX kosong (0 byte): '{path.name}'")
    except OSError as e:
        raise ParseError(path, f"Gagal mengakses file '{path.name}': {e}") from e

    try:
        doc = docx.Document(str(path))
    except Exception as e:
        raise ParseError(path, f"File DOCX korup atau tidak valid '{path.name}': {e}") from e

    parts: list[str] = []

    for child in doc.element.body:
        if isinstance(child, CT_P):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if text:
                parts.append(text)
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            for row in table.rows:
                seen_cells = set()
                row_cells: list[str] = []
                for cell in row.cells:
                    if cell._tc not in seen_cells:
                        seen_cells.add(cell._tc)
                        ct = cell.text.strip()
                        if ct:
                            row_cells.append(ct)
                if row_cells:
                    parts.append(" | ".join(row_cells))

    full_text = "\n\n".join(parts)
    cleaned = clean_text(full_text)
    if not cleaned:
        raise ParseError(path, f"File DOCX tidak memiliki teks: '{path.name}'")

    return [Segment(text=cleaned, page=None)]
