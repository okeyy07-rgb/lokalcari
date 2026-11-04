from pathlib import Path

from lokalcari.parsers.cleaner import clean_text
from lokalcari.parsers.models import ParseError, Segment


def parse_txt(path: Path) -> list[Segment]:
    """Parse file TXT/MD ke segment tunggal."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ParseError(path, f"Gagal membaca file '{path.name}': {e}") from e

    if not raw:
        raise ParseError(path, f"File kosong: '{path.name}'")

    try:
        # utf-8-sig otomatis menangani UTF-8 BOM jika ada, maupun plain UTF-8
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            content = raw.decode("cp1252")
        except UnicodeDecodeError as e:
            raise ParseError(
                path, f"Gagal decode file '{path.name}' dengan UTF-8 maupun cp1252"
            ) from e

    cleaned = clean_text(content)
    if not cleaned:
        raise ParseError(path, f"File tidak memiliki konten teks: '{path.name}'")

    return [Segment(text=cleaned, page=None)]
