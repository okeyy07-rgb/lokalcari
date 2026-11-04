from collections.abc import Callable
from pathlib import Path

from lokalcari.parsers.docx import parse_docx
from lokalcari.parsers.models import (
    ParsedDocument,
    ParseError,
    Segment,
    UnsupportedFileType,
)
from lokalcari.parsers.pdf import parse_pdf
from lokalcari.parsers.txt import parse_txt

ParserFunc = Callable[[Path], list[Segment]]

_PARSERS: dict[str, ParserFunc] = {
    ".txt": parse_txt,
    ".md": parse_txt,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}


def supported_extensions() -> set[str]:
    """Mengembalikan kumpulan ekstensi file yang didukung."""
    return set(_PARSERS.keys())


def parse_file(path: Path | str) -> ParsedDocument:
    """Memilih parser berdasarkan ekstensi file dan mengembalikan ParsedDocument."""
    file_path = Path(path)

    ext = file_path.suffix.lower()
    if ext not in _PARSERS:
        raise UnsupportedFileType(
            file_path, f"Ekstensi file '{file_path.name}' tidak didukung"
        )

    if not file_path.exists():
        raise ParseError(file_path, f"File tidak ditemukan: '{file_path}'")

    if not file_path.is_file():
        raise ParseError(file_path, f"Path bukan file reguler: '{file_path}'")

    parser = _PARSERS[ext]
    segments = parser(file_path)
    return ParsedDocument(path=file_path, segments=segments)
