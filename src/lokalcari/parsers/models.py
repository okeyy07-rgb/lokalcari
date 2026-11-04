from dataclasses import dataclass
from pathlib import Path


class UnsupportedFileType(Exception):
    """Exception ketika ekstensi file tidak didukung."""

    def __init__(self, path: Path | str, message: str | None = None) -> None:
        self.path = Path(path)
        msg = message or f"Ekstensi file '{self.path.name}' tidak didukung"
        super().__init__(msg)


class ParseError(Exception):
    """Exception ketika terjadi kegagalan parsing file."""

    def __init__(self, path: Path | str, message: str | None = None) -> None:
        self.path = Path(path)
        msg = message or f"Gagal memproses file '{self.path.name}'"
        super().__init__(msg)


@dataclass
class Segment:
    """Potongan teks dokumen dengan informasi halaman jika tersedia."""

    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    """Hasil parsing dokumen dengan path dan daftar segment."""

    path: Path
    segments: list[Segment]
