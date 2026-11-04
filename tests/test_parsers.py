import io
from pathlib import Path

import docx
import pypdf
import pytest

from lokalcari.parsers import (
    ParsedDocument,
    ParseError,
    Segment,
    UnsupportedFileType,
    parse,
    parse_file,
    supported_extensions,
)
from lokalcari.parsers.cleaner import clean_text


def create_minimal_pdf(pages_text: list[str]) -> bytes:
    """Helper untuk membuat PDF minimal dengan byte secara manual."""
    num_pages = len(pages_text)
    page_ids = [4 + 2 * i for i in range(num_pages)]

    catalog_obj = "<< /Type /Catalog /Pages 2 0 R >>"
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_obj = f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>"
    font_obj = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    objs: list[tuple[int, str]] = [
        (1, catalog_obj),
        (2, pages_obj),
        (3, font_obj),
    ]

    for i, text in enumerate(pages_text):
        pid = 4 + 2 * i
        cid = 5 + 2 * i
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if escaped:
            stream_content = f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET\n"
        else:
            stream_content = ""
        stream_bytes = stream_content.encode("latin1", errors="replace")
        content_obj = f"<< /Length {len(stream_bytes)} >>\nstream\n{stream_content}endstream"
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>"
        )
        objs.append((pid, page_obj))
        objs.append((cid, content_obj))

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {0: 0}
    for oid, ocontent in objs:
        offsets[oid] = out.tell()
        out.write(f"{oid} 0 obj\n{ocontent}\nendobj\n".encode("latin1"))

    startxref = out.tell()
    total_objs = len(objs) + 1
    out.write(f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin1"))
    for oid in range(1, total_objs):
        off = offsets[oid]
        out.write(f"{off:010d} 00000 n \n".encode("latin1"))
    out.write(
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode(
            "latin1"
        )
    )
    return out.getvalue()


def test_models_and_exceptions():
    p = Path("dokumen.pdf")
    err_unsupported = UnsupportedFileType(p)
    assert "dokumen.pdf" in str(err_unsupported)

    err_parse = ParseError(p, "Gagal membaca")
    assert "Gagal membaca" in str(err_parse)

    seg = Segment(text="teks sampel", page=1)
    doc = ParsedDocument(path=p, segments=[seg])
    assert doc.path == p
    assert len(doc.segments) == 1
    assert doc.segments[0].page == 1


def test_supported_extensions():
    exts = supported_extensions()
    assert {".txt", ".md", ".pdf", ".docx"}.issubset(exts)


def test_clean_text():
    raw = "Halo\x00 Dunia!\r\n\r\n\r\n\r\nParagraf baru.\n\n\nBaris ketiga."
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Halo Dunia!\n\nParagraf baru.\n\nBaris ketiga." == cleaned


def test_parse_txt_utf8_indonesian(tmp_path: Path):
    txt_file = tmp_path / "catatan_indonesia.txt"
    content = (
        "LokalCari mengindeks dokumen berbahasa Indonesia.\n"
        "Fitur pencarian menggunakan bahasa alami dengan karakter UTF-8: ‘kutipan’ & café."
    )
    txt_file.write_text(content, encoding="utf-8")

    doc = parse_file(txt_file)
    assert isinstance(doc, ParsedDocument)
    assert len(doc.segments) == 1
    assert doc.segments[0].page is None
    assert "berbahasa Indonesia" in doc.segments[0].text
    assert "‘kutipan’" in doc.segments[0].text


def test_parse_txt_utf8_with_bom(tmp_path: Path):
    txt_file = tmp_path / "with_bom.txt"
    txt_file.write_bytes(b"\xef\xbb\xbfKonten dengan UTF-8 BOM.")

    doc = parse_file(txt_file)
    assert len(doc.segments) == 1
    assert doc.segments[0].page is None
    assert doc.segments[0].text == "Konten dengan UTF-8 BOM."


def test_parse_txt_cp1252_fallback(tmp_path: Path):
    txt_file = tmp_path / "legacy_cp1252.txt"
    # 0x93 dan 0x94 adalah kutipan cerdas cp1252 yang invalid di UTF-8 murni
    raw_cp1252 = b"Dokumen lama dengan \x93kutipan cp1252\x94 dan simbol \x80 euro."
    txt_file.write_bytes(raw_cp1252)

    doc = parse_file(txt_file)
    assert len(doc.segments) == 1
    assert "kutipan cp1252" in doc.segments[0].text


def test_parse_md(tmp_path: Path):
    md_file = tmp_path / "README.md"
    md_file.write_text("# Judul Dokumen\n\nIni isi dokumen markdown.", encoding="utf-8")

    doc = parse_file(md_file)
    assert len(doc.segments) == 1
    assert doc.segments[0].page is None
    assert "# Judul Dokumen" in doc.segments[0].text


def test_parse_pdf_multipage_and_page_numbers(tmp_path: Path):
    pdf_bytes = create_minimal_pdf(
        ["Halaman satu bahasa Indonesia", "Halaman kedua berisi data penting"]
    )
    pdf_file = tmp_path / "multi_halaman.pdf"
    pdf_file.write_bytes(pdf_bytes)

    doc = parse_file(pdf_file)
    assert len(doc.segments) == 2
    assert doc.segments[0].page == 1
    assert "Halaman satu bahasa Indonesia" in doc.segments[0].text
    assert doc.segments[1].page == 2
    assert "Halaman kedua berisi data penting" in doc.segments[1].text


def test_parse_pdf_skip_empty_page(tmp_path: Path):
    # Halaman 2 kosong (misal hasil scan tanpa teks)
    pdf_bytes = create_minimal_pdf(["Teks halaman satu", "", "Teks halaman tiga"])
    pdf_file = tmp_path / "skip_halaman.pdf"
    pdf_file.write_bytes(pdf_bytes)

    doc = parse_file(pdf_file)
    assert len(doc.segments) == 2
    assert doc.segments[0].page == 1
    assert doc.segments[0].text == "Teks halaman satu"
    assert doc.segments[1].page == 3
    assert doc.segments[1].text == "Teks halaman tiga"


def test_parse_pdf_no_text_returns_zero_segments(tmp_path: Path):
    pdf_bytes = create_minimal_pdf(["", ""])
    pdf_file = tmp_path / "scan_tanpa_teks.pdf"
    pdf_file.write_bytes(pdf_bytes)

    doc = parse_file(pdf_file)
    assert len(doc.segments) == 0


def test_parse_docx_with_table_in_order(tmp_path: Path):
    docx_file = tmp_path / "dokumen_tabel.docx"
    doc_obj = docx.Document()
    doc_obj.add_paragraph("Paragraf pertama pembuka dokumen.")

    tbl = doc_obj.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "Kolom A"
    tbl.cell(0, 1).text = "Kolom B"
    tbl.cell(1, 0).text = "Nilai 1"
    tbl.cell(1, 1).text = "Nilai 2"

    doc_obj.add_paragraph("Paragraf penutup setelah tabel.")
    doc_obj.save(str(docx_file))

    doc = parse_file(docx_file)
    assert len(doc.segments) == 1
    assert doc.segments[0].page is None

    text = doc.segments[0].text
    pos_p1 = text.find("Paragraf pertama pembuka dokumen.")
    pos_tbl = text.find("Kolom A | Kolom B")
    pos_p2 = text.find("Paragraf penutup setelah tabel.")

    assert pos_p1 != -1
    assert pos_tbl != -1
    assert pos_p2 != -1
    assert pos_p1 < pos_tbl < pos_p2


def test_case_insensitive_extension(tmp_path: Path):
    txt_file = tmp_path / "TEST.TXT"
    txt_file.write_text("Teks ekstensi huruf besar", encoding="utf-8")

    doc = parse_file(txt_file)
    assert len(doc.segments) == 1
    assert "Teks ekstensi huruf besar" in doc.segments[0].text


def test_empty_files_raise_parse_error(tmp_path: Path):
    for ext in [".txt", ".md", ".pdf", ".docx"]:
        empty_file = tmp_path / f"empty{ext}"
        empty_file.write_bytes(b"")
        with pytest.raises(ParseError) as exc_info:
            parse_file(empty_file)
        assert empty_file.name in str(exc_info.value)


def test_corrupt_files_raise_parse_error(tmp_path: Path):
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"Bukan file pdf valid sama sekali 12345")
    with pytest.raises(ParseError) as exc_info:
        parse_file(corrupt_pdf)
    assert corrupt_pdf.name in str(exc_info.value)

    corrupt_docx = tmp_path / "corrupt.docx"
    corrupt_docx.write_bytes(b"Bukan format zip docx yang valid")
    with pytest.raises(ParseError) as exc_info:
        parse_file(corrupt_docx)
    assert corrupt_docx.name in str(exc_info.value)


def test_unsupported_extension_raises_unsupported_file_type(tmp_path: Path):
    unsupported = tmp_path / "data.csv"
    unsupported.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(UnsupportedFileType) as exc_info:
        parse_file(unsupported)
    assert unsupported.name in str(exc_info.value)
    assert exc_info.type is UnsupportedFileType


def test_parse_encrypted_pdf_raises_parse_error(tmp_path: Path):
    pdf_bytes = create_minimal_pdf(["Dokumen rahasia terenkripsi"])
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    writer = pypdf.PdfWriter()
    writer.append(reader)
    writer.encrypt(user_password="password123", algorithm="RC4-128")

    enc_pdf = tmp_path / "encrypted.pdf"
    with open(enc_pdf, "wb") as f:
        writer.write(f)

    with pytest.raises(ParseError) as exc_info:
        parse_file(enc_pdf)
    assert enc_pdf.name in str(exc_info.value)
    assert "terenkripsi" in str(exc_info.value) or "password" in str(exc_info.value)


def test_parse_file_wraps_unexpected_exception_as_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    txt_file = tmp_path / "error.txt"
    txt_file.write_text("konten", encoding="utf-8")

    def mock_parser(path: Path):
        raise RuntimeError("Kesalahan tidak terduga dalam parser!")

    monkeypatch.setitem(parse._PARSERS, ".txt", mock_parser)

    with pytest.raises(ParseError) as exc_info:
        parse_file(txt_file)

    assert txt_file.name in str(exc_info.value)
    assert "Kesalahan tidak terduga dalam parser!" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
