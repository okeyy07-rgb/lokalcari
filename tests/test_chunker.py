import re
from pathlib import Path

import pytest

from lokalcari.chunker import chunk_document, normalize_pdf_text
from lokalcari.parsers.models import ParsedDocument, Segment


def test_invalid_parameters():
    doc = ParsedDocument(path=Path("dummy.txt"), segments=[Segment(text="halo")])

    with pytest.raises(ValueError, match="max_chars"):
        chunk_document(doc, max_chars=0, overlap=0)

    with pytest.raises(ValueError, match="max_chars"):
        chunk_document(doc, max_chars=-10, overlap=0)

    with pytest.raises(ValueError, match="overlap"):
        chunk_document(doc, max_chars=100, overlap=-5)

    with pytest.raises(ValueError, match="overlap"):
        chunk_document(doc, max_chars=100, overlap=100)

    with pytest.raises(ValueError, match="overlap"):
        chunk_document(doc, max_chars=100, overlap=150)


def test_empty_document():
    doc = ParsedDocument(path=Path("empty.txt"), segments=[])
    chunks = chunk_document(doc)
    assert chunks == []


def test_short_text_single_chunk():
    text = "Ini adalah teks pendek yang kurang dari batas maksimum karakter."
    doc = ParsedDocument(
        path=Path("pendek.txt"), segments=[Segment(text=text, page=None)]
    )

    chunks = chunk_document(doc, max_chars=1000, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].page is None
    assert chunks[0].index == 0
    assert chunks[0].source == Path("pendek.txt")


def test_very_long_text():
    # Buat teks panjang ~250.000 karakter
    base_sentence = (
        "LokalCari mengindeks dokumen lokal secara aman tanpa koneksi internet. "
    )
    long_text = base_sentence * 3600
    assert len(long_text) > 250000

    doc = ParsedDocument(
        path=Path("panjang.txt"), segments=[Segment(text=long_text, page=None)]
    )
    chunks = chunk_document(doc, max_chars=1000, overlap=150)

    assert len(chunks) > 250
    for i, c in enumerate(chunks):
        assert len(c.text) <= 1000
        assert c.index == i
        assert c.page is None
        assert c.source == Path("panjang.txt")


def test_no_words_lost_indonesian():
    indo_text = (
        "LokalCari adalah aplikasi pencari dokumen pribadi dengan bahasa natural yang berjalan 100% lokal. "
        "Pengguna memilih folder, aplikasi mengindeks file PDF, DOCX, TXT, dan MD. "
        "Kemudian pengguna bisa bertanya dalam bahasa Indonesia atau Inggris untuk mencari dokumen. "
        "Sistem akan memberikan potongan teks yang sangat relevan sesuai konteks pencarian. "
        "Privasi pengguna sangat terlindungi karena seluruh proses dilakukan secara offline tanpa internet. "
        "Setiap berkas disimpan dalam indeks lokal dengan pemotongan teks yang rapi dan konsisten. "
        "Kecepatan pencarian tetap optimal meskipun jumlah dokumen bertambah banyak di dalam repositori. "
        "Pengembang lokal dapat berkontribusi untuk memperkaya fitur dan format file yang didukung."
    )
    doc = ParsedDocument(
        path=Path("indo.txt"), segments=[Segment(text=indo_text, page=None)]
    )
    # Gunakan max_chars kecil untuk memicu pemotongan berulang
    chunks = chunk_document(doc, max_chars=130, overlap=35)

    original_words = re.findall(r"\b\w+\b", indo_text)
    combined_chunks_text = " ".join(c.text for c in chunks)

    for word in original_words:
        assert word in combined_chunks_text


def test_overlap_contains_clean_words():
    indo_text = (
        "LokalCari adalah aplikasi pencari dokumen pribadi dengan bahasa natural yang berjalan secara offline. "
        "Pengguna memilih direktori lokal yang akan diindeks oleh sistem secara otomatis dan aman. "
        "Format dokumen yang didukung saat ini mencakup PDF, DOCX, TXT, dan Markdown secara berkala. "
        "Setiap dokumen diproses menjadi potongan teks yang mempertahankan struktur semantik dan nomor halaman asli."
    )
    doc = ParsedDocument(
        path=Path("overlap.txt"), segments=[Segment(text=indo_text, page=None)]
    )
    chunks = chunk_document(doc, max_chars=120, overlap=30)
    assert len(chunks) > 1

    for i in range(len(chunks) - 1):
        c1 = chunks[i].text
        c2 = chunks[i + 1].text

        # Kata pertama chunk n+1 harus ada di dalam chunk n (overlap tidak memotong kata)
        first_word_c2 = c2.split()[0]
        assert first_word_c2 in c1


def test_pdf_multipage_preserves_page_numbers():
    seg1 = Segment(
        text="Teks halaman satu yang cukup panjang dan informatif untuk diuji.", page=1
    )
    seg2 = Segment(
        text="Teks halaman dua dengan isi berbeda mengenai arsitektur sistem.", page=2
    )
    doc = ParsedDocument(path=Path("multi.pdf"), segments=[seg1, seg2])

    chunks = chunk_document(doc, max_chars=50, overlap=10)
    assert len(chunks) >= 2

    pages = [c.page for c in chunks]
    assert 1 in pages
    assert 2 in pages

    # Pastikan nomor index berurutan dari 0
    for idx, c in enumerate(chunks):
        assert c.index == idx

    # Pastikan tidak ada chunk yang mencampurkan konten halaman 1 dan 2
    for c in chunks:
        if c.page == 1:
            assert "halaman dua" not in c.text
        elif c.page == 2:
            assert "halaman satu" not in c.text


def test_pdf_newline_joined_vs_markdown_preserved():
    pdf_text = "Baris pertama kalimat\ndan ini sambungan kalimat.\n\nParagraf baru."
    doc_pdf = ParsedDocument(
        path=Path("doc.pdf"), segments=[Segment(text=pdf_text, page=1)]
    )
    chunks_pdf = chunk_document(doc_pdf, max_chars=1000, overlap=150)
    assert len(chunks_pdf) == 1
    assert "Baris pertama kalimat dan ini sambungan kalimat." in chunks_pdf[0].text
    assert "kalimat\ndan" not in chunks_pdf[0].text

    md_text = (
        "# Judul\n\n- Item daftar 1\n- Item daftar 2\n\n```python\nprint(1)\n```"
    )
    doc_md = ParsedDocument(
        path=Path("doc.md"), segments=[Segment(text=md_text, page=None)]
    )
    chunks_md = chunk_document(doc_md, max_chars=1000, overlap=150)
    assert len(chunks_md) == 1
    # Struktur baris markdown tetap utuh
    assert "- Item daftar 1\n- Item daftar 2" in chunks_md[0].text
    assert "```python\nprint(1)\n```" in chunks_md[0].text


def test_long_string_without_spaces_hard_cut():
    long_unbroken = "X" * 5000
    doc = ParsedDocument(
        path=Path("unbroken.txt"),
        segments=[Segment(text=long_unbroken, page=None)],
    )

    chunks = chunk_document(doc, max_chars=1000, overlap=150)
    assert len(chunks) == 6
    for i, c in enumerate(chunks):
        assert len(c.text) <= 1000
        assert c.index == i


def test_utf8_indonesian_characters():
    text_utf8 = (
        "Dokumen dengan tanda petik ganda ‘cantik’ dan “cerdas”, "
        "simbol mata uang Rp100.000, dan huruf beraksen café & naïve."
    )
    doc = ParsedDocument(
        path=Path("utf8.txt"), segments=[Segment(text=text_utf8, page=None)]
    )
    chunks = chunk_document(doc, max_chars=1000, overlap=150)
    assert len(chunks) == 1
    assert "‘cantik’" in chunks[0].text
    assert "“cerdas”" in chunks[0].text
    assert "café & naïve" in chunks[0].text


def test_normalize_pdf_text_helper():
    sample = (
        "Baris 1\nbaris 2\nbaris 3.\n\n"
        "Paragraf 2 baris a\nbaris b.\n\n"
        "Kata dengan tanda hubung: semi-\nkonduktor tetap utuh."
    )
    normalized = normalize_pdf_text(sample)
    assert "Baris 1 baris 2 baris 3." in normalized
    assert "Paragraf 2 baris a baris b." in normalized
    # Tanda hubung tidak digabungkan
    assert "semi- konduktor" in normalized
