import re
from dataclasses import dataclass
from pathlib import Path

from lokalcari.parsers.models import ParsedDocument


@dataclass
class Chunk:
    """Potongan teks dokumen hasil chunking."""

    text: str
    source: Path
    page: int | None
    index: int


def normalize_pdf_text(text: str) -> str:
    """Perapian teks PDF: gabungkan baris dalam paragraf, rapikan spasi berlebih, pertahankan baris kosong."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Pisahkan paragraf berdasarkan 2 atau lebih newline (termasuk baris kosong dengan spasi)
    paragraphs = re.split(r"\n[ \t]*\n+", text)
    cleaned_paras: list[str] = []
    for para in paragraphs:
        # Ganti satu newline di dalam paragraf dengan satu spasi
        p_clean = re.sub(r"[ \t]*\n[ \t]*", " ", para).strip()
        # Ubah deretan spasi/tab (2 atau lebih) di dalam baris menjadi satu spasi
        p_clean = re.sub(r"[ \t]{2,}", " ", p_clean)
        if p_clean:
            cleaned_paras.append(p_clean)
    return "\n\n".join(cleaned_paras)


def _find_word_boundary(
    text: str, ideal_pos: int, start: int, end: int
) -> int:
    """Mencari batas kata (awal kata) yang paling dekat dengan ideal_pos."""
    if ideal_pos <= start:
        return start + 1

    # Jika ideal_pos sudah berupa whitespace, cari awal kata berikutnya
    if text[ideal_pos].isspace():
        p = ideal_pos
        while p < end and text[p].isspace():
            p += 1
        return p if p < end else ideal_pos

    # Jika karakter sebelumnya whitespace, ideal_pos sudah merupakan awal kata
    if ideal_pos == 0 or text[ideal_pos - 1].isspace():
        return ideal_pos

    # ideal_pos berada di tengah kata: cari whitespace ke kiri
    ws_before = -1
    for i in range(ideal_pos - 1, start - 1, -1):
        if text[i].isspace():
            ws_before = i
            break
    cand_before = ws_before + 1 if ws_before != -1 and (ws_before + 1) > start else -1

    # Cari whitespace ke kanan
    ws_after = -1
    for i in range(ideal_pos + 1, end):
        if text[i].isspace():
            ws_after = i
            break
    cand_after = -1
    if ws_after != -1:
        p = ws_after
        while p < end and text[p].isspace():
            p += 1
        if p < end:
            cand_after = p

    if cand_before != -1 and cand_after != -1:
        if abs(cand_before - ideal_pos) <= abs(cand_after - ideal_pos):
            return cand_before
        return cand_after
    if cand_before != -1:
        return cand_before
    if cand_after != -1:
        return cand_after

    return ideal_pos


def _chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Memotong string teks menjadi potongan teks dengan batasan max_chars dan overlap."""
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    last_cut = 0
    min_new = int(0.4 * (max_chars - overlap))

    while start < text_len:
        # Lewati whitespace di awal
        while start < text_len and text[start].isspace():
            start += 1
        if start >= text_len:
            break

        end_limit = min(start + max_chars, text_len)
        if end_limit == text_len:
            candidate = text[start:end_limit].strip()
            if candidate:
                chunks.append(candidate)
            break

        candidate = text[start:end_limit]
        min_cut = max(start, last_cut)

        # Prioritas 1: Batas paragraf (baris kosong)
        para_matches = list(re.finditer(r"\n[ \t]*\n+", candidate))
        cut = -1
        for m in reversed(para_matches):
            c = start + m.start()
            if c - min_cut >= min_new:
                cut = c
                break

        # Prioritas 2: Batas kalimat (. ! ? diikuti spasi atau newline)
        if cut == -1:
            sent_matches = list(re.finditer(r"[.!?](?=[ \t\n\r]|$)", candidate))
            for m in reversed(sent_matches):
                c = start + m.end()
                if c - min_cut >= min_new:
                    cut = c
                    break

        # Prioritas 3: Batas kata (spasi)
        if cut == -1:
            space_matches = list(re.finditer(r"\s+", candidate))
            for m in reversed(space_matches):
                c = start + m.start()
                if c - min_cut >= min_new:
                    cut = c
                    break

        # Prioritas 4: Potong keras pada max_chars
        if cut == -1 or cut <= start:
            cut = end_limit

        chunk_content = text[start:cut].strip()
        if chunk_content:
            chunks.append(chunk_content)

        chunk_end = cut
        while chunk_end > start and text[chunk_end - 1].isspace():
            chunk_end -= 1

        last_cut = cut
        if chunk_end >= text_len:
            break

        # Hitung posisi start berikutnya dengan overlap
        if overlap > 0:
            ideal = chunk_end - overlap
            if ideal > start:
                next_start = _find_word_boundary(text, ideal, start, chunk_end)
            else:
                next_start = cut
        else:
            next_start = cut

        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_document(
    doc: ParsedDocument,
    max_chars: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Memotong ParsedDocument menjadi list of Chunk."""
    if max_chars <= 0:
        raise ValueError(f"max_chars harus > 0, didapat {max_chars}")
    if overlap < 0:
        raise ValueError(f"overlap harus >= 0, didapat {overlap}")
    if overlap >= max_chars:
        raise ValueError(
            f"overlap ({overlap}) harus lebih kecil dari max_chars ({max_chars})"
        )

    all_chunks: list[Chunk] = []
    chunk_index = 0

    for segment in doc.segments:
        text = segment.text
        if segment.page is not None:
            text = normalize_pdf_text(text)

        text_chunks = _chunk_text(text, max_chars=max_chars, overlap=overlap)
        for chunk_text in text_chunks:
            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    source=doc.path,
                    page=segment.page,
                    index=chunk_index,
                )
            )
            chunk_index += 1

    return all_chunks
