from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lokalcari.chunker import Chunk
from lokalcari.store import (
    DocumentRecord,
    Store,
    StoreError,
    StoreModelMismatch,
    default_db_path,
)


def _create_random_vectors(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Helper membuat n vektor float32 acak ternormalisasi."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(norms, 1e-12)).astype(np.float32)


def _create_mock_chunks(
    doc_path: Path, texts: list[str], start_index: int = 0
) -> list[Chunk]:
    """Helper membuat daftar Chunk dari list teks."""
    return [
        Chunk(text=t, source=doc_path, page=i + 1, index=start_index + i)
        for i, t in enumerate(texts)
    ]


def test_default_db_path(monkeypatch, tmp_path):
    """Pengujian default_db_path dengan env var dan fallback."""
    monkeypatch.delenv("LOKALCARI_DATA_DIR", raising=False)
    p_default = default_db_path()
    assert p_default == Path.home() / ".lokalcari" / "index.db"

    custom_dir = tmp_path / "custom_data"
    monkeypatch.setenv("LOKALCARI_DATA_DIR", str(custom_dir))
    assert default_db_path() == custom_dir / "index.db"

    custom_file = tmp_path / "custom.db"
    monkeypatch.setenv("LOKALCARI_DATA_DIR", str(custom_file))
    assert default_db_path() == custom_file


def test_roundtrip_document_and_vectors(tmp_path):
    """Roundtrip: simpan dokumen, baca kembali, dan pastikan vektor identik."""
    db_file = tmp_path / "test.db"
    doc_path = tmp_path / "laporan.pdf"
    doc_path.write_text("isi laporan dummy")

    texts = ["Paragraf satu mengenai AI.", "Paragraf dua mengenai VPS."]
    chunks = _create_mock_chunks(doc_path, texts)
    vectors = _create_random_vectors(2, 384)

    with Store(db_file) as store:
        store.upsert_document(
            doc_path,
            content_hash="hash_abc123",
            size=1024,
            mtime_ns=999888777,
            status="ok",
            error=None,
            chunks=chunks,
            vectors=vectors,
        )

        doc = store.get_document(doc_path)
        assert doc is not None
        assert isinstance(doc, DocumentRecord)
        assert doc.path == doc_path.resolve()
        assert doc.content_hash == "hash_abc123"
        assert doc.size == 1024
        assert doc.mtime_ns == 999888777
        assert doc.status == "ok"
        assert doc.error is None
        assert doc.chunk_count == 2
        assert doc.indexed_at != ""

        # Cari dengan vektor chunk pertama persis, skor harus mendekati 1.0
        hits = store.search(vectors[0], top_k=2)
        assert len(hits) == 2
        assert hits[0].text == texts[0]
        assert hits[0].chunk_index == 0
        assert hits[0].page == 1
        assert hits[0].path == doc_path.resolve()
        assert np.isclose(hits[0].score, 1.0, atol=1e-5)


def test_upsert_same_path_replaces_all_chunks(tmp_path):
    """Upsert pada path yang sama harus menghapus chunk lama tanpa sisa."""
    db_file = tmp_path / "test.db"
    doc_path = tmp_path / "doc.txt"

    with Store(db_file) as store:
        # Masukkan 3 chunk lama
        old_texts = ["A", "B", "C"]
        store.upsert_document(
            doc_path,
            content_hash="hash_old",
            size=500,
            mtime_ns=1000,
            status="ok",
            chunks=_create_mock_chunks(doc_path, old_texts),
            vectors=_create_random_vectors(3, 384, seed=1),
        )
        assert store.stats()["chunks"] == 3

        # Ganti dengan 2 chunk baru
        new_texts = ["X", "Y"]
        store.upsert_document(
            doc_path,
            content_hash="hash_new",
            size=600,
            mtime_ns=2000,
            status="ok",
            chunks=_create_mock_chunks(doc_path, new_texts),
            vectors=_create_random_vectors(2, 384, seed=2),
        )

        doc = store.get_document(doc_path)
        assert doc is not None
        assert doc.content_hash == "hash_new"
        assert doc.chunk_count == 2
        assert store.stats()["chunks"] == 2

        # Pastikan hanya teks baru yang ada
        q = _create_random_vectors(1, 384, seed=2)[0]
        hits = store.search(q, top_k=10)
        assert len(hits) == 2
        assert {h.text for h in hits} == {"X", "Y"}


def test_remove_document_and_cascade_delete(tmp_path):
    """remove_document dan ON DELETE CASCADE memastikan tidak ada chunk yatim."""
    db_file = tmp_path / "test.db"
    doc_path = tmp_path / "doc.txt"

    with Store(db_file) as store:
        store.upsert_document(
            doc_path,
            content_hash="h1",
            size=100,
            mtime_ns=10,
            status="ok",
            chunks=_create_mock_chunks(doc_path, ["Teks satu", "Teks dua"]),
            vectors=_create_random_vectors(2, 384),
        )
        assert store.stats()["documents"]["ok"] == 1
        assert store.stats()["chunks"] == 2

        # Hapus dokumen
        removed = store.remove_document(doc_path)
        assert removed is True
        assert store.get_document(doc_path) is None
        assert store.stats()["documents"]["ok"] == 0
        assert store.stats()["chunks"] == 0

        # Pencarian harus kosong
        q = _create_random_vectors(1, 384)[0]
        assert store.search(q, top_k=5) == []

        # Hapus file yang tidak ada mengembalikan False
        assert store.remove_document(tmp_path / "nonexistent.txt") is False


def test_search_matches_brute_force_argsort(tmp_path):
    """Pencarian argpartition harus menghasilkan skor dan urutan identik dengan argsort."""
    db_file = tmp_path / "test.db"
    doc_path = tmp_path / "data.txt"

    n_chunks = 100
    vectors = _create_random_vectors(n_chunks, 384, seed=77)
    texts = [f"Chunk number {i}" for i in range(n_chunks)]
    chunks = _create_mock_chunks(doc_path, texts)

    with Store(db_file) as store:
        store.upsert_document(
            doc_path,
            content_hash="hash_bench",
            size=10000,
            mtime_ns=5000,
            status="ok",
            chunks=chunks,
            vectors=vectors,
        )

        query_vec = _create_random_vectors(1, 384, seed=999)[0]

        # Brute force dengan argsort penuh
        all_scores = vectors @ query_vec
        expected_top10_indices = np.argsort(all_scores)[-10:][::-1]
        expected_top10_scores = all_scores[expected_top10_indices]

        # Cari dengan store
        hits = store.search(query_vec, top_k=10)
        assert len(hits) == 10

        actual_scores = np.array([h.score for h in hits], dtype=np.float32)
        actual_indices = [h.chunk_index for h in hits]

        # Skor menurun
        assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
        # Cocok persis dengan brute force
        assert actual_indices == list(expected_top10_indices)
        assert np.allclose(actual_scores, expected_top10_scores, atol=1e-5)


def test_search_edge_cases(tmp_path):
    """Pencarian pada kondisi batas: store kosong, top_k=0, top_k > n."""
    db_file = tmp_path / "test.db"

    with Store(db_file) as store:
        q = _create_random_vectors(1, 384)[0]
        # 1. Store kosong
        assert store.search(q, top_k=10) == []
        assert store.search(q, top_k=0) == []
        assert store.search(q, top_k=-5) == []

        # 2. Masukkan 2 chunk
        doc_path = tmp_path / "small.txt"
        store.upsert_document(
            doc_path,
            content_hash="h",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(doc_path, ["A", "B"]),
            vectors=_create_random_vectors(2, 384),
        )

        # top_k=0 menghasilkan list kosong
        assert store.search(q, top_k=0) == []
        assert store.search(q, top_k=-1) == []

        # top_k > n mengembalikan semua chunk
        hits = store.search(q, top_k=100)
        assert len(hits) == 2
        assert hits[0].score >= hits[1].score


def test_validation_errors(tmp_path):
    """Semua validasi salah bentuk/tipe harus melempar ValueError tanpa merusak database."""
    db_file = tmp_path / "test.db"
    doc_path = tmp_path / "doc.txt"

    with Store(db_file) as store:
        # Status salah
        with pytest.raises(ValueError, match="Status dokumen tidak valid"):
            store.upsert_document(
                doc_path,
                content_hash="h",
                size=1,
                mtime_ns=1,
                status="invalid_status",
            )

        # Chunks ada tapi vectors None
        chunks = _create_mock_chunks(doc_path, ["Teks"])
        with pytest.raises(ValueError, match="Vektor wajib diberikan"):
            store.upsert_document(
                doc_path,
                content_hash="h",
                size=1,
                mtime_ns=1,
                status="ok",
                chunks=chunks,
                vectors=None,
            )

        # Vectors bukan ndarray
        with pytest.raises(ValueError, match="Vektor harus berupa numpy ndarray"):
            store.upsert_document(
                doc_path,
                content_hash="h",
                size=1,
                mtime_ns=1,
                status="ok",
                chunks=chunks,
                vectors=[[0.1] * 384],  # type: ignore
            )

        # Vectors dtype bukan float32 (misal float64)
        v_float64 = np.zeros((1, 384), dtype=np.float64)
        with pytest.raises(ValueError, match="Dtype vektor harus float32"):
            store.upsert_document(
                doc_path,
                content_hash="h",
                size=1,
                mtime_ns=1,
                status="ok",
                chunks=chunks,
                vectors=v_float64,
            )

        # Vectors ndim bukan 2 (misal 1D)
        v_1d = np.zeros(384, dtype=np.float32)
        with pytest.raises(ValueError, match="Vektor harus berupa array 2D"):
            store.upsert_document(
                doc_path,
                content_hash="h",
                size=1,
                mtime_ns=1,
                status="ok",
                chunks=chunks,
                vectors=v_1d,
            )

        # Jumlah vektor != jumlah chunk
        v_wrong_count = np.zeros((2, 384), dtype=np.float32)
        with pytest.raises(ValueError, match="Jumlah vektor .* tidak sama"):
            store.upsert_document(
                doc_path,
                content_hash="h",
                size=1,
                mtime_ns=1,
                status="ok",
                chunks=chunks,  # len 1
                vectors=v_wrong_count,
            )

        # Dimensi vektor != embedding_dim
        v_wrong_dim = np.zeros((1, 512), dtype=np.float32)
        with pytest.raises(ValueError, match="Dimensi vektor .* tidak sesuai"):
            store.upsert_document(
                doc_path,
                content_hash="h",
                size=1,
                mtime_ns=1,
                status="ok",
                chunks=chunks,
                vectors=v_wrong_dim,
            )

        # Validasi query_vec pada search
        with pytest.raises(ValueError, match="query_vec harus berupa numpy ndarray"):
            store.search([0.1] * 384)  # type: ignore

        with pytest.raises(ValueError, match="Bentuk query_vec harus 1D"):
            store.search(np.zeros((1, 384), dtype=np.float32))

        with pytest.raises(ValueError, match="Dtype query_vec harus float32"):
            store.search(np.zeros(384, dtype=np.float64))

        # Pastikan database tetap bersih
        assert sum(store.stats()["documents"].values()) == 0
        assert store.stats()["chunks"] == 0


class _FailingChunk:
    """Chunk tiruan yang melempar RuntimeError ketika mengakses chunk ke-3."""

    def __init__(self, source: Path, index: int) -> None:
        self.source = source
        self.index = index
        self.page = 1

    @property
    def text(self) -> str:
        if self.index == 2:  # Chunk ke-3 (indeks 0, 1, 2)
            raise RuntimeError("Simulasi kegagalan pada chunk ke-3!")
        return f"Teks chunk {self.index}"


def test_atomicity_rollback_on_failure(tmp_path):
    """Kegagalan di tengah penyisipan harus memicu rollback dan menjaga dokumen lama."""
    db_file = tmp_path / "test.db"
    doc_path = tmp_path / "doc.txt"

    with Store(db_file) as store:
        # Masukkan dokumen awal dengan 2 chunk
        init_chunks = _create_mock_chunks(doc_path, ["Awal 1", "Awal 2"])
        init_vectors = _create_random_vectors(2, 384, seed=10)
        store.upsert_document(
            doc_path,
            content_hash="hash_awal",
            size=100,
            mtime_ns=1,
            status="ok",
            chunks=init_chunks,
            vectors=init_vectors,
        )

        doc_before = store.get_document(doc_path)
        assert doc_before is not None
        assert doc_before.content_hash == "hash_awal"
        assert store.stats()["chunks"] == 2

        # Coba perbarui dokumen dengan chunk yang gagal di chunk ke-3
        failing_chunks = [_FailingChunk(doc_path, i) for i in range(3)]
        new_vectors = _create_random_vectors(3, 384, seed=20)

        with pytest.raises(RuntimeError, match="Simulasi kegagalan pada chunk ke-3"):
            store.upsert_document(
                doc_path,
                content_hash="hash_gagal",
                size=200,
                mtime_ns=2,
                status="ok",
                chunks=failing_chunks,  # type: ignore
                vectors=new_vectors,
            )

    # Buka ulang database dan verifikasi data lama tetap utuh
    with Store(db_file) as store:
        doc = store.get_document(doc_path)
        assert doc is not None
        assert doc.content_hash == "hash_awal"
        assert doc.chunk_count == 2
        assert store.stats()["chunks"] == 2

        hits = store.search(init_vectors[0], top_k=5)
        assert len(hits) == 2
        assert {h.text for h in hits} == {"Awal 1", "Awal 2"}


def test_persistence_close_and_reopen(tmp_path):
    """Data harus persisten saat store ditutup dan dibuka kembali."""
    db_file = tmp_path / "persist.db"
    doc_path = tmp_path / "dokumen.txt"
    texts = ["Informasi satu", "Informasi dua"]
    vectors = _create_random_vectors(2, 384, seed=33)

    # Sesi 1: tulis
    with Store(db_file) as s1:
        s1.upsert_document(
            doc_path,
            content_hash="hash_persis",
            size=123,
            mtime_ns=456,
            status="ok",
            chunks=_create_mock_chunks(doc_path, texts),
            vectors=vectors,
        )

    # Sesi 2: baca dan cari
    with Store(db_file) as s2:
        doc = s2.get_document(doc_path)
        assert doc is not None
        assert doc.content_hash == "hash_persis"
        assert doc.chunk_count == 2

        hits = s2.search(vectors[1], top_k=2)
        assert len(hits) == 2
        assert hits[0].text == "Informasi dua"
        assert np.isclose(hits[0].score, 1.0, atol=1e-5)


def test_store_model_mismatch(tmp_path):
    """StoreModelMismatch harus dilempar jika model atau dimensi berbeda dari metadata."""
    db_file = tmp_path / "meta_test.db"

    # Inisialisasi awal dengan model A dan dim 384
    with Store(db_file, model_name="model-a", embedding_dim=384) as s:
        assert s.model_name == "model-a"

    # Buka dengan model berbeda
    with pytest.raises(StoreModelMismatch) as exc_model:
        Store(db_file, model_name="model-b", embedding_dim=384)
    msg_model = str(exc_model.value)
    assert "model-a" in msg_model
    assert "model-b" in msg_model

    # Buka dengan dimensi berbeda
    with pytest.raises(StoreModelMismatch) as exc_dim:
        Store(db_file, model_name="model-a", embedding_dim=512)
    msg_dim = str(exc_dim.value)
    assert "384" in msg_dim
    assert "512" in msg_dim


def test_corrupt_database_raises_store_error(tmp_path):
    """File bukan SQLite atau rusak harus menghasilkan StoreError tanpa DatabaseError mentah."""
    bad_file = tmp_path / "corrupt.db"
    bad_file.write_bytes(b"INI BUKAN FILE SQLITE SAMA SEKALI GIBBERISH")

    with pytest.raises(StoreError) as exc_info:
        Store(bad_file)
    assert "Database SQLite rusak atau tidak valid" in str(exc_info.value)
    assert not isinstance(exc_info.value, StoreModelMismatch)


def test_empty_and_error_status_documents(tmp_path):
    """Dokumen status empty dan error harus tersimpan dan terbaca dengan baik."""
    db_file = tmp_path / "status.db"
    doc_empty = tmp_path / "kosong.txt"
    doc_error = tmp_path / "rusak.bin"

    with Store(db_file) as store:
        # Dokumen kosong
        store.upsert_document(
            doc_empty,
            content_hash="hash_empty",
            size=0,
            mtime_ns=100,
            status="empty",
            error=None,
            chunks=(),
            vectors=None,
        )

        # Dokumen error
        store.upsert_document(
            doc_error,
            content_hash="hash_err",
            size=999,
            mtime_ns=200,
            status="error",
            error="Berkas terenkripsi dengan kata sandi",
            chunks=(),
            vectors=None,
        )

        d_empty = store.get_document(doc_empty)
        assert d_empty is not None
        assert d_empty.status == "empty"
        assert d_empty.chunk_count == 0
        assert d_empty.error is None

        d_err = store.get_document(doc_error)
        assert d_err is not None
        assert d_err.status == "error"
        assert d_err.chunk_count == 0
        assert d_err.error == "Berkas terenkripsi dengan kata sandi"

        stats = store.stats()
        assert stats["documents"]["empty"] == 1
        assert stats["documents"]["error"] == 1
        assert sum(stats["documents"].values()) == 2
        assert stats["chunks"] == 0


def test_utf8_indonesian_text_and_path_with_spaces(tmp_path):
    """Mendukung karakter UTF-8 Indonesia dan path dengan spasi."""
    db_file = tmp_path / "database utama.db"
    folder_spasi = tmp_path / "Folder Rahasia Dokumen Penting"
    folder_spasi.mkdir()
    doc_path = folder_spasi / "Surat Keputusan Direksi 2026.docx"

    teks_indo = (
        "Pasal 1: Penggunaan kecerdasan artifisial (AI) harus bertanggung jawab. "
        "Biaya operasional sebesar Rp150.000,00 per bulan disetujui tanpa 'markup'."
    )
    chunks = _create_mock_chunks(doc_path, [teks_indo])
    vectors = _create_random_vectors(1, 384)

    with Store(db_file) as store:
        store.upsert_document(
            doc_path,
            content_hash="hash_utf8",
            size=2048,
            mtime_ns=8888,
            status="ok",
            chunks=chunks,
            vectors=vectors,
        )

        doc = store.get_document(doc_path)
        assert doc is not None
        assert doc.path == doc_path.resolve()

        hits = store.search(vectors[0], top_k=1)
        assert len(hits) == 1
        assert hits[0].path == doc_path.resolve()
        assert "Rp150.000,00" in hits[0].text
        assert "bertanggung jawab" in hits[0].text


def test_list_documents_and_stats(tmp_path):
    """Memverifikasi list_documents dan stats()."""
    db_file = tmp_path / "stats.db"

    with Store(db_file) as store:
        d1 = tmp_path / "doc1.txt"
        d2 = tmp_path / "doc2.txt"

        store.upsert_document(
            d1,
            content_hash="h1",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(d1, ["Satu", "Dua"]),
            vectors=_create_random_vectors(2, 384),
        )
        store.upsert_document(
            d2,
            content_hash="h2",
            size=20,
            mtime_ns=2,
            status="empty",
        )

        docs = store.list_documents()
        assert len(docs) == 2
        paths = {d.path for d in docs}
        assert paths == {d1.resolve(), d2.resolve()}

        st = store.stats()
        assert set(st.keys()) == {"documents", "chunks", "db_size_bytes"}
        assert st["documents"]["ok"] == 1
        assert st["documents"]["empty"] == 1
        assert st["documents"]["error"] == 0
        assert sum(st["documents"].values()) == 2
        assert st["chunks"] == 2
        assert st["db_size_bytes"] > 0


def test_cross_connection_cache_invalidation_data_version(tmp_path):
    """Store A harus menyegarkan cache otomatis saat Store B menambah dokumen via PRAGMA data_version."""
    db_file = tmp_path / "multi_conn.db"

    v1 = _create_random_vectors(1, 384, seed=101)[0]
    v2 = _create_random_vectors(1, 384, seed=102)[0]

    with Store(db_file) as store_a, Store(db_file) as store_b:
        d1 = tmp_path / "doc1.txt"
        store_b.upsert_document(
            d1,
            content_hash="h1",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(d1, ["Dokumen pertama"]),
            vectors=np.array([v1]),
        )

        # Store A melakukan search pertama -> memuat cache ke memori
        hits_a = store_a.search(v1, top_k=5)
        assert len(hits_a) == 1
        assert hits_a[0].text == "Dokumen pertama"
        ver_before = store_a._cached_data_version
        assert ver_before is not None

        # Store B menambahkan dokumen kedua
        d2 = tmp_path / "doc2.txt"
        store_b.upsert_document(
            d2,
            content_hash="h2",
            size=20,
            mtime_ns=2,
            status="ok",
            chunks=_create_mock_chunks(d2, ["Dokumen kedua"]),
            vectors=np.array([v2]),
        )

        # Buktikan data_version berubah dari koneksi Store A
        conn_a = store_a._ensure_open()
        cur_a = conn_a.cursor()
        cur_a.execute("PRAGMA data_version;")
        ver_after = cur_a.fetchone()[0]
        assert ver_after != ver_before

        # Store A melakukan pencarian lagi -> harus otomatis mendeteksi perubahan dan menemukan doc2
        hits_a_new = store_a.search(v2, top_k=5)
        assert len(hits_a_new) == 2
        assert any(h.text == "Dokumen kedua" for h in hits_a_new)


def test_locked_database_raises_store_error(tmp_path):
    """Database yang terkunci proses lain harus melempar StoreError dengan pesan proses lain dan data lama aman."""
    import sqlite3

    db_file = tmp_path / "locked.db"
    doc_path = tmp_path / "doc.txt"

    # Inisialisasi dokumen awal
    init_v = _create_random_vectors(1, 384, seed=201)
    with Store(db_file) as s:
        s.upsert_document(
            doc_path,
            content_hash="h_init",
            size=50,
            mtime_ns=100,
            status="ok",
            chunks=_create_mock_chunks(doc_path, ["Teks awal"]),
            vectors=init_v,
        )

    # Buka koneksi sqlite3 terpisah dan tahan lock eksklusif dengan BEGIN IMMEDIATE
    external_conn = sqlite3.connect(str(db_file), timeout=0.1)
    external_conn.execute("BEGIN IMMEDIATE;")

    try:
        # Coba upsert dengan Store bertimeout rendah (0.1s)
        new_v = _create_random_vectors(1, 384, seed=202)
        with Store(db_file, timeout=0.1) as locked_store:
            with pytest.raises(StoreError) as exc_info:
                locked_store.upsert_document(
                    doc_path,
                    content_hash="h_new",
                    size=100,
                    mtime_ns=200,
                    status="ok",
                    chunks=_create_mock_chunks(doc_path, ["Teks baru"]),
                    vectors=new_v,
                )

            err_msg = str(exc_info.value)
            assert "sedang digunakan oleh proses lain" in err_msg
            assert "rusak atau tidak valid" not in err_msg
    finally:
        external_conn.rollback()
        external_conn.close()

    # Pastikan data lama tetap utuh
    with Store(db_file) as s_verify:
        doc = s_verify.get_document(doc_path)
        assert doc is not None
        assert doc.content_hash == "h_init"
        hits = s_verify.search(init_v[0], top_k=5)
        assert len(hits) == 1
        assert hits[0].text == "Teks awal"


def test_data_version_read_before_load_avoids_stale_cache(tmp_path):
    """Pembacaan data_version sebelum pemuatan vektor mencegah cache basi saat ada penulisan konkuren."""
    db_file = tmp_path / "race.db"
    doc1_path = tmp_path / "doc1.txt"
    doc2_path = tmp_path / "doc2.txt"

    v1 = _create_random_vectors(1, 384, seed=301)[0]
    v2 = _create_random_vectors(1, 384, seed=302)[0]

    with Store(db_file) as s1, Store(db_file) as s2:
        s2.upsert_document(
            doc1_path,
            content_hash="h1",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(doc1_path, ["Dokumen pertama"]),
            vectors=np.array([v1]),
        )

        real_conn = s1._conn
        assert real_conn is not None
        inserted_concurrently = False

        class _InterceptingCursor:
            def __init__(self, cur: Any) -> None:
                self._cur = cur

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                nonlocal inserted_concurrently
                res = self._cur.execute(sql, *args, **kwargs)
                if (
                    "SELECT id, embedding FROM chunks" in sql
                    and not inserted_concurrently
                ):
                    s2.upsert_document(
                        doc2_path,
                        content_hash="h2",
                        size=20,
                        mtime_ns=2,
                        status="ok",
                        chunks=_create_mock_chunks(doc2_path, ["Dokumen kedua"]),
                        vectors=np.array([v2]),
                    )
                    inserted_concurrently = True
                return res

            def fetchone(self) -> Any:
                return self._cur.fetchone()

            def fetchall(self) -> Any:
                return self._cur.fetchall()

            def __iter__(self) -> Any:
                return iter(self._cur)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cur, name)

        class _InterceptingConn:
            def __init__(self, conn: Any) -> None:
                self._conn = conn

            def cursor(self) -> Any:
                return _InterceptingCursor(self._conn.cursor())

            def __getattr__(self, name: str) -> Any:
                return getattr(self._conn, name)

        s1._conn = _InterceptingConn(real_conn)  # type: ignore

        # Search pertama memicu pemuatan cache.
        # Saat query SELECT id, embedding dijalankan, s2 menyisipkan dokumen kedua secara konkuren.
        hits_1 = s1.search(v1, top_k=5)
        assert len(hits_1) == 1
        assert hits_1[0].text == "Dokumen pertama"
        assert inserted_concurrently is True

        # Search kedua tanpa penulisan lain harus menemukan dokumen baru
        hits_2 = s1.search(v2, top_k=5)
        assert len(hits_2) == 2
        assert any(h.text == "Dokumen kedua" for h in hits_2)

        # Pulihkan koneksi asli agar s1.close() tertib
        s1._conn = real_conn


def test_chunk_ids_autoincrement_never_reused(tmp_path):
    """Skema chunks dengan AUTOINCREMENT menjamin ID chunk baru lebih besar dari ID lama."""
    import sqlite3

    db_file = tmp_path / "autoincrement.db"
    doc_path = tmp_path / "doc.txt"

    v1 = _create_random_vectors(2, 384, seed=401)
    v2 = _create_random_vectors(2, 384, seed=402)

    with Store(db_file) as s:
        s.upsert_document(
            doc_path,
            content_hash="h1",
            size=100,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(doc_path, ["Teks A", "Teks B"]),
            vectors=v1,
        )

    # Baca ID chunk awal via koneksi terpisah
    conn_raw = sqlite3.connect(str(db_file))
    cur = conn_raw.cursor()
    cur.execute("SELECT id FROM chunks ORDER BY id ASC;")
    first_ids = [r[0] for r in cur.fetchall()]
    assert len(first_ids) == 2

    # Upsert kedua untuk path yang sama dengan jumlah chunk yang sama
    with Store(db_file) as s:
        s.upsert_document(
            doc_path,
            content_hash="h2",
            size=100,
            mtime_ns=2,
            status="ok",
            chunks=_create_mock_chunks(doc_path, ["Teks C", "Teks D"]),
            vectors=v2,
        )

    cur.execute("SELECT id FROM chunks ORDER BY id ASC;")
    second_ids = [r[0] for r in cur.fetchall()]
    conn_raw.close()

    assert len(second_ids) == 2
    # Semua ID chunk setelah upsert kedua harus lebih besar dari semua ID sebelumnya
    assert min(second_ids) > max(first_ids)


def test_concurrent_insert_during_vector_cache_load(tmp_path):
    """Penyisipan konkuren di antara count(*) dan fetch embedding harus tetap konsisten tanpa IndexError mentah."""
    db_file = tmp_path / "insert_race.db"
    doc1_path = tmp_path / "doc1.txt"
    doc2_path = tmp_path / "doc2.txt"

    v1 = _create_random_vectors(1, 384, seed=501)[0]
    v2 = _create_random_vectors(1, 384, seed=502)[0]

    with Store(db_file) as s1, Store(db_file) as s2:
        s2.upsert_document(
            doc1_path,
            content_hash="h1",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(doc1_path, ["Dokumen 1"]),
            vectors=np.array([v1]),
        )

        real_conn = s1._conn
        assert real_conn is not None
        inserted_concurrently = False

        class _InterceptingCursor:
            def __init__(self, cur: Any) -> None:
                self._cur = cur

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                nonlocal inserted_concurrently
                res = self._cur.execute(sql, *args, **kwargs)
                if "SELECT count(*) FROM chunks" in sql and not inserted_concurrently:
                    s2.upsert_document(
                        doc2_path,
                        content_hash="h2",
                        size=20,
                        mtime_ns=2,
                        status="ok",
                        chunks=_create_mock_chunks(doc2_path, ["Dokumen 2"]),
                        vectors=np.array([v2]),
                    )
                    inserted_concurrently = True
                return res

            def fetchone(self) -> Any:
                return self._cur.fetchone()

            def fetchall(self) -> Any:
                return self._cur.fetchall()

            def __iter__(self) -> Any:
                return iter(self._cur)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cur, name)

        class _InterceptingConn:
            def __init__(self, conn: Any) -> None:
                self._conn = conn

            def cursor(self) -> Any:
                return _InterceptingCursor(self._conn.cursor())

            def rollback(self) -> Any:
                return self._conn.rollback()

            @property
            def in_transaction(self) -> bool:
                return self._conn.in_transaction

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._conn, name)

        s1._conn = _InterceptingConn(real_conn)  # type: ignore

        try:
            hits = s1.search(v1, top_k=5)
            assert inserted_concurrently is True
            assert len(hits) == 1
            assert s1._cached_chunk_ids is not None
            assert len(s1._cached_chunk_ids) == 1
            assert 0 not in s1._cached_chunk_ids
            assert s1._cached_matrix is not None
            assert len(s1._cached_matrix) == 1
            assert hits[0].path == doc1_path.resolve()
        finally:
            s1._conn = real_conn

        # Search kedua tanpa penulisan lain: cache dimuat ulang karena data_version berubah
        hits2 = s1.search(v1, top_k=5)
        assert s1._cached_chunk_ids is not None
        assert len(s1._cached_chunk_ids) == 2
        assert 0 not in s1._cached_chunk_ids
        hit_paths = {h.path for h in hits2}
        assert hit_paths == {doc1_path.resolve(), doc2_path.resolve()}


def test_concurrent_delete_during_vector_cache_load(tmp_path):
    """Penghapusan konkuren di antara count(*) dan fetch embedding tidak boleh meninggalkan chunk ID 0 atau vektor tak terisi."""
    db_file = tmp_path / "delete_race.db"
    doc1_path = tmp_path / "doc1.txt"
    doc2_path = tmp_path / "doc2.txt"

    v1 = _create_random_vectors(1, 384, seed=601)[0]
    v2 = _create_random_vectors(1, 384, seed=602)[0]

    with Store(db_file) as s1, Store(db_file) as s2:
        s2.upsert_document(
            doc1_path,
            content_hash="h1",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(doc1_path, ["Dokumen 1"]),
            vectors=np.array([v1]),
        )
        s2.upsert_document(
            doc2_path,
            content_hash="h2",
            size=20,
            mtime_ns=2,
            status="ok",
            chunks=_create_mock_chunks(doc2_path, ["Dokumen 2"]),
            vectors=np.array([v2]),
        )

        real_conn = s1._conn
        assert real_conn is not None
        deleted_concurrently = False

        class _InterceptingCursor:
            def __init__(self, cur: Any) -> None:
                self._cur = cur

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                nonlocal deleted_concurrently
                res = self._cur.execute(sql, *args, **kwargs)
                if "SELECT count(*) FROM chunks" in sql and not deleted_concurrently:
                    s2.remove_document(doc2_path)
                    deleted_concurrently = True
                return res

            def fetchone(self) -> Any:
                return self._cur.fetchone()

            def fetchall(self) -> Any:
                return self._cur.fetchall()

            def __iter__(self) -> Any:
                return iter(self._cur)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cur, name)

        class _InterceptingConn:
            def __init__(self, conn: Any) -> None:
                self._conn = conn

            def cursor(self) -> Any:
                return _InterceptingCursor(self._conn.cursor())

            def rollback(self) -> Any:
                return self._conn.rollback()

            @property
            def in_transaction(self) -> bool:
                return self._conn.in_transaction

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._conn, name)

        s1._conn = _InterceptingConn(real_conn)  # type: ignore

        try:
            hits = s1.search(v1, top_k=5)
            assert deleted_concurrently is True
            assert s1._cached_chunk_ids is not None
            assert len(s1._cached_chunk_ids) == 2
            assert 0 not in s1._cached_chunk_ids
            assert s1._cached_matrix is not None
            for row in s1._cached_matrix:
                norm = float(np.linalg.norm(row))
                assert 0.99 <= norm <= 1.01
            assert all(h.path != doc2_path.resolve() for h in hits)
        finally:
            s1._conn = real_conn

        # Search kedua tanpa penulisan lain: cache dimuat ulang hanya memuat doc1
        hits2 = s1.search(v1, top_k=5)
        assert s1._cached_chunk_ids is not None
        assert len(s1._cached_chunk_ids) == 1
        assert 0 not in s1._cached_chunk_ids
        assert len(hits2) == 1
        assert hits2[0].path == doc1_path.resolve()


def test_failed_search_releases_read_transaction(tmp_path):
    """Kegagalan di tengah search/load_vector_cache harus merollback transaksi baca sehingga upsert berikutnya tetap berhasil."""
    import sqlite3

    db_file = tmp_path / "rollback_test.db"
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"

    v = _create_random_vectors(1, 384, seed=701)

    with Store(db_file) as s:
        s.upsert_document(
            doc1,
            content_hash="h1",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=_create_mock_chunks(doc1, ["Teks 1"]),
            vectors=v,
        )

        real_conn = s._conn
        assert real_conn is not None

        class _FailingCursor:
            def __init__(self, cur: Any) -> None:
                self._cur = cur

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                if "SELECT id, embedding FROM chunks" in sql:
                    raise sqlite3.OperationalError("Simulasi error baca")
                return self._cur.execute(sql, *args, **kwargs)

            def fetchone(self) -> Any:
                return self._cur.fetchone()

            def fetchall(self) -> Any:
                return self._cur.fetchall()

            def __iter__(self) -> Any:
                return iter(self._cur)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cur, name)

        class _FailingConn:
            def __init__(self, conn: Any) -> None:
                self._conn = conn

            def cursor(self) -> Any:
                return _FailingCursor(self._conn.cursor())

            def rollback(self) -> Any:
                return self._conn.rollback()

            @property
            def in_transaction(self) -> bool:
                return self._conn.in_transaction

            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._conn, name)

        s._conn = _FailingConn(real_conn)  # type: ignore

        with pytest.raises(StoreError, match="Gagal memuat cache vektor"):
            s.search(v[0])

        assert real_conn.in_transaction is False

        # Pulihkan koneksi asli
        s._conn = real_conn

        # upsert_document pada store yang SAMA harus tetap berhasil tanpa terhalang transaksi lama
        s.upsert_document(
            doc2,
            content_hash="h2",
            size=20,
            mtime_ns=2,
            status="ok",
            chunks=_create_mock_chunks(doc2, ["Teks 2"]),
            vectors=v,
        )

        assert len(s.list_documents()) == 2
        assert s.get_document(doc2) is not None


def test_idx_documents_hash_exists(tmp_path: Path) -> None:
    """Index idx_documents_hash harus ada di sqlite_master."""
    s = Store(tmp_path / "test.db")
    assert s._conn is not None
    rows = s._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_documents_hash'"
    ).fetchall()
    assert len(rows) == 1


def test_find_documents_by_hash_filters_error_and_orders_by_id(tmp_path: Path) -> None:
    """find_documents_by_hash hanya mengembalikan dokumen 'ok'/'empty' terurut id."""
    s = Store(tmp_path / "test.db")
    doc_ok1 = tmp_path / "ok1.txt"
    doc_err = tmp_path / "err.txt"
    doc_ok2 = tmp_path / "ok2.txt"
    doc_empty = tmp_path / "empty.txt"

    v = _create_random_vectors(1)
    s.upsert_document(
        doc_ok1,
        content_hash="hash_same",
        size=10,
        mtime_ns=1,
        status="ok",
        chunks=_create_mock_chunks(doc_ok1, ["Teks 1"]),
        vectors=v,
    )
    s.upsert_document(
        doc_err,
        content_hash="hash_same",
        size=10,
        mtime_ns=1,
        status="error",
        error="gagal baca",
    )
    s.upsert_document(
        doc_ok2,
        content_hash="hash_same",
        size=10,
        mtime_ns=1,
        status="ok",
        chunks=_create_mock_chunks(doc_ok2, ["Teks 2"]),
        vectors=v,
    )
    s.upsert_document(
        doc_empty,
        content_hash="hash_same",
        size=0,
        mtime_ns=1,
        status="empty",
        error="kosong",
    )

    found = s.find_documents_by_hash("hash_same")
    assert len(found) == 3
    statuses = [d.status for d in found]
    assert "error" not in statuses
    assert statuses == ["ok", "ok", "empty"]
    # Terurut id (ok1 masuk lebih dulu dari ok2, lalu empty)
    paths = [d.path for d in found]
    assert paths == [doc_ok1.resolve(), doc_ok2.resolve(), doc_empty.resolve()]


def test_get_document_chunks_roundtrip_and_order(tmp_path: Path) -> None:
    """get_document_chunks mengembalikan chunk terurut index dan vektor identik."""
    s = Store(tmp_path / "test.db")
    doc = tmp_path / "multi.txt"
    n = 4
    v = _create_random_vectors(n)
    chunks = [
        Chunk(f"Teks bagian {i}", doc, page=1 if i < 2 else 2, index=i)
        for i in range(n)
    ]
    s.upsert_document(
        doc,
        content_hash="hash_multi",
        size=100,
        mtime_ns=1,
        status="ok",
        chunks=chunks,
        vectors=v,
    )

    got_chunks, got_matrix = s.get_document_chunks(doc)
    assert len(got_chunks) == n
    assert got_matrix.shape == (n, 384)
    for i in range(n):
        assert got_chunks[i].index == i
        assert got_chunks[i].text == f"Teks bagian {i}"
        assert got_chunks[i].source == doc.resolve()
        np.testing.assert_array_equal(got_matrix[i], v[i])


def test_get_document_chunks_not_found_or_no_chunks(tmp_path: Path) -> None:
    """Dokumen tidak ada atau tanpa chunk menghasilkan ([], array bentuk (0, dim))."""
    s = Store(tmp_path / "test.db")
    # Dokumen tidak ada
    c1, m1 = s.get_document_chunks(tmp_path / "nonexistent.txt")
    assert c1 == []
    assert m1.shape == (0, 384)

    # Dokumen empty tanpa chunk
    doc_empty = tmp_path / "empty.txt"
    s.upsert_document(
        doc_empty,
        content_hash="empty_hash",
        size=0,
        mtime_ns=1,
        status="empty",
        error="File kosong",
    )
    c2, m2 = s.get_document_chunks(doc_empty)
    assert c2 == []
    assert m2.shape == (0, 384)
