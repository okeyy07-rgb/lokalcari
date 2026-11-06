"""Tests untuk indexer: scan_folder dan index_folder."""

from __future__ import annotations

import hashlib
import itertools
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from lokalcari.chunker import Chunk
from lokalcari.embedder import Embedder, EmbeddingError, ModelNotInstalled
from lokalcari.indexer import (
    IndexerError,
    ProgressEvent,
    index_folder,
    scan_folder,
)
from lokalcari.store import Store

# ============================================================
# Fake Embedder Backend
# ============================================================


class _FakeBackend:
    """Backend embedding deterministik: vektor dari hash teks, dinormalisasi."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.embed_count = 0  # jumlah teks yang di-embed

    def embed(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        results = []
        for t in texts:
            self.embed_count += 1
            # Vektor deterministik dari hash teks
            h = hashlib.sha256(t.encode()).digest()
            # Buat array float32 dari nilai byte 0-255 yang diskalakan
            needed = self.dim
            repeated = (h * (needed // len(h) + 1))[:needed]
            raw = np.array([float(b) - 128.0 for b in repeated], dtype=np.float32)
            norm = float(np.linalg.norm(raw))
            if norm > 1e-6:
                raw = raw / norm
            results.append(raw)
        return results


def _make_embedder(dim: int = 384) -> tuple[Embedder, _FakeBackend]:
    backend = _FakeBackend(dim=dim)
    embedder = Embedder(backend=backend)
    return embedder, backend


# ============================================================
# Helpers
# ============================================================


def _write(
    path: Path, content: str = "Teks konten dokumen untuk pengujian indexer."
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_store(tmp_path: Path, dim: int = 384) -> Store:
    return Store(tmp_path / "index.db", embedding_dim=dim)


def _collect_progress(events: list[ProgressEvent]) -> None:
    """Callback progress yang mengumpulkan event."""
    # dipakai dengan closure


# ============================================================
# Tests: scan_folder
# ============================================================


class TestScanFolder:
    def test_basic_supported_extensions(self, tmp_path: Path) -> None:
        _write(tmp_path / "doc.txt")
        _write(tmp_path / "doc.md")
        _write(tmp_path / "doc.pdf")
        _write(tmp_path / "doc.docx")
        _write(tmp_path / "doc.xlsx")  # tidak didukung
        _write(tmp_path / "doc.csv")  # tidak didukung

        result = scan_folder(tmp_path)
        exts = {p.suffix.lower() for p in result}
        assert ".txt" in exts
        assert ".md" in exts
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".xlsx" not in exts
        assert ".csv" not in exts

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        _write(tmp_path / "DOC.TXT")
        _write(tmp_path / "Report.PDF")
        _write(tmp_path / "note.Md")
        result = scan_folder(tmp_path)
        names = {p.name for p in result}
        assert "DOC.TXT" in names
        assert "Report.PDF" in names
        assert "note.Md" in names

    def test_skip_temp_word_files(self, tmp_path: Path) -> None:
        _write(tmp_path / "~$document.docx")
        _write(tmp_path / "real.docx")
        result = scan_folder(tmp_path)
        names = {p.name for p in result}
        assert "~$document.docx" not in names
        assert "real.docx" in names

    def test_skip_libreoffice_lock(self, tmp_path: Path) -> None:
        _write(tmp_path / ".~lock.presentation.pptx#")
        _write(tmp_path / "real.txt")
        result = scan_folder(tmp_path)
        names = {p.name for p in result}
        assert ".~lock.presentation.pptx#" not in names
        assert "real.txt" in names

    def test_skip_hidden_files(self, tmp_path: Path) -> None:
        _write(tmp_path / ".hidden.txt")
        _write(tmp_path / "visible.txt")
        result = scan_folder(tmp_path)
        names = {p.name for p in result}
        assert ".hidden.txt" not in names
        assert "visible.txt" in names

    def test_skip_ignore_dirs(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        _write(tmp_path / ".git" / "note.txt")
        (tmp_path / "node_modules").mkdir()
        _write(tmp_path / "node_modules" / "readme.md")
        (tmp_path / "__pycache__").mkdir()
        _write(tmp_path / "__pycache__" / "cache.txt")
        _write(tmp_path / "real.txt")
        result = scan_folder(tmp_path)
        paths_str = [str(p) for p in result]
        assert not any(".git" in s for s in paths_str)
        assert not any("node_modules" in s for s in paths_str)
        assert not any("__pycache__" in s for s in paths_str)
        assert any("real.txt" in s for s in paths_str)

    def test_skip_dot_dirs(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden_dir").mkdir()
        _write(tmp_path / ".hidden_dir" / "doc.txt")
        _write(tmp_path / "visible.txt")
        result = scan_folder(tmp_path)
        paths_str = [str(p) for p in result]
        assert not any(".hidden_dir" in s for s in paths_str)
        assert any("visible.txt" in s for s in paths_str)

    def test_sorted_deterministic(self, tmp_path: Path) -> None:
        _write(tmp_path / "z.txt")
        _write(tmp_path / "a.txt")
        _write(tmp_path / "m.txt")
        result = scan_folder(tmp_path)
        assert result == sorted(result)

    def test_path_with_spaces_and_utf8(self, tmp_path: Path) -> None:
        folder = tmp_path / "folder dengan spasi" / "sub-berkas"
        folder.mkdir(parents=True)
        fpath = folder / "dokumen bahasa 日本語.txt"
        _write(fpath)
        result = scan_folder(tmp_path)
        assert fpath.resolve() in result

    def test_root_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IndexerError):
            scan_folder(tmp_path / "tidak_ada")

    def test_root_not_dir_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        _write(f)
        with pytest.raises(IndexerError):
            scan_folder(f)

    def test_unreadable_dir_calls_on_error(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(sub / "doc.txt")

        errors_found: list[tuple[Path, Exception]] = []

        def _on_error(p: Path, e: Exception) -> None:
            errors_found.append((p, e))

        # Simulasi error saat stat direktori dengan mocking os.lstat
        original_lstat = os.lstat

        def _mock_lstat(path: str, *args: Any, **kwargs: Any) -> os.stat_result:
            if "sub" in str(path) and not str(path).endswith(".txt"):
                raise OSError("Permission denied")
            return original_lstat(path, *args, **kwargs)

        with patch("os.lstat", side_effect=_mock_lstat):
            scan_folder(tmp_path, on_error=_on_error)

        assert len(errors_found) >= 1

    def test_scan_folder_reports_unreadable_dir_via_scandir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / "ok" / "a.txt")
        locked_dir = tmp_path / "terkunci"
        _write(locked_dir / "b.txt")

        errors_found: list[tuple[Path, Exception]] = []
        original_scandir = os.scandir

        def _mock_scandir(path: str | os.PathLike[str]) -> Any:
            if os.path.basename(os.fspath(path)) == "terkunci":
                raise PermissionError("akses ditolak")
            return original_scandir(path)

        monkeypatch.setattr(os, "scandir", _mock_scandir)
        result = scan_folder(
            tmp_path,
            on_error=lambda path, error: errors_found.append((path, error)),
        )

        assert result == [(tmp_path / "ok" / "a.txt").resolve()]
        assert len(errors_found) == 1
        assert errors_found[0][0] == locked_dir
        assert isinstance(errors_found[0][1], PermissionError)

    @pytest.mark.skipif(
        os.name != "nt" and not hasattr(os, "symlink"),
        reason="Symlink tidak tersedia",
    )
    def test_skip_symlink_files(self, tmp_path: Path) -> None:
        real = tmp_path / "real.txt"
        _write(real)
        link = tmp_path / "link.txt"
        try:
            os.symlink(str(real), str(link))
        except (OSError, NotImplementedError):
            pytest.skip("Symlink tidak bisa dibuat di OS ini")

        result = scan_folder(tmp_path)
        names = {p.name for p in result}
        assert "real.txt" in names
        assert "link.txt" not in names


# ============================================================
# Tests: store tambahan (find_documents_by_hash, get_document_chunks)
# ============================================================


class TestStorePart1:
    def test_find_documents_by_hash_ok(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        path = tmp_path / "a.txt"
        v = np.ones((1, 384), dtype=np.float32)
        v /= np.linalg.norm(v)
        store.upsert_document(
            path,
            content_hash="abc123",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=[Chunk("Teks", path, None, 0)],
            vectors=v,
        )
        docs = store.find_documents_by_hash("abc123")
        assert len(docs) == 1
        assert docs[0].path.resolve() == path.resolve()

    def test_find_documents_by_hash_no_error_status(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        path = tmp_path / "a.txt"
        store.upsert_document(
            path,
            content_hash="xyz",
            size=10,
            mtime_ns=1,
            status="error",
            error="gagal",
        )
        docs = store.find_documents_by_hash("xyz")
        assert docs == []

    def test_find_documents_by_hash_includes_empty(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        path = tmp_path / "empty.txt"
        store.upsert_document(
            path,
            content_hash="emptyhash",
            size=0,
            mtime_ns=1,
            status="empty",
            error="File kosong",
        )
        docs = store.find_documents_by_hash("emptyhash")
        assert len(docs) == 1

    def test_get_document_chunks_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        path = tmp_path / "doc.txt"
        v1 = np.array([[0.1] * 384], dtype=np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = np.array([[0.9] * 384], dtype=np.float32)
        v2 /= np.linalg.norm(v2)
        vecs = np.vstack([v1, v2])
        chunks = [
            Chunk("Bagian satu", path, 1, 0),
            Chunk("Bagian dua", path, 1, 1),
        ]
        store.upsert_document(
            path,
            content_hash="hash1",
            size=100,
            mtime_ns=1,
            status="ok",
            chunks=chunks,
            vectors=vecs,
        )
        got_chunks, got_matrix = store.get_document_chunks(path)
        assert len(got_chunks) == 2
        assert got_chunks[0].text == "Bagian satu"
        assert got_chunks[1].text == "Bagian dua"
        assert got_chunks[0].index == 0
        assert got_chunks[1].index == 1
        # Vektor identik persis
        np.testing.assert_array_equal(got_matrix[0], vecs[0])
        np.testing.assert_array_equal(got_matrix[1], vecs[1])

    def test_get_document_chunks_order(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        path = tmp_path / "doc2.txt"
        n = 5
        vecs = np.random.rand(n, 384).astype(np.float32)
        for i in range(n):
            vecs[i] /= np.linalg.norm(vecs[i])
        chunks = [Chunk(f"Teks {i}", path, None, i) for i in range(n)]
        store.upsert_document(
            path,
            content_hash="h2",
            size=50,
            mtime_ns=1,
            status="ok",
            chunks=chunks,
            vectors=vecs,
        )
        got_chunks, _got_matrix = store.get_document_chunks(path)
        for i, c in enumerate(got_chunks):
            assert c.index == i

    def test_get_document_chunks_not_found(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        chunks, matrix = store.get_document_chunks(tmp_path / "tidak_ada.txt")
        assert chunks == []
        assert matrix.shape == (0, 384)

    def test_get_document_chunks_source_is_given_path(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        path = tmp_path / "doc3.txt"
        v = np.ones((1, 384), dtype=np.float32)
        v /= np.linalg.norm(v)
        store.upsert_document(
            path,
            content_hash="h3",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=[Chunk("Teks", path, None, 0)],
            vectors=v,
        )
        got_chunks, _ = store.get_document_chunks(path)
        assert got_chunks[0].source == path.resolve()


# ============================================================
# Tests: index_folder
# ============================================================


class TestIndexFolder:
    def test_first_run_indexes_all(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        _write(
            folder / "a.txt",
            "Ini adalah dokumen pertama dengan cukup teks untuk diindeks.",
        )
        _write(
            folder / "b.txt",
            "Ini adalah dokumen kedua dengan lebih banyak teks untuk pengujian.",
        )

        store = _make_store(tmp_path)
        embedder, backend = _make_embedder()

        report = index_folder(folder, store, embedder)

        assert report.indexed == 2
        assert report.files_found == 2
        assert report.chunks_added > 0
        assert backend.embed_count > 0
        assert report.errors == 0
        assert report.cancelled is False

    def test_second_run_all_unchanged(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        _write(folder / "a.txt", "Teks dokumen yang tidak berubah untuk run kedua.")
        _write(folder / "b.txt", "Teks lain yang stabil tanpa perubahan apapun.")

        store = _make_store(tmp_path)
        embedder1, backend1 = _make_embedder()
        index_folder(folder, store, embedder1)
        _ = backend1.embed_count

        # Run kedua
        embedder2, backend2 = _make_embedder()
        report2 = index_folder(folder, store, embedder2)

        assert report2.unchanged == 2
        assert report2.indexed == 0
        assert backend2.embed_count == 0

    def test_changed_file_reindexed(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        fa = folder / "a.txt"
        fb = folder / "b.txt"
        _write(fa, "Konten awal dokumen A yang akan diubah nanti.")
        _write(fb, "Konten dokumen B yang tetap sama dan tidak berubah.")

        store = _make_store(tmp_path)
        embedder1, _ = _make_embedder()
        index_folder(folder, store, embedder1)

        # Ubah isi dan mtime file A
        time.sleep(0.01)
        fa.write_text(
            "Konten baru dokumen A setelah perubahan signifikan.", encoding="utf-8"
        )
        # pastikan mtime berubah
        new_mtime = time.time() + 1
        os.utime(str(fa), (new_mtime, new_mtime))

        embedder2, backend2 = _make_embedder()
        report2 = index_folder(folder, store, embedder2)

        assert report2.indexed == 1
        assert report2.unchanged == 1
        assert backend2.embed_count > 0

    def test_touch_only_reuses(self, tmp_path: Path) -> None:
        """File di-touch (isi sama, mtime baru) -> action reused, embed 0, mtime_ns diperbarui."""
        folder = tmp_path / "docs"
        fa = folder / "a.txt"
        content = "Isi file yang tidak berubah sama sekali untuk test reuse."
        _write(fa, content)

        store = _make_store(tmp_path)
        embedder1, _ = _make_embedder()
        index_folder(folder, store, embedder1)

        old_rec = store.get_document(fa)
        assert old_rec is not None

        # Touch: ubah mtime, isi tetap sama
        new_mtime = time.time() + 2
        os.utime(str(fa), (new_mtime, new_mtime))

        embedder2, backend2 = _make_embedder()
        report2 = index_folder(folder, store, embedder2)

        assert report2.reused == 1
        assert backend2.embed_count == 0

        new_rec = store.get_document(fa)
        assert new_rec is not None
        assert new_rec.mtime_ns != old_rec.mtime_ns

    def test_duplicate_content_reuses(self, tmp_path: Path) -> None:
        """Dua file isi sama: embed hanya sekali, kedua path bisa dicari."""
        folder = tmp_path / "docs"
        same_content = "Konten identik yang sama persis untuk kedua file ini ya."
        fa = folder / "a.txt"
        fb = folder / "b.txt"
        _write(fa, same_content)
        _write(fb, same_content)

        store = _make_store(tmp_path)
        embedder, _backend = _make_embedder()
        report = index_folder(folder, store, embedder)

        assert report.indexed == 1
        assert report.reused == 1
        # Kedua path ada di store
        assert store.get_document(fa) is not None
        assert store.get_document(fb) is not None

    def test_deleted_file_pruned(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        fa = folder / "a.txt"
        fb = folder / "b.txt"
        _write(fa, "Dokumen A yang akan dihapus kemudian.")
        _write(fb, "Dokumen B yang tetap ada dan tidak dihapus.")

        store = _make_store(tmp_path)
        embedder1, _ = _make_embedder()
        index_folder(folder, store, embedder1)

        # Hapus file A
        fa.unlink()

        embedder2, _ = _make_embedder()
        report2 = index_folder(folder, store, embedder2)

        assert report2.removed == 1
        assert store.get_document(fa) is None
        assert store.get_document(fb) is not None

    def test_prune_skipped_when_dir_unreadable_keeps_documents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "docs"
        fa = folder / "ok" / "a.txt"
        fb = folder / "terkunci" / "b.txt"
        _write(fa, "Dokumen a yang tetap terbaca untuk diuji prune.")
        _write(fb, "Dokumen b di folder yang tidak terbaca.")

        store = _make_store(tmp_path)
        embedder1, _ = _make_embedder()
        index_folder(folder, store, embedder1)
        assert store.get_document(fb) is not None

        original_scandir = os.scandir

        def _mock_scandir(path: str | os.PathLike[str]) -> Any:
            if os.path.basename(os.fspath(path)) == "terkunci":
                raise PermissionError("akses ditolak")
            return original_scandir(path)

        monkeypatch.setattr(os, "scandir", _mock_scandir)
        embedder2, _ = _make_embedder()
        report = index_folder(folder, store, embedder2)

        assert report.prune_skipped_reason is not None
        assert report.removed == 0
        assert store.get_document(fb) is not None
        hits = store.search(embedder2.embed_query("dokumen b"))
        assert any(hit.path.resolve() == fb.resolve() for hit in hits)

    def test_prune_no_touch_outside_root(self, tmp_path: Path) -> None:
        """Dokumen di luar root tidak tersentuh saat prune."""
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        fa = root1 / "a.txt"
        fkeep = root1 / "keep.txt"
        fb = root2 / "b.txt"
        _write(fa, "Dokumen root satu untuk pengujian multi-root store.")
        _write(fkeep, "Dokumen penjaga di root satu.")
        _write(fb, "Dokumen root dua yang harus tidak tersentuh saat prune root1.")

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()

        # Indeks keduanya ke store yang sama
        index_folder(root1, store, embedder)
        index_folder(root2, store, embedder)

        # Hapus fa tapi prune hanya untuk root1
        fa.unlink()
        embedder2, _ = _make_embedder()
        report = index_folder(root1, store, embedder2)

        # fa harus di-prune
        assert report.removed == 1
        assert store.get_document(fa) is None
        # fb harus tetap ada (di luar root1)
        assert store.get_document(fb) is not None
        # fkeep tetap ada
        assert store.get_document(fkeep) is not None

    def test_prune_skipped_when_cancelled(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        _write(folder / "a.txt", "Dokumen untuk pengujian pembatalan prune.")
        _write(folder / "b.txt", "Dokumen kedua untuk pembatalan prune.")

        store = _make_store(tmp_path)
        embedder1, _ = _make_embedder()
        index_folder(folder, store, embedder1)

        # Hapus file pertama
        (folder / "a.txt").unlink()

        # Batal langsung sebelum file pertama diproses
        cancelled = True
        embedder2, _ = _make_embedder()
        report = index_folder(folder, store, embedder2, should_cancel=lambda: cancelled)

        assert report.cancelled is True
        assert report.prune_skipped_reason is not None
        # a.txt masih ada di store karena prune dilewati
        assert store.get_document(folder / "a.txt") is not None

    def test_prune_skipped_zero_files(self, tmp_path: Path) -> None:
        """Prune dilewati jika pemindaian menemukan 0 file tapi store punya dokumen."""
        folder = tmp_path / "docs"
        fa = folder / "a.pdf"  # hanya pdf tapi folder sebenarnya kosong
        _write(fa, "PDF palsu untuk test.")

        store = _make_store(tmp_path)
        _embedder1, _ = _make_embedder()
        # Indeks manual
        v = np.ones((1, 384), dtype=np.float32)
        v /= np.linalg.norm(v)
        store.upsert_document(
            fa,
            content_hash="x",
            size=10,
            mtime_ns=1,
            status="ok",
            chunks=[Chunk("Teks", fa, None, 0)],
            vectors=v,
        )

        # Scan folder yang berbeda (kosong)
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()
        _embedder2, _ = _make_embedder()
        # index_folder pada folder kosong tapi store punya dokumen di bawah folder lain
        # -> test via scan_errors atau zero files
        # Buat folder yang isinya hanya file non-didukung
        only_csv = tmp_path / "only_csv"
        only_csv.mkdir()
        _write(only_csv / "data.csv", "a,b,c")
        # Buat dokumen di store yang mengacu ke only_csv
        fake_doc = only_csv / "doc.txt"
        store.upsert_document(
            fake_doc,
            content_hash="y",
            size=5,
            mtime_ns=1,
            status="ok",
            chunks=[Chunk("Ini teks", fake_doc, None, 0)],
            vectors=v,
        )

        embedder3, _ = _make_embedder()
        report = index_folder(only_csv, store, embedder3)
        assert report.files_found == 0
        assert report.prune_skipped_reason is not None
        # Dokumen di store yang mengacu ke only_csv masih ada
        assert store.get_document(fake_doc) is not None

    def test_error_file_retried_next_run(self, tmp_path: Path) -> None:
        """File error di run pertama diproses ulang di run berikutnya."""
        folder = tmp_path / "docs"
        fa = folder / "broken.txt"
        _write(fa, "Konten yang akan dipakai pada run kedua setelah diperbaiki.")

        store = _make_store(tmp_path)
        # Run pertama: simulasi error dengan patch parse_file
        embedder1, _ = _make_embedder()
        with patch(
            "lokalcari.indexer.parse_file", side_effect=Exception("Error buatan")
        ):
            report1 = index_folder(folder, store, embedder1)

        assert report1.errors == 1
        rec = store.get_document(fa)
        assert rec is not None and rec.status == "error"

        # Run kedua: tanpa patch, harus berhasil
        embedder2, _ = _make_embedder()
        report2 = index_folder(folder, store, embedder2)
        assert report2.indexed == 1
        rec2 = store.get_document(fa)
        assert rec2 is not None and rec2.status == "ok"

    def test_empty_file_status(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        _write(folder / "empty.txt", "")

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()
        report = index_folder(folder, store, embedder)

        assert report.empty == 1
        rec = store.get_document(folder / "empty.txt")
        assert rec is not None and rec.status == "empty"

    def test_max_file_bytes_error(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        fa = folder / "big.txt"
        _write(fa, "Teks konten normal untuk pengujian batas ukuran file.")

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()
        report = index_folder(folder, store, embedder, max_file_bytes=5)

        assert report.errors == 1
        rec = store.get_document(fa)
        assert rec is not None and rec.status == "error"
        assert "terlalu besar" in (rec.error or "").lower()

    def test_model_not_installed_propagates(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        _write(
            folder / "a.txt",
            "Teks untuk pengujian propagasi error model tidak terpasang.",
        )

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()

        # Simpan record sebelumnya
        v = np.ones((1, 384), dtype=np.float32)
        v /= np.linalg.norm(v)
        store.upsert_document(
            folder / "pre.txt",
            content_hash="pre",
            size=5,
            mtime_ns=1,
            status="ok",
            chunks=[Chunk("Pre", folder / "pre.txt", None, 0)],
            vectors=v,
        )

        def _raise_model_not_installed(
            texts: list[str], batch_size: int = 32
        ) -> np.ndarray:
            raise ModelNotInstalled("Model belum terpasang")

        embedder.embed_passages = _raise_model_not_installed  # type: ignore

        with pytest.raises(ModelNotInstalled):
            index_folder(folder, store, embedder)

        # Record pre.txt harus tetap ada
        assert store.get_document(folder / "pre.txt") is not None
        # a.txt tidak boleh tercatat sebagai error
        rec = store.get_document(folder / "a.txt")
        assert rec is None or rec.status not in ("error",)

    def test_embedding_error_propagates(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        _write(
            folder / "a.txt",
            "Teks untuk pengujian propagasi EmbeddingError dari embedder.",
        )

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()

        def _raise_embedding_error(
            texts: list[str], batch_size: int = 32
        ) -> np.ndarray:
            raise EmbeddingError("Gagal embed")

        embedder.embed_passages = _raise_embedding_error  # type: ignore

        with pytest.raises(EmbeddingError):
            index_folder(folder, store, embedder)

        # a.txt tidak boleh tercatat sebagai error biasa
        rec = store.get_document(folder / "a.txt")
        assert rec is None or rec.status != "error"

    def test_cancellation_mid_run(self, tmp_path: Path) -> None:
        """Pembatalan di tengah run: cancelled True, tidak ada prune, run berikutnya lanjut."""
        folder = tmp_path / "docs"
        for i in range(5):
            _write(
                folder / f"doc{i}.txt",
                f"Ini adalah konten dokumen nomor {i} untuk pengujian pembatalan.",
            )

        store = _make_store(tmp_path)
        embedder1, _ = _make_embedder()

        seen_indexed = [0]

        def _on_progress(event: ProgressEvent) -> None:
            if event.phase == "index" and event.action == "indexed":
                seen_indexed[0] += 1

        def _should_cancel() -> bool:
            return seen_indexed[0] >= 2

        report = index_folder(
            folder,
            store,
            embedder1,
            on_progress=_on_progress,
            should_cancel=_should_cancel,
        )

        assert report.cancelled is True
        assert report.prune_skipped_reason is not None

        # Run berikutnya harus menyelesaikan sisanya
        embedder3, _backend3 = _make_embedder()
        report3 = index_folder(folder, store, embedder3)
        assert report3.cancelled is False
        total_indexed = sum(1 for d in store.list_documents() if d.status == "ok")
        assert total_indexed == 5

    def test_cancellation_no_partial_document(self, tmp_path: Path) -> None:
        """Pembatalan di tengah dokumen besar: tidak ada dokumen setengah jadi."""
        folder = tmp_path / "docs"
        # Buat file dengan cukup teks untuk > 64 chunk dengan max_chars kecil
        big_content = " ".join(
            [
                f"Kata nomor {i} dalam dokumen besar yang sangat panjang sekali."
                for i in range(500)
            ]
        )
        _write(folder / "big.txt", big_content)

        store = _make_store(tmp_path)
        embedder, _backend = _make_embedder()

        batch_calls = [0]
        original_embed = embedder.embed_passages

        def _cancel_mid_embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
            batch_calls[0] += 1
            return original_embed(texts, batch_size=batch_size)

        embedder.embed_passages = _cancel_mid_embed  # type: ignore

        def _should_cancel() -> bool:
            return batch_calls[0] >= 1

        report = index_folder(
            folder,
            store,
            embedder,
            max_chars=50,
            overlap=10,
            should_cancel=_should_cancel,
        )

        # big.txt tidak boleh tersimpan karena dibatalkan di tengah
        assert report.cancelled is True
        rec = store.get_document(folder / "big.txt")
        assert rec is None

    def test_progress_events_correct(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        for i in range(3):
            _write(
                folder / f"doc{i}.txt",
                f"Konten dokumen {i} untuk pengujian progress event.",
            )

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()

        events: list[ProgressEvent] = []

        def _on_progress(e: ProgressEvent) -> None:
            events.append(e)

        index_folder(folder, store, embedder, on_progress=_on_progress)

        scan_events = [e for e in events if e.phase == "scan"]
        index_events = [e for e in events if e.phase == "index"]

        assert len(scan_events) >= 1
        assert scan_events[0].total == 3

        currents = [e.current for e in index_events]
        assert currents == sorted(currents)
        assert max(currents) == 3
        assert min(currents) >= 1
        # Tidak ada lompatan
        for a, b in itertools.pairwise(currents):
            assert b - a <= 1

    def test_force_reindexes_all(self, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        _write(
            folder / "a.txt",
            "Konten yang sudah ada sebelumnya di store untuk force test.",
        )

        store = _make_store(tmp_path)
        embedder1, backend1 = _make_embedder()
        index_folder(folder, store, embedder1)
        _ = backend1.embed_count

        # Force run: harus embed ulang
        embedder2, backend2 = _make_embedder()
        report2 = index_folder(folder, store, embedder2, force=True)

        assert report2.indexed == 1
        assert backend2.embed_count > 0

    def test_file_changed_during_indexing_skipped(self, tmp_path: Path) -> None:
        """File yang berubah saat diindeks tidak disimpan dan masuk problems."""
        folder = tmp_path / "docs"
        fa = folder / "a.txt"
        _write(fa, "Konten awal yang akan diubah saat proses indexing sedang berjalan.")

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()

        original_parse = __import__(
            "lokalcari.indexer", fromlist=["parse_file"]
        ).parse_file

        def _parse_and_modify(path: Path) -> Any:
            result = original_parse(path)
            # Ubah file saat parsing selesai (simulasi perubahan saat diindeks)
            fa.write_text("Konten berubah setelah parsing selesai.", encoding="utf-8")
            new_mtime = time.time() + 10
            os.utime(str(fa), (new_mtime, new_mtime))
            return result

        with patch("lokalcari.indexer.parse_file", side_effect=_parse_and_modify):
            report = index_folder(folder, store, embedder)

        assert report.skipped == 1
        assert len(report.problems) == 1
        # File tidak boleh tersimpan sebagai ok
        rec = store.get_document(fa)
        assert rec is None or rec.status != "ok"

    def test_pdf_no_text_possible_scans(self, tmp_path: Path) -> None:
        """PDF tanpa teks -> status empty + masuk possible_scans."""
        from lokalcari.parsers.models import ParsedDocument

        folder = tmp_path / "docs"
        fa = folder / "scan.pdf"
        fa.parent.mkdir(parents=True, exist_ok=True)
        fa.write_bytes(b"fake pdf content")

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()

        # Mock parse_file mengembalikan dokumen tanpa segment
        def _mock_parse(path: Path) -> ParsedDocument:
            return ParsedDocument(path=path, segments=[])

        with patch("lokalcari.indexer.parse_file", side_effect=_mock_parse):
            report = index_folder(folder, store, embedder)

        assert report.empty >= 1
        assert fa.resolve() in report.possible_scans

    def test_other_files_indexed_despite_one_error(self, tmp_path: Path) -> None:
        """File error tidak menghentikan pengindeksan file lain."""
        folder = tmp_path / "docs"
        _write(
            folder / "good.txt", "Ini dokumen yang baik dan harus berhasil diindeks."
        )
        _write(folder / "bad.txt", "Ini dokumen yang akan gagal saat parsing.")

        store = _make_store(tmp_path)
        embedder, _ = _make_embedder()

        original_parse = __import__(
            "lokalcari.indexer", fromlist=["parse_file"]
        ).parse_file

        def _selective_error(path: Path) -> Any:
            if path.name == "bad.txt":
                from lokalcari.parsers.models import ParseError

                raise ParseError(path, "Error buatan untuk bad.txt")
            return original_parse(path)

        with patch("lokalcari.indexer.parse_file", side_effect=_selective_error):
            report = index_folder(folder, store, embedder)

        assert report.indexed >= 1
        assert report.errors >= 1
        assert store.get_document(folder / "good.txt") is not None

    def test_vectors_match_fake_embedder(self, tmp_path: Path) -> None:
        """Vektor yang tersimpan harus cocok dengan yang dihasilkan fake backend."""
        folder = tmp_path / "docs"
        fa = folder / "a.txt"
        content = "Dokumen dengan konten yang cukup panjang untuk membuat satu chunk besar ya."
        _write(fa, content)

        store = _make_store(tmp_path)
        embedder, _backend = _make_embedder()
        index_folder(folder, store, embedder)

        # Ambil chunk dari store
        got_chunks, got_matrix = store.get_document_chunks(fa)
        assert len(got_chunks) > 0

        # Hitung ulang vektor dengan fake backend untuk perbandingan
        backend2 = _FakeBackend(dim=384)
        expected_vecs = backend2.embed([f"passage: {c.text}" for c in got_chunks])

        for i, expected in enumerate(expected_vecs):
            np.testing.assert_array_almost_equal(got_matrix[i], expected, decimal=5)


# ============================================================
# Bukti sensitivitas: jalankan via pytest langsung
# (Mutasi M1-M4 diuji manual sesuai instruksi task)
# ============================================================
