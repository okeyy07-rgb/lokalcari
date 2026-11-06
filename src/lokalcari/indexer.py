"""Indexer: pemindai folder dan pengindeks dokumen inkremental."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lokalcari.chunker import Chunk, chunk_document
from lokalcari.embedder import Embedder, EmbeddingError, ModelNotInstalled
from lokalcari.parsers import parse_file, supported_extensions
from lokalcari.store import Store, StoreError

DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "$RECYCLE.BIN",
        "System Volume Information",
    }
)

_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


@dataclass
class ProgressEvent:
    phase: str  # "scan" | "index" | "prune"
    current: int
    total: int  # untuk phase "index": jumlah file hasil pemindaian
    path: Path | None
    action: (
        str | None
    )  # "indexed" | "reused" | "unchanged" | "empty" | "error" | "skipped" | "removed"


@dataclass
class FileProblem:
    path: Path
    message: str


@dataclass
class IndexReport:
    root: Path
    files_found: int
    indexed: int
    reused: int
    unchanged: int
    empty: int
    errors: int
    skipped: int
    removed: int
    chunks_added: int
    cancelled: bool
    prune_skipped_reason: str | None
    problems: list[FileProblem]
    possible_scans: list[Path]
    duration_s: float


class IndexerError(Exception):
    """Exception untuk kesalahan konfigurasi atau I/O di indexer."""


def _is_reparse_point(st: os.stat_result) -> bool:
    """Periksa apakah entri adalah reparse point (junction) Windows."""
    file_attrs = getattr(st, "st_file_attributes", None)
    if file_attrs is None:
        return False
    return bool(file_attrs & _FILE_ATTRIBUTE_REPARSE_POINT)


def scan_folder(
    root: Path | str,
    *,
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> list[Path]:
    """Memindai folder dan mengembalikan daftar path file yang didukung, terurut.

    Hanya ekstensi yang didukung oleh parsers.supported_extensions().
    File dan direktori tersembunyi, symlink, dan direktori yang diabaikan dilewati.
    root tidak ada atau bukan direktori: lempar IndexerError.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise IndexerError(f"Root folder tidak ditemukan: '{root}'")
    if not root.is_dir():
        raise IndexerError(f"Root bukan direktori: '{root}'")

    exts = supported_extensions()
    result: list[Path] = []
    pending_dirs = [root]

    while pending_dirs:
        dirpath = pending_dirs.pop()

        def _walk_error(error: OSError, current_dir: Path = dirpath) -> None:
            error_path = (
                Path(error.filename) if error.filename is not None else current_dir
            )
            if on_error is not None:
                on_error(error_path, error)

        # Walk satu tingkat saja supaya callback tahu direktori saat ini jika filename None.
        for dirpath_str, dirnames, filenames in os.walk(
            str(dirpath), followlinks=False, onerror=_walk_error
        ):
            current_path = Path(dirpath_str)
            filtered: list[str] = []
            for dirname in dirnames:
                if dirname.startswith(".") or dirname in ignore_dirs:
                    continue

                child_path = current_path / dirname
                try:
                    child_stat = os.lstat(str(child_path))
                except OSError as e:
                    if on_error is not None:
                        on_error(child_path, e)
                    continue
                if stat.S_ISLNK(child_stat.st_mode) or _is_reparse_point(child_stat):
                    continue
                filtered.append(dirname)

            pending_dirs.extend(
                current_path / name for name in sorted(filtered, reverse=True)
            )
            dirnames.clear()

            for filename in sorted(filenames):
                if filename.startswith((".", "~$", ".~lock.")):
                    continue

                file_path = current_path / filename
                if Path(filename).suffix.lower() not in exts:
                    continue

                try:
                    file_stat = os.lstat(str(file_path))
                except OSError as e:
                    if on_error is not None:
                        on_error(file_path, e)
                    continue
                if stat.S_ISLNK(file_stat.st_mode):
                    continue

                result.append(file_path.resolve())

    return sorted(result)


def _hash_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Menghitung SHA-256 isi file secara bertahap."""
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def index_folder(
    root: Path | str,
    store: Store,
    embedder: Embedder,
    *,
    max_chars: int = 1000,
    overlap: int = 150,
    force: bool = False,
    prune: bool = True,
    max_file_bytes: int = 100 * 1024 * 1024,
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> IndexReport:
    """Mengindeks semua dokumen di folder secara inkremental.

    File baru atau berubah diparse, di-chunk, dan di-embed.
    File tidak berubah dilewati (unchanged).
    File dengan isi sama di-reuse chunk-nya tanpa embedding ulang.
    Dokumen yang sudah dihapus dari disk dibersihkan dari store (prune).
    """
    t_start = time.monotonic()
    root = Path(root).resolve()

    # === Pemindaian ===
    scan_errors: list[tuple[Path, Exception]] = []

    def _on_scan_error(p: Path, e: Exception) -> None:
        scan_errors.append((p, e))

    files = scan_folder(root, ignore_dirs=ignore_dirs, on_error=_on_scan_error)

    files_found = len(files)
    if on_progress is not None:
        on_progress(
            ProgressEvent(
                phase="scan",
                current=files_found,
                total=files_found,
                path=None,
                action=None,
            )
        )

    # === Pengindeksan ===
    indexed = 0
    reused = 0
    unchanged = 0
    empty = 0
    errors = 0
    skipped = 0
    chunks_added = 0
    cancelled = False
    problems: list[FileProblem] = []
    possible_scans: list[Path] = []

    for file_idx, fpath in enumerate(files):
        # Cek pembatalan sebelum setiap file
        if should_cancel is not None and should_cancel():
            cancelled = True
            break

        def _emit(act: str, _idx: int = file_idx, _path: Path = fpath) -> None:
            if on_progress is not None:
                on_progress(
                    ProgressEvent(
                        phase="index",
                        current=_idx + 1,
                        total=files_found,
                        path=_path,
                        action=act,
                    )
                )

        # --- Langkah 1: stat file ---
        try:
            st0 = os.stat(str(fpath))
        except OSError:
            skipped += 1
            _emit("skipped")
            continue

        size0 = st0.st_size
        mtime_ns0 = st0.st_mtime_ns

        # --- Langkah 2: terlalu besar ---
        if size0 > max_file_bytes:
            errors += 1
            store.upsert_document(
                fpath,
                content_hash="",
                size=size0,
                mtime_ns=mtime_ns0,
                status="error",
                error=f"File terlalu besar ({size0} bytes, batas {max_file_bytes} bytes)",
            )
            _emit("error")
            continue

        # --- Langkah 3: unchanged (hanya jika bukan force) ---
        if not force:
            existing = store.get_document(fpath)
            if (
                existing is not None
                and existing.size == size0
                and existing.mtime_ns == mtime_ns0
                and existing.status in ("ok", "empty")
            ):
                unchanged += 1
                _emit("unchanged")
                continue

        # --- Langkah 4: file kosong ---
        if size0 == 0:
            empty += 1
            store.upsert_document(
                fpath,
                content_hash="",
                size=0,
                mtime_ns=mtime_ns0,
                status="empty",
                error="File kosong",
            )
            _emit("empty")
            continue

        # --- Langkah 5: hitung SHA-256 ---
        try:
            content_hash = _hash_file(fpath)
        except OSError as e:
            errors += 1
            store.upsert_document(
                fpath,
                content_hash="",
                size=size0,
                mtime_ns=mtime_ns0,
                status="error",
                error=str(e)[:500],
            )
            _emit("error")
            continue

        # --- Langkah 6: reuse via hash (jika bukan force) ---
        if not force:
            matching = store.find_documents_by_hash(content_hash)
            # Filter: hanya reuse dari path LAIN atau path yang sama
            # (jika path yang sama, berarti hanya mtime berubah — tetap reuse)
            if matching:
                donor = matching[0]
                old_chunks, old_matrix = store.get_document_chunks(donor.path)
                # Buat ulang chunk dengan source = path file ini
                new_chunks = [
                    Chunk(text=c.text, source=fpath, page=c.page, index=c.index)
                    for c in old_chunks
                ]
                reused_status = donor.status
                reused_error = donor.error
                store.upsert_document(
                    fpath,
                    content_hash=content_hash,
                    size=size0,
                    mtime_ns=mtime_ns0,
                    status=reused_status,
                    error=reused_error,
                    chunks=new_chunks,
                    vectors=old_matrix if len(new_chunks) > 0 else None,
                )
                reused += 1
                if reused_status == "empty":
                    empty += 1
                else:
                    chunks_added += len(new_chunks)
                _emit("reused")
                continue

        # --- Langkah 7: parse + chunk + embed ---
        # Parse
        try:
            parsed_doc = parse_file(fpath)
            doc_chunks = chunk_document(
                parsed_doc, max_chars=max_chars, overlap=overlap
            )
        except (ModelNotInstalled, EmbeddingError, StoreError):
            raise
        except Exception as e:  # noqa: BLE001
            errors += 1
            store.upsert_document(
                fpath,
                content_hash=content_hash,
                size=size0,
                mtime_ns=mtime_ns0,
                status="error",
                error=str(e)[:500],
            )
            _emit("error")
            continue

        # Tanpa chunk -> empty
        if not doc_chunks:
            is_pdf = fpath.suffix.lower() == ".pdf"
            if is_pdf and len(parsed_doc.segments) == 0:
                msg = "Tidak ada teks yang bisa diekstrak (kemungkinan PDF hasil scan, butuh OCR)"
                possible_scans.append(fpath)
            else:
                msg = "Dokumen tidak menghasilkan chunk"

            empty += 1
            store.upsert_document(
                fpath,
                content_hash=content_hash,
                size=size0,
                mtime_ns=mtime_ns0,
                status="empty",
                error=msg,
            )
            _emit("empty")
            continue

        # Embed dalam potongan 64 teks
        BATCH = 64
        texts = [c.text for c in doc_chunks]
        all_vectors: list[np.ndarray] = []
        aborted = False

        for batch_start in range(0, len(texts), BATCH):
            # Cek pembatalan di antara potongan
            if should_cancel is not None and should_cancel():
                cancelled = True
                aborted = True
                break
            batch_texts = texts[batch_start : batch_start + BATCH]
            # Biarkan ModelNotInstalled, EmbeddingError, StoreError propagate ke atas
            vecs = embedder.embed_passages(batch_texts)
            all_vectors.append(vecs)

        if aborted:
            # Jangan simpan dokumen yang belum selesai
            skipped += 1
            _emit("skipped")
            break

        vectors = np.vstack(all_vectors).astype(np.float32)

        # --- Langkah 8: stat ulang setelah embed ---
        try:
            st1 = os.stat(str(fpath))
        except OSError:
            skipped += 1
            problems.append(
                FileProblem(path=fpath, message="File hilang setelah diindeks")
            )
            _emit("skipped")
            continue

        if st1.st_size != size0 or st1.st_mtime_ns != mtime_ns0:
            skipped += 1
            problems.append(
                FileProblem(path=fpath, message="File berubah saat diindeks, dilewati")
            )
            _emit("skipped")
            continue

        # --- Simpan ---
        store.upsert_document(
            fpath,
            content_hash=content_hash,
            size=size0,
            mtime_ns=mtime_ns0,
            status="ok",
            chunks=doc_chunks,
            vectors=vectors,
        )
        indexed += 1
        chunks_added += len(doc_chunks)
        _emit("indexed")

    # Emit progress untuk file yang tersisa jika dibatalkan
    if cancelled and on_progress is not None:
        # Sudah emit via _emit("skipped") untuk file terakhir yang aborted
        pass

    # === Pembersihan (prune) ===
    removed = 0
    prune_skipped_reason: str | None = None

    if not prune:
        pass
    elif cancelled:
        prune_skipped_reason = "Proses dibatalkan; pembersihan dilewati"
    elif scan_errors:
        prune_skipped_reason = f"Ada {len(scan_errors)} direktori/file yang gagal dibaca; pembersihan dilewati"
    else:
        # Periksa apakah ada dokumen di bawah root di store
        all_docs = store.list_documents()
        docs_under_root = [d for d in all_docs if _path_is_under(d.path, root)]

        file_set = set(files)

        if files_found == 0 and docs_under_root:
            prune_skipped_reason = (
                "Pemindaian menemukan 0 file sementara store punya dokumen di bawah root; "
                "pembersihan dilewati"
            )
        else:
            for doc in docs_under_root:
                if doc.path not in file_set:
                    store.remove_document(doc.path)
                    removed += 1
                    if on_progress is not None:
                        on_progress(
                            ProgressEvent(
                                phase="prune",
                                current=removed,
                                total=removed,
                                path=doc.path,
                                action="removed",
                            )
                        )

    duration_s = time.monotonic() - t_start

    return IndexReport(
        root=root,
        files_found=files_found,
        indexed=indexed,
        reused=reused,
        unchanged=unchanged,
        empty=empty,
        errors=errors,
        skipped=skipped,
        removed=removed,
        chunks_added=chunks_added,
        cancelled=cancelled,
        prune_skipped_reason=prune_skipped_reason,
        problems=problems,
        possible_scans=possible_scans,
        duration_s=duration_s,
    )


def _path_is_under(path: Path, root: Path) -> bool:
    """Periksa apakah path berada di bawah root (case-insensitive di Windows)."""
    try:
        norm_path = os.path.normcase(str(path))
        norm_root = os.path.normcase(str(root))
        common = os.path.commonpath([norm_path, norm_root])
        return common == norm_root
    except ValueError:
        return False
