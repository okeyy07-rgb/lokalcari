import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

from lokalcari.chunker import Chunk

SCHEMA_VERSION = "1"
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_EMBEDDING_DIM = 384


class StoreError(Exception):
    """Exception umum untuk kesalahan pada Store."""


class StoreModelMismatch(StoreError):
    """Exception ketika model_name atau embedding_dim pada database berbeda."""


def default_db_path() -> Path:
    """Mengembalikan path default untuk file database SQLite index."""
    env_dir = os.environ.get("LOKALCARI_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.suffix == ".db":
            return p
        return p / "index.db"
    return Path.home() / ".lokalcari" / "index.db"


@dataclass
class DocumentRecord:
    """Representasi rekaman dokumen dalam store."""

    path: Path
    content_hash: str
    size: int
    mtime_ns: int
    status: str
    error: str | None
    chunk_count: int
    indexed_at: str


@dataclass
class SearchHit:
    """Hasil pencarian vektor chunk terdekat."""

    path: Path
    page: int | None
    chunk_index: int
    text: str
    score: float


class Store:
    """Penyimpanan dokumen, chunk, dan vektor ke file SQLite lokal.

    Satu objek Store hanya boleh dipakai dari satu thread; untuk thread lain, buka objek Store sendiri.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        timeout: float = 5.0,
    ) -> None:
        if db_path is None:
            self.db_path = default_db_path()
        else:
            self.db_path = Path(db_path)

        self.model_name = str(model_name)
        self.embedding_dim = int(embedding_dim)
        self.timeout = float(timeout)

        self._conn: sqlite3.Connection | None = None
        self._cached_matrix: np.ndarray | None = None
        self._cached_chunk_ids: list[int] | None = None
        self._cached_data_version: int | None = None

        self._connect()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Menutup koneksi SQLite secara rapi."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
        self._invalidate_vector_cache()

    def _ensure_open(self) -> sqlite3.Connection:
        """Memastikan koneksi database terbuka."""
        if self._conn is None:
            raise StoreError("Koneksi Store sudah ditutup.")
        return self._conn

    def _invalidate_vector_cache(self) -> None:
        """Menghapus cache matriks vektor dan ID chunk dalam memori."""
        self._cached_matrix = None
        self._cached_chunk_ids = None
        self._cached_data_version = None

    @staticmethod
    def _handle_sqlite_error(e: Exception, default_action: str) -> StoreError:
        """Membungkus sqlite3.Error menjadi StoreError dengan pesan deskriptif."""
        err_msg = str(e).lower()
        if "database is locked" in err_msg or "locked" in err_msg or "busy" in err_msg:
            return StoreError(f"Database sedang digunakan oleh proses lain: {e}")
        return StoreError(f"{default_action}: {e}")

    def _connect(self) -> None:
        """Membuka koneksi SQLite, mengatur PRAGMA, dan inisialisasi skema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._init_schema_and_meta()
        except StoreModelMismatch:
            self.close()
            raise
        except (sqlite3.DatabaseError, sqlite3.OperationalError, sqlite3.Error) as e:
            self.close()
            err_msg = str(e).lower()
            if (
                "database is locked" in err_msg
                or "locked" in err_msg
                or "busy" in err_msg
            ):
                raise StoreError(
                    f"Database sedang digunakan oleh proses lain: {e}"
                ) from e
            raise StoreError(f"Database SQLite rusak atau tidak valid: {e}") from e
        except Exception:
            self.close()
            raise

    def _init_schema_and_meta(self) -> None:
        """Membuat tabel jika belum ada dan memverifikasi metadata model."""
        conn = self._ensure_open()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            content_hash TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('ok', 'empty', 'error')),
            error TEXT,
            chunk_count INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page INTEGER,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
        """)

        cur.execute("SELECT key, value FROM meta;")
        meta = dict(cur.fetchall())

        if not meta:
            with conn:
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?);",
                    (SCHEMA_VERSION,),
                )
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('model_name', ?);",
                    (self.model_name,),
                )
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('embedding_dim', ?);",
                    (str(self.embedding_dim),),
                )
        else:
            stored_schema = meta.get("schema_version")
            if stored_schema != SCHEMA_VERSION:
                raise StoreError(
                    f"Versi skema database tidak didukung: '{stored_schema}', diharapkan '{SCHEMA_VERSION}'."
                )

            stored_model = meta.get("model_name")
            stored_dim_str = meta.get("embedding_dim")
            stored_dim = int(stored_dim_str) if stored_dim_str is not None else None

            if stored_model != self.model_name:
                raise StoreModelMismatch(
                    f"Model embedding tidak cocok: database menggunakan '{stored_model}', "
                    f"tetapi aplikasi meminta '{self.model_name}'."
                )
            if stored_dim != self.embedding_dim:
                raise StoreModelMismatch(
                    f"Dimensi embedding tidak cocok: database menggunakan {stored_dim}, "
                    f"tetapi aplikasi meminta {self.embedding_dim}."
                )

    def upsert_document(
        self,
        path: Path | str,
        *,
        content_hash: str,
        size: int,
        mtime_ns: int,
        status: str,
        error: str | None = None,
        chunks: Sequence[Chunk] = (),
        vectors: np.ndarray | None = None,
    ) -> None:
        """Menyisipkan atau memperbarui dokumen beserta chunk-nya dalam satu transaksi atomik."""
        conn = self._ensure_open()

        if status not in ("ok", "empty", "error"):
            raise ValueError(
                f"Status dokumen tidak valid: '{status}'. Harus 'ok', 'empty', atau 'error'."
            )

        num_chunks = len(chunks)

        if num_chunks == 0:
            if vectors is not None and len(vectors) != 0:
                raise ValueError(
                    f"Jumlah vektor ({len(vectors)}) tidak sama dengan jumlah chunk (0)."
                )
        else:
            if vectors is None:
                raise ValueError("Vektor wajib diberikan ketika chunks tidak kosong.")
            if not isinstance(vectors, np.ndarray):
                raise ValueError("Vektor harus berupa numpy ndarray.")
            if vectors.dtype != np.float32:
                raise ValueError(
                    f"Dtype vektor harus float32, diterima: {vectors.dtype}."
                )
            if vectors.ndim != 2:
                raise ValueError(
                    f"Vektor harus berupa array 2D (jumlah_chunk, dim), diterima ndim: {vectors.ndim}."
                )
            if vectors.shape[0] != num_chunks:
                raise ValueError(
                    f"Jumlah vektor ({vectors.shape[0]}) tidak sama dengan jumlah chunk ({num_chunks})."
                )
            if vectors.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Dimensi vektor ({vectors.shape[1]}) tidak sesuai dengan embedding_dim ({self.embedding_dim})."
                )

        resolved_path = str(Path(path).resolve())
        indexed_at = datetime.now(UTC).isoformat()

        cur = conn.cursor()
        try:
            with conn:
                cur.execute("DELETE FROM documents WHERE path = ?;", (resolved_path,))
                cur.execute(
                    """
                    INSERT INTO documents (path, content_hash, size, mtime_ns, status, error, chunk_count, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        resolved_path,
                        content_hash,
                        size,
                        mtime_ns,
                        status,
                        error,
                        num_chunks,
                        indexed_at,
                    ),
                )
                doc_id = cur.lastrowid
                if num_chunks > 0 and vectors is not None:
                    chunk_rows = [
                        (
                            doc_id,
                            c.index,
                            c.page,
                            c.text,
                            vectors[i].tobytes(),
                        )
                        for i, c in enumerate(chunks)
                    ]
                    cur.executemany(
                        """
                        INSERT INTO chunks (document_id, chunk_index, page, text, embedding)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        chunk_rows,
                    )
            self._invalidate_vector_cache()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            self._invalidate_vector_cache()
            raise self._handle_sqlite_error(e, "Gagal melakukan upsert document") from e
        except Exception:
            self._invalidate_vector_cache()
            raise

    def get_document(self, path: Path | str) -> DocumentRecord | None:
        """Mengambil rekaman dokumen berdasarkan path-nya."""
        conn = self._ensure_open()
        resolved_path = str(Path(path).resolve())
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT path, content_hash, size, mtime_ns, status, error, chunk_count, indexed_at
                FROM documents
                WHERE path = ?;
                """,
                (resolved_path,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return DocumentRecord(
                path=Path(row[0]),
                content_hash=row[1],
                size=row[2],
                mtime_ns=row[3],
                status=row[4],
                error=row[5],
                chunk_count=row[6],
                indexed_at=row[7],
            )
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            raise self._handle_sqlite_error(e, "Gagal mengambil dokumen") from e

    def list_documents(self) -> list[DocumentRecord]:
        """Mengambil daftar semua rekaman dokumen dalam store."""
        conn = self._ensure_open()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT path, content_hash, size, mtime_ns, status, error, chunk_count, indexed_at
                FROM documents
                ORDER BY id ASC;
                """
            )
            rows = cur.fetchall()
            return [
                DocumentRecord(
                    path=Path(r[0]),
                    content_hash=r[1],
                    size=r[2],
                    mtime_ns=r[3],
                    status=r[4],
                    error=r[5],
                    chunk_count=r[6],
                    indexed_at=r[7],
                )
                for r in rows
            ]
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            raise self._handle_sqlite_error(e, "Gagal mengambil daftar dokumen") from e

    def remove_document(self, path: Path | str) -> bool:
        """Menghapus dokumen dan seluruh chunk terkait (via cascade delete)."""
        conn = self._ensure_open()
        resolved_path = str(Path(path).resolve())
        try:
            with conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM documents WHERE path = ?;", (resolved_path,))
                removed = cur.rowcount > 0
            if removed:
                self._invalidate_vector_cache()
            return removed
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            raise self._handle_sqlite_error(e, "Gagal menghapus dokumen") from e

    def _load_vector_cache(self) -> None:
        """Memuat semua vektor dan ID chunk dari database ke memori."""
        conn = self._ensure_open()
        try:
            cur = conn.cursor()
            # Baca data_version paling awal sebelum transaksi dimulai
            cur.execute("PRAGMA data_version;")
            row = cur.fetchone()
            initial_data_version = row[0] if row else 0

            # Jalankan count(*) dan fetch embedding dalam satu transaksi baca eksplisit
            # agar keduanya melihat snapshot yang konsisten.
            conn.execute("BEGIN;")
            try:
                cur.execute("SELECT count(*) FROM chunks;")
                total_chunks = cur.fetchone()[0]

                if total_chunks == 0:
                    self._cached_matrix = np.empty(
                        (0, self.embedding_dim), dtype=np.float32
                    )
                    self._cached_chunk_ids = []
                else:
                    matrix = np.empty(
                        (total_chunks, self.embedding_dim), dtype=np.float32
                    )
                    chunk_ids = [0] * total_chunks

                    cur.execute("SELECT id, embedding FROM chunks ORDER BY id ASC;")
                    for i, (cid, blob) in enumerate(cur):
                        chunk_ids[i] = cid
                        matrix[i] = np.frombuffer(blob, dtype=np.float32)

                    self._cached_matrix = matrix
                    self._cached_chunk_ids = chunk_ids

                self._cached_data_version = initial_data_version
            finally:
                if conn.in_transaction:
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            raise self._handle_sqlite_error(e, "Gagal memuat cache vektor") from e
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"Gagal memuat cache vektor: {e}") from e

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[SearchHit]:
        """Mencari chunk terdekat menggunakan dot product vektor query."""
        if not isinstance(query_vec, np.ndarray):
            raise ValueError("query_vec harus berupa numpy ndarray.")  # noqa: TRY004
        if query_vec.ndim != 1 or query_vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Bentuk query_vec harus 1D ({self.embedding_dim},), diterima bentuk: {query_vec.shape}."
            )
        if query_vec.dtype != np.float32:
            raise ValueError(
                f"Dtype query_vec harus float32, diterima: {query_vec.dtype}."
            )

        if top_k <= 0:
            return []

        conn = self._ensure_open()
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA data_version;")
            row = cur.fetchone()
            current_version = row[0] if row else 0

            if (
                self._cached_matrix is None
                or self._cached_chunk_ids is None
                or self._cached_data_version is None
                or self._cached_data_version != current_version
            ):
                self._load_vector_cache()

            assert self._cached_matrix is not None
            assert self._cached_chunk_ids is not None

            num_chunks = len(self._cached_chunk_ids)
            if num_chunks == 0:
                return []

            scores = self._cached_matrix @ query_vec

            if top_k >= num_chunks:
                ranked_indices = np.argsort(scores)[::-1]
            else:
                partition_indices = np.argpartition(scores, -top_k)[-top_k:]
                sorted_part = np.argsort(scores[partition_indices])[::-1]
                ranked_indices = partition_indices[sorted_part]

            top_chunk_ids = [self._cached_chunk_ids[idx] for idx in ranked_indices]
            top_scores = {
                self._cached_chunk_ids[idx]: float(scores[idx])
                for idx in ranked_indices
            }

            placeholders = ",".join("?" for _ in top_chunk_ids)
            query = f"""
                SELECT c.id, d.path, c.page, c.chunk_index, c.text
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id IN ({placeholders});
            """
            cur.execute(query, top_chunk_ids)
            rows = cur.fetchall()
            row_map = {r[0]: (Path(r[1]), r[2], r[3], r[4]) for r in rows}

            hits: list[SearchHit] = []
            for cid in top_chunk_ids:
                if cid in row_map:
                    path, page, chunk_index, text = row_map[cid]
                    hits.append(
                        SearchHit(
                            path=path,
                            page=page,
                            chunk_index=chunk_index,
                            text=text,
                            score=top_scores[cid],
                        )
                    )
            return hits
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            raise self._handle_sqlite_error(e, "Gagal melakukan pencarian") from e

    def stats(self) -> dict[str, Any]:
        """Mengembalikan statistik dokumen, chunk, dan ukuran file database."""
        conn = self._ensure_open()
        try:
            cur = conn.cursor()
            cur.execute("SELECT status, count(*) FROM documents GROUP BY status;")
            status_counts = {"ok": 0, "empty": 0, "error": 0}
            for status, count in cur.fetchall():
                if status in status_counts:
                    status_counts[status] = count

            cur.execute("SELECT count(*) FROM chunks;")
            chunk_count = cur.fetchone()[0]

            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            wal_path = Path(str(self.db_path) + "-wal")
            if wal_path.exists():
                db_size += wal_path.stat().st_size

            return {
                "documents": status_counts,
                "chunks": chunk_count,
                "db_size_bytes": db_size,
            }
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            raise self._handle_sqlite_error(
                e, "Gagal mengambil statistik database"
            ) from e
