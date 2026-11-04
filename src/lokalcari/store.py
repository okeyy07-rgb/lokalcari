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
    """Penyimpanan dokumen, chunk, dan vektor ke file SQLite lokal."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        if db_path is None:
            self.db_path = default_db_path()
        else:
            self.db_path = Path(db_path)

        self.model_name = str(model_name)
        self.embedding_dim = int(embedding_dim)

        self._conn: sqlite3.Connection | None = None
        self._cached_matrix: np.ndarray | None = None
        self._cached_metadata: list[tuple[Path, int | None, int, str]] | None = None

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
        """Menghapus cache matriks vektor dan metadata dalam memori."""
        self._cached_matrix = None
        self._cached_metadata = None

    def _connect(self) -> None:
        """Membuka koneksi SQLite, mengatur PRAGMA, dan inisialisasi skema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._init_schema_and_meta()
        except StoreModelMismatch:
            self.close()
            raise
        except (sqlite3.DatabaseError, sqlite3.OperationalError, sqlite3.Error) as e:
            self.close()
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
            id INTEGER PRIMARY KEY,
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
            raise StoreError(f"Gagal melakukan upsert document: {e}") from e
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
            raise StoreError(f"Gagal mengambil dokumen: {e}") from e

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
            raise StoreError(f"Gagal mengambil daftar dokumen: {e}") from e

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
            raise StoreError(f"Gagal menghapus dokumen: {e}") from e

    def _load_vector_cache(self) -> None:
        """Memuat semua vektor dan metadata chunk dari database ke memori."""
        conn = self._ensure_open()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT c.page, c.chunk_index, c.text, d.path, c.embedding
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.id ASC;
            """)
            rows = cur.fetchall()
            if not rows:
                self._cached_matrix = np.empty(
                    (0, self.embedding_dim), dtype=np.float32
                )
                self._cached_metadata = []
            else:
                raw_bytes = b"".join(r[4] for r in rows)
                self._cached_matrix = np.frombuffer(
                    raw_bytes, dtype=np.float32
                ).reshape(len(rows), self.embedding_dim)
                self._cached_metadata = [(Path(r[3]), r[0], r[1], r[2]) for r in rows]
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
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

        if self._cached_matrix is None or self._cached_metadata is None:
            self._load_vector_cache()

        assert self._cached_matrix is not None
        assert self._cached_metadata is not None

        num_chunks = len(self._cached_metadata)
        if num_chunks == 0:
            return []

        scores = self._cached_matrix @ query_vec

        if top_k >= num_chunks:
            ranked_indices = np.argsort(scores)[::-1]
        else:
            partition_indices = np.argpartition(scores, -top_k)[-top_k:]
            sorted_part = np.argsort(scores[partition_indices])[::-1]
            ranked_indices = partition_indices[sorted_part]

        hits: list[SearchHit] = []
        for idx in ranked_indices:
            path, page, chunk_index, text = self._cached_metadata[idx]
            hits.append(
                SearchHit(
                    path=path,
                    page=page,
                    chunk_index=chunk_index,
                    text=text,
                    score=float(scores[idx]),
                )
            )
        return hits

    def stats(self) -> dict[str, Any]:
        """Mengembalikan statistik dokumen, chunk, dan ukuran file database."""
        conn = self._ensure_open()
        try:
            cur = conn.cursor()
            cur.execute("SELECT status, count(*) FROM documents GROUP BY status;")
            status_counts = {"ok": 0, "empty": 0, "error": 0}
            total_docs = 0
            for status, count in cur.fetchall():
                if status in status_counts:
                    status_counts[status] = count
                total_docs += count

            cur.execute("SELECT count(*) FROM chunks;")
            chunk_count = cur.fetchone()[0]

            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "documents": status_counts,
                "documents_by_status": status_counts,
                "total_documents": total_docs,
                "chunk_count": chunk_count,
                "total_chunks": chunk_count,
                "chunks": chunk_count,
                "db_size_bytes": db_size,
                "db_size": db_size,
            }
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            raise StoreError(f"Gagal mengambil statistik database: {e}") from e
