import os
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class EmbeddingError(Exception):
    """Exception ketika terjadi kegagalan saat inisialisasi model atau pembuatan embedding."""


class ModelNotInstalled(Exception):
    """Exception ketika model embedding belum diunduh ke cache lokal."""

    def __init__(
        self,
        message: str = (
            "Model embedding belum terpasang. "
            "Silakan jalankan 'lokalcari setup' terlebih dahulu."
        ),
    ):
        super().__init__(message)


def model_cache_dir() -> Path:
    """Mengembalikan direktori cache model embedding."""
    env_dir = os.environ.get("LOKALCARI_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".lokalcari" / "models"


def _ensure_custom_model_registered() -> None:
    """Mendaftarkan model multilingual-e5-small ke registry fastembed bila belum ada."""
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    for m in TextEmbedding.list_supported_models():
        if m["model"] == MODEL_NAME:
            return

    try:
        TextEmbedding.add_custom_model(
            model=MODEL_NAME,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=MODEL_NAME),
            dim=EMBEDDING_DIM,
            model_file="onnx/model.onnx",
        )
    except ValueError:
        pass


def download_model() -> Path:
    """Mengunduh model ke model_cache_dir() dan mengembalikan path-nya.

    SATU-SATUNYA fungsi dalam aplikasi yang mengakses internet untuk mengunduh model.
    """
    cache_dir = model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _ensure_custom_model_registered()

    from fastembed import TextEmbedding

    TextEmbedding(
        MODEL_NAME,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )
    return cache_dir


class Embedder:
    """Penghasil embedding teks menggunakan ONNX embedding model lokal."""

    def __init__(self, backend: Any = None):
        self._backend = backend

    @property
    def dim(self) -> int:
        """Dimensi embedding vektor."""
        return EMBEDDING_DIM

    def _get_backend(self) -> Any:
        """Memuat backend embedding secara malas (lazy loading)."""
        if self._backend is not None:
            return self._backend

        cache_dir = model_cache_dir()
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            raise ModelNotInstalled(
                f"Model embedding belum terpasang di '{cache_dir}'. "
                "Silakan jalankan 'lokalcari setup' terlebih dahulu."
            )

        _ensure_custom_model_registered()

        try:
            from fastembed import TextEmbedding

            self._backend = TextEmbedding(
                MODEL_NAME,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        except ValueError as e:
            if "Could not load model" in str(e):
                raise ModelNotInstalled(
                    f"Model embedding belum lengkap atau belum terpasang di '{cache_dir}'. "
                    "Silakan jalankan 'lokalcari setup' terlebih dahulu."
                ) from e
            raise EmbeddingError(f"Gagal memuat model embedding: {e}") from e
        except (FileNotFoundError, OSError) as e:
            raise ModelNotInstalled(
                f"File model embedding tidak ditemukan di '{cache_dir}'. "
                "Silakan jalankan 'lokalcari setup' terlebih dahulu."
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Gagal memuat model embedding: {e}") from e

        return self._backend

    def embed_passages(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """Mengubah potongan teks menjadi matriks embedding (n, 384)."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        backend = self._get_backend()
        passages = [f"{PASSAGE_PREFIX}{t}" for t in texts]
        try:
            vectors = list(backend.embed(passages, batch_size=batch_size))
        except (ModelNotInstalled, EmbeddingError):
            raise
        except Exception as e:
            raise EmbeddingError(f"Gagal melakukan embedding passage: {e}") from e

        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)
        return arr.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Mengubah query teks menjadi vektor embedding (384,)."""
        backend = self._get_backend()
        query = f"{QUERY_PREFIX}{text}"
        try:
            vectors = list(backend.embed([query], batch_size=1))
        except (ModelNotInstalled, EmbeddingError):
            raise
        except Exception as e:
            raise EmbeddingError(f"Gagal melakukan embedding query: {e}") from e

        arr = np.array(vectors[0], dtype=np.float32)
        norm = np.linalg.norm(arr)
        arr = arr / max(norm, 1e-12)
        return arr.astype(np.float32)
