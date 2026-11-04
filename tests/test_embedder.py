import os
import socket
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from lokalcari.cli import app
from lokalcari.embedder import (
    EMBEDDING_DIM,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    Embedder,
    EmbeddingError,
    ModelNotInstalled,
    model_cache_dir,
)


class MockBackend:
    """Backend tiruan untuk pengujian tanpa model ONNX asli."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.recorded_calls: list[dict] = []

    def embed(self, texts: list[str], batch_size: int = 32):
        self.recorded_calls.append({"texts": list(texts), "batch_size": batch_size})
        for t in texts:
            # Vektor deterministik berbasis karakter teks
            seed = sum(ord(c) for c in t) % 1000
            vec = np.arange(self.dim, dtype=np.float32) + seed + 1.0
            norm = np.linalg.norm(vec)
            yield vec / max(norm, 1e-12)


def test_embedder_dim():
    backend = MockBackend()
    embedder = Embedder(backend=backend)
    assert embedder.dim == EMBEDDING_DIM


def test_embed_passages_prefixes_and_shape():
    backend = MockBackend()
    embedder = Embedder(backend=backend)

    texts = ["teks satu", "teks dua"]
    arr = embedder.embed_passages(texts, batch_size=16)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, EMBEDDING_DIM)
    assert arr.dtype == np.float32

    # Pastikan awalan passage: ditambahkan
    recorded = backend.recorded_calls[0]
    assert recorded["texts"] == [
        f"{PASSAGE_PREFIX}teks satu",
        f"{PASSAGE_PREFIX}teks dua",
    ]
    assert recorded["batch_size"] == 16

    # Pastikan ternormalisasi (norma 1)
    for row in arr:
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-5)


def test_embed_passages_empty_list():
    backend = MockBackend()
    embedder = Embedder(backend=backend)

    arr = embedder.embed_passages([])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (0, EMBEDDING_DIM)
    assert arr.dtype == np.float32
    assert len(backend.recorded_calls) == 0


def test_embed_passages_batching_equivalence():
    backend = MockBackend()
    embedder = Embedder(backend=backend)

    texts = [f"dokumen contoh nomor {i}" for i in range(10)]
    arr_batch_2 = embedder.embed_passages(texts, batch_size=2)
    arr_batch_5 = embedder.embed_passages(texts, batch_size=5)

    assert np.allclose(arr_batch_2, arr_batch_5, atol=1e-6)


def test_embed_query_prefix_and_shape():
    backend = MockBackend()
    embedder = Embedder(backend=backend)

    q = "apa itu lokalcari?"
    vec = embedder.embed_query(q)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBEDDING_DIM,)
    assert vec.dtype == np.float32

    # Pastikan awalan query: ditambahkan
    recorded = backend.recorded_calls[0]
    assert recorded["texts"] == [f"{QUERY_PREFIX}{q}"]

    # Pastikan ternormalisasi (norma 1)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_model_not_installed_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("LOKALCARI_MODEL_DIR", str(tmp_path))

    embedder = Embedder()
    with pytest.raises(ModelNotInstalled, match="lokalcari setup"):
        embedder.embed_query("halo")

    with pytest.raises(ModelNotInstalled, match="lokalcari setup"):
        embedder.embed_passages(["halo"])


def test_model_not_installed_half_corrupt_directory(tmp_path, monkeypatch):
    model_dir = tmp_path / "corrupt_model"
    model_dir.mkdir(parents=True)
    # File parsial atau snapshot tidak lengkap
    (model_dir / "incomplete.tmp").write_text("dummy")

    monkeypatch.setenv("LOKALCARI_MODEL_DIR", str(model_dir))
    embedder = Embedder()

    with pytest.raises(ModelNotInstalled, match="lokalcari setup"):
        embedder.embed_query("halo")

    with pytest.raises(ModelNotInstalled, match="lokalcari setup"):
        embedder.embed_passages(["halo"])


def test_runtime_error_wrapped_as_embedding_error():
    class FailingBackend:
        def embed(self, texts: list[str], batch_size: int = 32):
            raise RuntimeError("Internal ONNX inference failure: memory allocation")

    embedder = Embedder(backend=FailingBackend())

    with pytest.raises(EmbeddingError, match="Internal ONNX inference failure"):
        embedder.embed_query("halo")

    with pytest.raises(EmbeddingError, match="Internal ONNX inference failure"):
        embedder.embed_passages(["halo"])


def test_model_cache_dir_resolution(tmp_path, monkeypatch):
    # Default path
    monkeypatch.delenv("LOKALCARI_MODEL_DIR", raising=False)
    default_dir = model_cache_dir()
    assert default_dir == Path.home() / ".lokalcari" / "models"

    # Dari environment variable
    monkeypatch.setenv("LOKALCARI_MODEL_DIR", str(tmp_path / "custom_models"))
    custom_dir = model_cache_dir()
    assert custom_dir == tmp_path / "custom_models"


def test_cli_setup_success(tmp_path, monkeypatch):
    runner = CliRunner()

    # Buat file tiruan dalam tmp_path
    dummy_model_file = tmp_path / "model.onnx"
    dummy_model_file.write_bytes(b"0" * 1024 * 1024)  # 1 MB

    monkeypatch.setattr(
        "lokalcari.embedder.download_model",
        lambda: tmp_path,
    )

    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "Model berhasil disiapkan di" in result.output
    assert str(tmp_path) in result.output
    assert "1.00 MB" in result.output


def test_cli_setup_network_error(monkeypatch):
    runner = CliRunner()

    def mock_download_error():
        raise OSError("Koneksi jaringan terputus (Host unreachable)")

    monkeypatch.setattr(
        "lokalcari.embedder.download_model",
        mock_download_error,
    )

    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 1
    assert "Gagal mengunduh model: Koneksi jaringan terputus" in result.output
    # Pastikan tidak ada traceback Python mentah
    assert "Traceback" not in result.output


@pytest.mark.model
def test_real_model_integration(monkeypatch):
    """Uji integrasi dengan model embedding ONNX asli (hanya jika diaktifkan)."""
    if os.environ.get("LOKALCARI_RUN_MODEL_TESTS") != "1":
        pytest.skip("Test model dilewati karena LOKALCARI_RUN_MODEL_TESTS!=1")

    # Blokir akses jaringan pada socket level untuk membuktikan model 100% offline
    def guard_socket(*args, **kwargs):
        raise OSError("Akses jaringan dilarang pada test model integrasi!")

    monkeypatch.setattr(socket.socket, "connect", guard_socket)
    monkeypatch.setattr(socket, "create_connection", guard_socket)

    embedder = Embedder()
    assert embedder.dim == 384

    sentences = [
        "Kucing anggora tidur nyenyak di atas kasur yang empuk.",
        "Server Ubuntu berjalan stabil di cloud provider untuk hosting aplikasi.",
        "Resep masakan rendang daging sapi khas Minangkabau sangat lezat.",
    ]
    query = "Bagaimana cara deploy sistem di server Linux?"

    passages_vec = embedder.embed_passages(sentences)
    assert passages_vec.shape == (3, 384)
    assert passages_vec.dtype == np.float32

    # Verifikasi norma 1
    for v in passages_vec:
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)

    query_vec = embedder.embed_query(query)
    assert query_vec.shape == (384,)
    assert query_vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(query_vec), 1.0, atol=1e-5)

    # Uji semantik: kalimat 2 (server Ubuntu/cloud) harus memiliki kesamaan kosinus tertinggi
    similarities = np.dot(passages_vec, query_vec)
    best_match_idx = int(np.argmax(similarities))
    assert best_match_idx == 1, (
        f"Kalimat 2 harusnya paling relevan, didapat idx {best_match_idx} "
        f"dengan skor {similarities}"
    )
