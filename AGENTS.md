# AGENTS.md — LokalCari

## Tujuan produk
LokalCari adalah aplikasi pencari dokumen pribadi dengan bahasa natural yang berjalan 100% lokal (tanpa internet saat dipakai). Pengguna memilih folder, aplikasi mengindeks file (PDF, DOCX, TXT, MD), lalu pengguna bertanya dalam bahasa Indonesia atau Inggris dan mendapat file serta potongan teks yang relevan. Nilai jual utama: privasi, data tidak keluar dari komputer.

## Tech stack
- Python 3.11+ (target uji: 3.11, 3.12, 3.13)
- CLI: Typer
- Penyimpanan: SQLite
- Test: pytest, lint: ruff
- Embedding dan parser dokumen akan ditambahkan di task khusus. Jangan menambahkannya sebelum diminta.

## Struktur folder target
```
lokalcari/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/lokalcari/
│   ├── parsers/
│   ├── chunker.py
│   ├── embedder.py
│   ├── store.py
│   ├── indexer.py
│   ├── search.py
│   └── cli.py
└── tests/
```

## Aturan kerja
1. Setiap fitur wajib punya test.
2. Commit kecil dengan pesan jelas.
3. Jangan menambah dependency baru tanpa izin eksplisit dari pemilik project.
4. Jangan memanggil API eksternal atau internet saat runtime aplikasi. Download model hanya boleh lewat langkah setup yang eksplisit.
5. Kerja di branch terpisah (feat/nama-task). Jangan merge ke main dan jangan push kecuali diminta.
6. Jangan mengubah file di luar lingkup task.
7. Jangan mengubah AGENTS.md tanpa izin pemilik project.
8. Selalu pakai virtual environment `.venv` di root project.

## Definisi selesai
Kode berjalan, test lulus, `ruff check .` bersih, dan ada laporan akhir.

## Format laporan akhir
1. Daftar file yang dibuat atau diubah
2. Output asli perintah yang dijalankan (jangan diringkas)
3. Hal yang gagal atau tidak pasti