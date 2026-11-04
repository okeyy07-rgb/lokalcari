# LokalCari

LokalCari adalah aplikasi pencari dokumen pribadi dengan bahasa natural yang berjalan 100% lokal tanpa memerlukan koneksi internet saat runtime. Pengguna dapat memilih folder dokumen lokal (PDF, DOCX, TXT, MD) untuk diindeks, kemudian mengajukan pertanyaan dalam bahasa Indonesia atau Inggris untuk menemukan dokumen serta potongan teks yang relevan dengan privasi data yang sepenuhnya terjaga di komputer lokal.

## Instalasi Pengembangan

1. Clone repositori dan masuk ke direktori proyek:
   ```bash
   git clone <repo-url>
   cd lokalcari
   ```

2. Buat dan aktifkan virtual environment (Python 3.11+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Di Windows: .venv\Scripts\activate
   ```

3. Install dependensi dalam mode editable:
   ```bash
   pip install -e .[dev]
   ```

4. Jalankan perintah verifikasi:
   ```bash
   lokalcari version
   lokalcari --help
   pytest
   ruff check .
   ```
