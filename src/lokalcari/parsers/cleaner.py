import re


def clean_text(text: str) -> str:
    """Pembersihan ringan teks: hapus karakter null dan rapikan baris kosong berlebihan."""
    if not text:
        return ""
    # Hapus karakter null
    cleaned = text.replace("\x00", "")
    # Normalisasi line ending ke \n
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Rapikan baris kosong berlebihan (3 atau lebih newline berurutan dijadikan 2 newline)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
