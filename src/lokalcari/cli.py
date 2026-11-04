from pathlib import Path

import typer

from lokalcari import __version__

app = typer.Typer(
    name="lokalcari",
    help="LokalCari: Aplikasi pencari dokumen pribadi 100% lokal.",
    no_args_is_help=True,
)


@app.callback()
def main():
    """LokalCari: Aplikasi pencari dokumen pribadi 100% lokal."""


@app.command()
def version():
    """Tampilkan versi aplikasi."""
    typer.echo(f"lokalcari version: {__version__}")


def _calculate_disk_usage(directory: Path) -> int:
    """Hitung ukuran disk sebenarnya dari file unik tanpa mengikuti symlink/reparse point."""
    import os
    import stat

    seen_inodes = set()
    total_bytes = 0
    for root, _dirs, files in os.walk(directory, followlinks=False):
        for f in files:
            p = Path(root) / f
            if p.is_symlink():
                continue
            try:
                lst = p.lstat()
                is_reparse = hasattr(lst, "st_file_attributes") and hasattr(
                    stat, "FILE_ATTRIBUTE_REPARSE_POINT"
                ) and bool(lst.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
                if is_reparse:
                    continue
                key = (lst.st_dev, lst.st_ino)
                if key not in seen_inodes:
                    seen_inodes.add(key)
                    total_bytes += lst.st_size
            except OSError:
                continue
    return total_bytes


@app.command()
def setup():
    """Unduh dan siapkan model embedding lokal."""
    from lokalcari.embedder import download_model

    typer.echo("Menyiapkan model embedding lokal...")
    try:
        model_dir = download_model()
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Gagal mengunduh model: {e}", err=True)
        raise typer.Exit(code=1)

    total_bytes = _calculate_disk_usage(model_dir)
    size_mb = total_bytes / (1024 * 1024)
    typer.echo(f"Model berhasil disiapkan di: {model_dir}")
    typer.echo(f"Ukuran di disk: {size_mb:.2f} MB")


if __name__ == "__main__":
    app()
