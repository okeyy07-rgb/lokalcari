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

    total_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_mb = total_bytes / (1024 * 1024)
    typer.echo(f"Model berhasil disiapkan di: {model_dir}")
    typer.echo(f"Ukuran folder model: {size_mb:.2f} MB")


if __name__ == "__main__":
    app()
