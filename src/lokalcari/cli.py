import typer

from lokalcari import __version__

app = typer.Typer(
    name="lokalcari",
    help="LokalCari: Aplikasi pencari dokumen pribadi 100% lokal.",
    no_args_is_help=True,
)


@app.command()
def version():
    """Tampilkan versi aplikasi."""
    typer.echo(f"lokalcari version: {__version__}")


if __name__ == "__main__":
    app()
