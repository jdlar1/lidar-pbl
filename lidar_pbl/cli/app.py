import typer

app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Path to the input file"),
    output_file: str = typer.Argument(..., help="Path to the output file"),
):
    """
    Command line interface for the lidar_pbl package.
    """
    typer.echo(f"Input file: {input_file}")
    typer.echo(f"Output file: {output_file}")


@app.command()
def convert(
    input_file: str = typer.Argument(..., help="Path to the input file"),
    output_file: str = typer.Argument(..., help="Path to the output file"),
):
    """
    Command line interface for the lidar_pbl package.
    """
    typer.echo(f"Input file: {input_file}")
    typer.echo(f"Output file: {output_file}")


def run():
    app()
