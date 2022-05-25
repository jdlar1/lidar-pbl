import typer

from lidar_pbl import LidarDataset

app = typer.Typer()


@app.command()
def quicklook(
    data_dir: str = typer.Argument(..., help="Path to the data directory"),
    dark_current_dir: str = typer.Argument(
        ..., help="Path to the dark current directory"
    ),
    bin_zero: int = typer.Option(12, help="Number of bins to be removed from the bottom"),
    max_height: float = typer.Option(2000, help="Maximum height in the Quicklook"),
    max_height_methods: float = typer.Option(1250, help="Maximum height to search for the PBL"),
    methods: bool = typer.Option(False, help="Plot the methods"),
):
    """
    Command line interface for the lidar_pbl package.
    """
    lidar_dataset = LidarDataset(
        data_dir=data_dir,
        dark_current_dir=dark_current_dir,
        bin_zero=bin_zero,
    )

    lidar_dataset.quicklook(max_height=max_height)

    if methods:
        lidar_dataset.gradient_pbl(min_height=400, max_height=max_height_methods, min_grad=-0.05)
        lidar_dataset.wavelet_pbl(min_height=400, max_height=max_height_methods, a_meters=90)
        lidar_dataset.variance_pbl(min_height=400, max_height=max_height_methods)

    lidar_dataset.show()


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
