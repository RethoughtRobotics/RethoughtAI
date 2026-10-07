import click
from adapter import __version__


@click.group()
@click.version_option(__version__)
def main():
    """RAI Adapter — convert MCAP recordings to LeRobot datasets."""
    pass


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Robot config YAML")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="MCAP recordings directory")
@click.option("--output", default="./dataset", show_default=True, help="Output LeRobot dataset path")
@click.option("--quality-threshold", default=0.7, show_default=True, help="Minimum quality score (0-1)")
@click.option("--push-to-hub", default=None, help="HuggingFace repo to push dataset (e.g. username/dataset-name)")
def convert(config, input_path, output, quality_threshold, push_to_hub):
    """Convert MCAP recordings to a LeRobot dataset."""
    click.echo(f"Loading config: {config}")
    click.echo(f"Input: {input_path}")
    click.echo("adapter convert — not yet implemented")


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def quality(dataset_path):
    """Score episode quality in a dataset."""
    click.echo("adapter quality — not yet implemented")
