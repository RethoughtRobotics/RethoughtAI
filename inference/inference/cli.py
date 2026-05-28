import click
from inference import __version__


@click.group()
@click.version_option(__version__)
def main():
    """RAI Inference — deploy trained policies to ROS2 robots."""
    pass


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Robot config YAML")
@click.option("--model", required=True, help="HuggingFace model repo (e.g. username/model-name)")
def run(config, model):
    """Run a trained policy on the robot."""
    click.echo(f"Loading config: {config}")
    click.echo(f"Model: {model}")
    click.echo("inference run — not yet implemented")
