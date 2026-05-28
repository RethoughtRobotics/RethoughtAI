import importlib.resources
from pathlib import Path

import click

from recorder import __version__


def resolve_config(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    bundled = importlib.resources.files("recorder") / "configs" / Path(path).name
    try:
        resolved = Path(str(bundled))
        if resolved.exists():
            return resolved
    except (TypeError, FileNotFoundError):
        pass
    raise click.BadParameter(f"Config not found: {path}")


@click.group()
@click.version_option(__version__)
def main():
    """RAI Recorder — collect robot demonstrations."""
    pass


@main.command()
@click.option("--config", required=True, help="Robot config YAML (name or path)")
@click.option("--name", default=None, help="Episode name prefix")
@click.option("--output", default="./recordings", show_default=True, help="Output directory")
@click.option("--no-viz", is_flag=True, help="Disable Rerun visualization")
def record(config, name, output, no_viz):
    """Start a recording session."""
    config_path = resolve_config(config)
    click.echo(f"Loading config: {config_path}")
    click.echo("recorder record — not yet implemented")


@main.command()
def stop():
    """Stop the active recording session."""
    click.echo("recorder stop — not yet implemented")


@main.command()
def commit():
    """Commit the current snapshot buffer as a successful episode."""
    click.echo("recorder commit — not yet implemented")


@main.command()
def discard():
    """Discard the current snapshot buffer."""
    click.echo("recorder discard — not yet implemented")


@main.command()
def status():
    """Show currently subscribed topics."""
    click.echo("recorder status — not yet implemented")
