from pathlib import Path

import click
import yaml

from recorder import __version__
from recorder.session import make_session_dir, run_session


def resolve_config(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    fallback = Path("configs") / p.name
    if fallback.exists():
        return fallback
    raise click.BadParameter(f"Config not found: {path}")


def all_topics(config: dict) -> list[str]:
    t = config["topics"]
    return (
        [i["topic"] for i in t["images"]]
        + [s["topic"] for s in t["state"]]
        + t.get("extra", [])
    )


@click.group()
@click.version_option(__version__)
def main():
    """RAI Recorder — collect robot demonstrations."""
    pass


@main.command()
@click.option("--config", required=True, help="Robot config YAML (name or path)")
@click.option("--output", default="./recordings", show_default=True, help="Output directory")
@click.option("--no-viz", is_flag=True, help="Disable Rerun visualization")
def record(config, output, no_viz):
    """Start a recording session."""
    config_path = resolve_config(config)
    cfg = yaml.safe_load(config_path.read_text())
    session_dir = make_session_dir(output, config_path)
    run_session(session_dir, cfg, all_topics(cfg))


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
