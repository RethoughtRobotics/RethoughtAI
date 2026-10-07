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
    """RAI record_cli — collect robot demonstrations."""
    pass


@main.command()
@click.argument("config")
@click.option("--output", default="./recorder/data", show_default=True, help="Output directory")
@click.option("--no-viz", is_flag=True, help="Disable Rerun visualization")
@click.option("--session", default=None, help="Session name: episodes go to <output>/<session>/ "
              "(reused if it exists). Default: session_<timestamp>")
def record(config, output, no_viz, session):
    """Start a recording session.

    \b
    Examples:
      record_cli record baxter.yaml
      record_cli record baxter.yaml --output ./my_recordings --no-viz
      record_cli record baxter.yaml --session CupInCube
    """
    config_path = resolve_config(config)
    cfg = yaml.safe_load(config_path.read_text())
    session_dir = make_session_dir(output, config_path, session)
    run_session(session_dir, cfg, all_topics(cfg), viz=not no_viz)
