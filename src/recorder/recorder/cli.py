import re
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


def exclude_topics(config: dict, patterns: tuple[str, ...]) -> dict:
    """Drop configured topics matching any regex in patterns (e.g. 'right' for a left-arm task).
    `operator` topics are kept: VR controllers can drive the other side's arm."""
    def keep(topic):
        return not any(re.search(p, topic) for p in patterns)

    t = config["topics"]
    t["images"] = [i for i in t["images"] if keep(i["topic"])]
    t["state"] = [s for s in t["state"] if keep(s["topic"])]
    t["extra"] = [e for e in t.get("extra", []) if keep(e)]
    return config


def all_topics(config: dict) -> list[str]:
    t = config["topics"]
    return (
        [i["topic"] for i in t["images"]]
        + [s["topic"] for s in t["state"]]
        + t.get("extra", [])
        + t.get("operator", [])
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
@click.option("--exclude", multiple=True, metavar="REGEX",
              help="Skip configured topics matching REGEX (repeatable), e.g. --exclude right")
def record(config, output, no_viz, session, exclude):
    """Start a recording session.

    \b
    Examples:
      record_cli record baxter.yaml
      record_cli record baxter.yaml --output ./my_recordings --no-viz
      record_cli record baxter.yaml --session CupInCube
      record_cli record baxter.yaml --session CupInCube --exclude right
    """
    config_path = resolve_config(config)
    cfg = exclude_topics(yaml.safe_load(config_path.read_text()), exclude)
    if exclude:
        click.echo(f"Excluding topics matching {list(exclude)}")
    session_dir = make_session_dir(output, config_path, session)
    run_session(session_dir, cfg, all_topics(cfg), viz=not no_viz)
