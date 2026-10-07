import click
import yaml
from pathlib import Path
from adapter import __version__


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def resolve_config(name: str) -> Path:
    """Find a config by path or by name inside adapter/configs/."""
    p = Path(name)
    if p.exists():
        return p
    fallback = CONFIGS_DIR / p.name
    if fallback.exists():
        return fallback
    raise click.BadParameter(f"Config not found: {name} (looked in {CONFIGS_DIR})")


@click.group()
@click.version_option(__version__)
def main():
    """RAI Adapter — convert session recordings to LeRobot v3 datasets."""
    pass


@main.command()
@click.argument("config")
@click.argument("sessions", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", default=None, help="Override output directory from config")
@click.option("--hub-repo", default=None, help="Override HuggingFace repo from config")
def adapt(config, sessions, output, hub_repo):
    """Convert session(s) to LeRobot v3 using a config file.

    \b
    Examples:
      adapt_cli adapt baxter.yaml recordings/session_2026_06_03
      adapt_cli adapt baxter.yaml recordings/session_* --hub-repo user/my-dataset
    """
    from adapter.adapter import to_lerobot
    cfg = yaml.safe_load(resolve_config(config).read_text())

    to_lerobot(
        sessions,
        output      or cfg.get("output", "./dataset"),
        hub_repo    = hub_repo or cfg.get("hub_repo") or None,
        robot_type  = cfg.get("robot_type", "unknown"),
        fps         = cfg.get("fps", 30.0),
        camera_mapping = cfg.get("camera_mapping"),
    )


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def quality(dataset_path):
    """Score episode quality in a LeRobot v3 dataset."""
    click.echo("adapt_cli quality — not yet implemented")
