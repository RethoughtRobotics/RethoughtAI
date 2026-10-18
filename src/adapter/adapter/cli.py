import sys

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
@click.argument("session", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", default="configs/baxter.yaml", show_default=True, type=click.Path(exists=True),
              help="Robot config: cameras from topics.images, arm and fps from inference")
@click.option("--repo-id", required=True, help="Hub dataset id, e.g. user/CupToContainer")
@click.option("--task", required=True, help="Language instruction stored with every frame")
@click.option("--output", default=None, type=click.Path(path_type=Path),
              help="Dataset folder (default: datasets/<session name>)")
def lerobot(session, config, repo_id, task, output):
    """Convert a teleop session (every episode_*) to a LeRobot v3 dataset on labels.py's actions.

    \b
    Example:
      adapt lerobot recorder/data/CupToContainer --repo-id andnetdeboer/CupToContainer \\
          --task "Pick up the blue cup and place it in the red container"
    """
    from adapter.lerobot_export import export

    cfg = yaml.safe_load(Path(config).read_text())
    cameras = {i["topic"]: i["key"].removeprefix("observation.images.") for i in cfg["topics"]["images"]}
    arm, fps = cfg["inference"]["arm"], cfg["inference"]["fps"]
    other = "right" if arm == "left" else "left"
    cameras = {topic: key for topic, key in cameras.items() if other not in topic}  # the other arm's camera
    output = output or Path("datasets") / session.name
    if output.exists():
        raise click.UsageError(f"{output} exists; remove it to rebuild the dataset")
    export(session, repo_id, output, task, cameras, arm=arm, fps=fps, log=click.echo)
    click.echo(f"Dataset written to {output}")


@main.command()
@click.argument("dataset", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--repo-id", required=True, help="Hub dataset id, e.g. user/CupToContainer")
@click.option("--private/--public", default=True, show_default=True, help="Visibility if the repo is new")
@click.option("--collection", default=None, help="Collection slug to add the dataset to")
def push(dataset, repo_id, private, collection):
    """Upload a LeRobot dataset folder with `hf upload-large-folder` (resumable) and tag it.

    \b
    Example:
      adapt push datasets/CupToContainer --repo-id andnetdeboer/CupToContainer \\
          --collection andnetdeboer/rethoughtai-6ac6adec399ec52e90e21556
    Pull it back with: hf download andnetdeboer/CupToContainer --repo-type dataset --local-dir DIR
    """
    import shutil
    import subprocess

    from huggingface_hub import HfApi
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

    hf = shutil.which("hf") or str(Path(sys.executable).with_name("hf"))
    subprocess.run([hf, "upload-large-folder", repo_id, str(dataset), "--repo-type", "dataset",
                    "--private" if private else "--no-private"], check=True)
    api = HfApi()
    # LeRobotDataset loads the revision tagged with its format version; upload-large-folder does not tag.
    api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset", exist_ok=True)
    if collection:
        api.add_collection_item(collection, item_id=repo_id, item_type="dataset", exists_ok=True)
    click.echo(f"Done -> https://huggingface.co/datasets/{repo_id} (tag {CODEBASE_VERSION})")


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def quality(dataset_path):
    """Score episode quality in a LeRobot v3 dataset."""
    click.echo("adapt_cli quality — not yet implemented")
