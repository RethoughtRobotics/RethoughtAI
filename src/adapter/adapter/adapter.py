"""Convert recorded sessions to LeRobot v3 and optionally push to HuggingFace Hub."""
from pathlib import Path

import click


def to_lerobot(
    sessions,
    output_dir: str | Path,
    *,
    hub_repo: str | None = None,
    robot_type: str = "baxter",
    fps: float = 30.0,
    camera_mapping: dict | None = None,
) -> Path:
    """Convert one or more session directories into a single LeRobot v3 dataset.

    Each episode_XXX/ bag within each session becomes one episode in the output.
    Bags are ordered by session, then episode number within each session.
    """
    try:
        from forge.formats.registry import FormatRegistry
    except ImportError:
        raise RuntimeError("forge-robotics not installed — run: uv add forge-robotics")

    sessions = [sessions] if isinstance(sessions, (str, Path)) else list(sessions)
    bags = sorted(b for s in sessions for b in Path(s).glob("episode_*/"))
    if not bags:
        raise ValueError(f"No episode bags found under: {sessions}")

    output_dir = Path(output_dir)
    click.echo(f"Converting {len(bags)} episode(s) → {output_dir}")

    reader = FormatRegistry.get_reader("rosbag")
    writer = FormatRegistry.get_writer("lerobot-v3")
    info   = reader.inspect(bags[0])

    # Mirror how Converter._configure_writer() applies overrides.
    if hasattr(writer, "config"):
        cfg = writer.config
        for attr, val in [("fps", fps), ("robot_type", robot_type), ("repo_id", hub_repo),
                          ("camera_name_mapping", camera_mapping)]:
            if val is not None and hasattr(cfg, attr):
                setattr(cfg, attr, val)

    for i, bag in enumerate(bags):
        click.echo(f"  [{i + 1}/{len(bags)}] {bag.name}")
        for episode in reader.read_episodes(bag):
            writer.write_episode(episode, output_dir, episode_index=i)

    writer.finalize(output_dir, info)
    click.echo(f"Dataset written → {output_dir}")

    if hub_repo:
        _push_to_hub(output_dir, hub_repo)

    return output_dir


def _push_to_hub(output_dir: Path, hub_repo: str) -> None:
    """Upload the LeRobot v3 dataset directory to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface-hub not installed — run: uv add huggingface-hub")
    click.echo(f"Pushing to hf://{hub_repo} …")
    HfApi().upload_folder(folder_path=str(output_dir), repo_id=hub_repo, repo_type="dataset")
    click.echo(f"Done → https://huggingface.co/datasets/{hub_repo}")
