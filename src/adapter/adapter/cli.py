import math
import sys
from pathlib import Path

import click
import yaml

from adapter import __version__

CONFIGS_DIR = Path(__file__).parent.parent / 'configs'


def resolve_config(name: str) -> Path:
    """Find a config by path or by name inside adapter/configs/."""
    p = Path(name)
    if p.exists():
        return p
    fallback = CONFIGS_DIR / p.name
    if fallback.exists():
        return fallback
    raise click.BadParameter(f'Config not found: {name} (looked in {CONFIGS_DIR})')


@click.group()
@click.version_option(__version__)
def main():
    """RAI Adapter: convert session recordings to LeRobot v3 datasets."""
    pass


@main.command()
@click.argument('config')
@click.argument('sessions', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option('--output', default=None, help='Override output directory from config')
@click.option('--hub-repo', default=None, help='Override HuggingFace repo from config')
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
        output or cfg.get('output', './dataset'),
        hub_repo=hub_repo or cfg.get('hub_repo') or None,
        robot_type=cfg.get('robot_type', 'unknown'),
        fps=cfg.get('fps', 30.0),
        camera_mapping=cfg.get('camera_mapping'),
    )


@main.command()
@click.argument('session', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--config',
    default='configs/baxter.yaml',
    show_default=True,
    type=click.Path(exists=True),
    help='Robot config: cameras from topics.images, arm and fps from inference',
)
@click.option('--repo-id', required=True, help='Hub dataset id, e.g. user/CupToContainer')
@click.option('--task', required=True, help='Language instruction stored with every frame')
@click.option(
    '--output', default=None, type=click.Path(path_type=Path), help='Dataset folder (default: datasets/<session name>)'
)
@click.option(
    '--arms', default=None, help="Comma-separated arms, e.g. left,right (default: the config's inference arm)"
)
@click.option(
    '--labels',
    type=click.Choice(['goal', 'measured']),
    default='goal',
    show_default=True,
    help='goal: what the teleop servo chased; measured: the measured TCP (kinesthetic demos)',
)
@click.option('--lead', default=1, show_default=True, help='measured labels: the TCP this many steps later')
@click.option(
    '--gripper/--no-gripper',
    default=True,
    show_default=True,
    help='Gripper in state and action (--no-gripper when the hands hold tools)',
)
@click.option(
    '--trim-idle',
    default=None,
    type=(float, float),
    help='DISTANCE ANGLE (m, deg): start each episode just before its TCP first moves that far (the wait at the start)',
)
@click.option(
    '--trim-still',
    default=None,
    type=(float, float),
    help='DISTANCE ANGLE (m, deg): cut every still stretch (TCP, goal and gripper unchanged) to its first frame',
)
@click.option(
    '--workers', default=6, show_default=True, help='Episodes exported in parallel processes, then joined in order'
)
def lerobot(session, config, repo_id, task, output, arms, labels, lead, gripper, trim_idle, trim_still, workers):
    """Convert a teleop session (every episode_*) to a LeRobot v3 dataset on labels.py's actions.

    \b
    Examples:
      adapt lerobot recorder/data/CupToContainer --repo-id andnetdeboer/CupToContainer \\
          --task "Pick up the blue cup and place it in the red container"
      adapt lerobot recorder/data/SweepTrash_Kin --repo-id andnetdeboer/SweepTrash_Kin \\
          --arms left,right --labels measured --no-gripper --task "Sweep the trash into the dustpan and empty it into the bin"
    """
    from adapter.lerobot_export import export

    cfg = yaml.safe_load(Path(config).read_text())
    cameras = {i['topic']: i['key'].removeprefix('observation.images.') for i in cfg['topics']['images']}
    arm, fps = cfg['inference']['arm'], cfg['inference']['fps']
    arms = arms.split(',') if arms else [arm]
    others = {'left', 'right'} - set(arms)
    cameras = {
        topic: key for topic, key in cameras.items() if not any(o in topic for o in others)
    }  # unused arms' cameras
    output = output or Path('datasets') / session.name
    if output.exists():
        raise click.UsageError(f'{output} exists; remove it to rebuild the dataset')
    export(
        session,
        repo_id,
        output,
        task,
        cameras,
        arm=arms[0],
        fps=fps,
        log=click.echo,
        arms=arms,
        labels=labels,
        lead=lead,
        gripper=gripper,
        workers=workers,
        trim_idle=(trim_idle[0], math.radians(trim_idle[1])) if trim_idle else None,
        trim_still=(trim_still[0], math.radians(trim_still[1])) if trim_still else None,
    )
    click.echo(f'Dataset written to {output}')


@main.command()
@click.argument('bags', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option('--out', required=True, type=click.Path(path_type=Path), help='Session directory (created or added to)')
@click.option('--arm', default='left', show_default=True)
@click.option('--min-duration', default=2.0, show_default=True, help='s; shorter corrections are left out')
def corrections(bags, out, arm, min_duration):
    """Cut each correction (VR takeover, cuff squeeze) of rollout bags into its own episode.

    \b
    Examples:
      adapt corrections ../data/rollouts/20261021_165656_real_vr_takeover --out recorder/data/BlueCup_corrections
      adapt lerobot recorder/data/BlueCup_corrections --labels goal --repo-id ... --task ...
    """
    from adapter.corrections import extract

    rows = extract(bags, out, arm, min_duration)
    for r in rows:
        click.echo(
            f'{r["episode"]}  {r["kind"]:8s} {r["start_s"]:7.1f} s  {r["duration_s"]:5.1f} s  {Path(r["bag"]).name}'
        )
    click.echo(f'{len(rows)} corrections into {out}')


@main.command()
@click.argument('dataset', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--repo-id', required=True, help='Hub dataset id, e.g. user/CupToContainer')
@click.option('--private/--public', default=True, show_default=True, help='Visibility if the repo is new')
@click.option('--collection', default=None, help='Collection slug to add the dataset to')
def push(dataset, repo_id, private, collection):
    """Upload a LeRobot dataset folder with `hf upload-large-folder` (resumable) and tag it.

    \b
    Example:
      adapt push datasets/CupToContainer --repo-id andnetdeboer/CupToContainer \\
          --collection andnetdeboer/rethoughtai-6ac6adec399ec52e90e21556
    Pull it back with: hf download andnetdeboer/CupToContainer --repo-type dataset --local-dir DIR
    """
    import os
    import shutil
    import subprocess

    from huggingface_hub import HfApi
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

    hf = shutil.which('hf') or str(Path(sys.executable).with_name('hf'))
    # Xet's high-performance mode: more parallel transfers and memory, for a machine with both
    env = {'HF_XET_HIGH_PERFORMANCE': '1', **os.environ}
    subprocess.run(
        [
            hf,
            'upload-large-folder',
            repo_id,
            str(dataset),
            '--repo-type',
            'dataset',
            '--private' if private else '--no-private',
        ],
        check=True,
        env=env,
    )
    api = HfApi()
    # LeRobotDataset loads the revision tagged with its format version; upload-large-folder does not tag.
    api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type='dataset', exist_ok=True)
    if collection:
        api.add_collection_item(collection, item_id=repo_id, item_type='dataset', exists_ok=True)
    click.echo(f'Done -> https://huggingface.co/datasets/{repo_id} (tag {CODEBASE_VERSION})')


@main.command()
@click.argument('dataset_path', type=click.Path(exists=True))
def quality(dataset_path):
    """Score episode quality in a LeRobot v3 dataset."""
    click.echo('adapt_cli quality: not yet implemented')
