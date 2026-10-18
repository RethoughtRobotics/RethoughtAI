from pathlib import Path

import click
import yaml

from inference import __version__


def resolve_config(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    fallback = Path("configs") / p.name
    if fallback.exists():
        return fallback
    raise click.BadParameter(f"Config not found: {path}")


@click.group()
@click.version_option(__version__)
def main():
    """RAI Inference — deploy trained policies to ROS2 robots."""
    pass


@main.command()
@click.argument("config")
@click.option("--replay", "episode", type=click.Path(exists=True, file_okay=False),
              help="Play a recorded episode back as the policy")
@click.option("--model", default=None, help="HuggingFace model repo (e.g. username/model-name)")
def run(config, episode, model):
    """Run a policy on the robot, through its teleop (the config's `inference:` section).

    \b
    Examples:
      inference run baxter.yaml --replay recorder/data/CupToContainer/episode_042
    """
    if model is not None or episode is None:
        raise click.UsageError("only --replay is implemented so far")
    cfg = yaml.safe_load(resolve_config(config).read_text())["inference"]

    import rclpy
    from inference.policies import ReplayPolicy
    from inference.runtime import Runtime

    policy = ReplayPolicy(episode, cfg["arm"], cfg["fps"], cfg["chunk_size"])
    click.echo(f"Replaying {episode}: {len(policy)} steps ({len(policy) / cfg['fps']:.1f} s), "
               f"chunks of {cfg['chunk_size']}, ensembling {cfg.get('temporal_ensemble_coeff')}")
    start = policy.episode.tcp_p[0]
    click.echo(f"Demo starts with the TCP at {start.round(3).tolist()} m (base). Hold the deadman "
               "(trackpad) to run, release to hold; B sends the arm home.")

    rclpy.init()
    node = rclpy.create_node("policy_runtime")
    Runtime(node, cfg, policy)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
