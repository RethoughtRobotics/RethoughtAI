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


def policy_settings(cfg, model_cfg):
    """inference.policies settings for this checkpoint's type."""
    kind = model_cfg["type"]
    if kind not in cfg.get("policies", {}):
        raise click.UsageError(f"no inference.policies.{kind} in the config (have: {', '.join(cfg.get('policies', {}))})")
    return cfg["policies"][kind]


def spin(name, setup, on_exit=None):
    """Spin a node set up by setup(node) until Ctrl+C. on_exit(), if given, runs before the node goes,
    so it can still publish: rclpy's own SIGINT handler would shut the context down first."""
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(signal_handler_options=SignalHandlerOptions.ALL if on_exit is None else SignalHandlerOptions.NO)
    node = rclpy.create_node(name)
    setup(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if on_exit is not None:
            on_exit()
        node.destroy_node()
        rclpy.try_shutdown()


@click.group()
@click.version_option(__version__)
def main():
    """RAI Inference — deploy trained policies to ROS2 robots."""
    pass


@main.command()
@click.argument("config")
@click.option("--model", required=True, help="LeRobot checkpoint: Hub repo (user/name) or local directory")
@click.option("--device", default="cuda", show_default=True)
@click.option("--revision", default=None, help="Hub branch, tag or commit of --model (e.g. step-005000)")
def serve(config, model, device, revision):
    """Run a policy on the robot's topics, publishing its action chunks for `run`.

    \b
    Examples:
      inference serve baxter.yaml --model andnetdeboer/ACT_20261007-2113_ckpt010000
    """
    full = yaml.safe_load(resolve_config(config).read_text())
    cfg = full["inference"]

    from inference.policies import LeRobotPolicy, model_config
    from inference.runtime import check_merge

    settings = policy_settings(cfg, model_config(model, revision))
    policy = LeRobotPolicy(model, device, cfg.get("task", ""), settings.get("rtc"), revision, settings.get("overrides"))
    check_merge(settings["merge"], policy.chunk_size, settings.get("every", 1))
    click.echo(f"Serving {model} ({policy.type}, chunks of {policy.chunk_size}) on {device}: every "
               f"{settings.get('every', 1)} step(s){', RTC' if policy.rtc else ''}, chunks on {cfg['chunk']}")

    from inference.policy_node import PolicyNode, RecordingLight

    lights = []

    def setup(node):
        PolicyNode(node, cfg, full["topics"]["images"], policy, settings.get("every", 1))
        if full.get("trigger", {}).get("light"):  # lit as in every recorded frame
            lights.append(RecordingLight(node, full["trigger"]["light"]))

    spin("policy_server", setup, on_exit=lambda: [light.off() for light in lights])


@main.command()
@click.argument("config")
@click.option("--replay", "episode", type=click.Path(exists=True, file_okay=False),
              help="Play a recorded episode back as the policy")
@click.option("--model", default=None, help="Execute this checkpoint's chunks from `serve` (its merge settings)")
@click.option("--start-at", type=click.Path(exists=True, file_okay=False),
              help="With --model: first move to this recorded episode's start pose (shadow rollout)")
@click.option("--player", default=None,
              help="With --start-at: the paused `ros2 bag play` node (e.g. /rosbag2_player) to seek to the "
                   "episode's start and resume once the arm is there")
@click.option("--revision", default=None, help="Hub branch, tag or commit of --model (e.g. step-005000)")
def run(config, episode, model, start_at, player, revision):
    """Run a policy on the robot, through its teleop (the config's `inference:` section).

    \b
    Examples:
      inference run baxter.yaml --replay recorder/data/CupToContainer/episode_042
      inference run baxter.yaml --model andnetdeboer/ACT_20261007-2113_ckpt010000
    """
    if (episode is None) == (model is None):
        raise click.UsageError("give one of --replay or --model")
    if start_at and not model or player and not start_at:
        raise click.UsageError("--start-at needs --model, --player needs --start-at")
    cfg = yaml.safe_load(resolve_config(config).read_text())["inference"]

    from adapter.labels import load_episode
    from inference.runtime import Runtime, check_merge

    if episode:
        from inference.policies import ReplayPolicy

        policy = ReplayPolicy(episode, cfg["arm"], cfg["fps"], cfg["chunk_size"])
        click.echo(f"Replaying {episode}: {len(policy)} steps ({len(policy) / cfg['fps']:.1f} s), "
                   f"chunks of {cfg['chunk_size']}, ensembling {cfg.get('temporal_ensemble_coeff')}")
        start = policy.episode.tcp_p[0]
        click.echo(f"Demo starts with the TCP at {start.round(3).tolist()} m (base). Hold the deadman "
                   "(trackpad) to run, release to hold; B sends the arm home.")
        spin("policy_runtime", lambda node: Runtime(node, cfg, policy))
        return

    from inference.policies import chunk_length, model_config

    model_cfg = model_config(model, revision)
    settings = policy_settings(cfg, model_cfg)
    warning = check_merge(settings["merge"], chunk_length(model_cfg), settings.get("every", 1))
    if warning:
        click.echo(f"Warning: {warning}")
    cfg = {**cfg, "chunk_size": chunk_length(model_cfg)}
    start = None
    if start_at:
        recorded = load_episode(start_at, cfg["arm"], cfg["fps"])
        start, first = (recorded.tcp_p[0], recorded.tcp_q[0]), int(recorded.t[0])
        click.echo(f"Shadow rollout of {start_at}: moving to its start pose "
                   f"{start[0].round(3).tolist()} m (base) first" + (f", then playing {player}" if player else ""))
    click.echo(f"Executing {model_cfg['type']} chunks from {cfg['chunk']}, merge {settings['merge']}. Engage with "
               f"the deadman or `ros2 service call {cfg['engage']} std_srvs/srv/SetBool \"{{data: true}}\"`.")

    def setup(node):
        on_start = start_player(node, player, first) if player else None
        Runtime(node, cfg, merge=settings["merge"], start=start, on_start=on_start)

    spin("policy_runtime", setup)


def start_player(node, player, t_ns):
    """on_start for a shadow rollout: play the episode's cameras from its first step."""
    from rclpy.time import Time
    from rosbag2_interfaces.srv import Resume, Seek

    seek = node.create_client(Seek, f"{player}/seek")
    resume = node.create_client(Resume, f"{player}/resume")

    def go():
        if not (seek.service_is_ready() and resume.service_is_ready()):
            node.get_logger().error(f"{player} has no seek/resume services: is `ros2 bag play --start-paused` running?")
            return
        future = seek.call_async(Seek.Request(time=Time(nanoseconds=t_ns).to_msg()))
        future.add_done_callback(lambda _: resume.call_async(Resume.Request()))
        node.get_logger().info(f"Playing the episode's cameras from {player}")

    return go


@main.command()
@click.argument("model")
@click.option("--config", default="baxter.yaml", show_default=True, help="Its inference.task is the VLA's prompt")
@click.option("--device", default="cuda", show_default=True)
@click.option("--depth", type=int, default=None, help="Levels of the tree to print (default: all)")
@click.option("--json", "out", type=click.Path(dir_okay=False), help="Also write the tree as JSON (for recap)")
def probe(model, config, device, depth, out):
    """Print a policy's module tree, traced through one inference: calls and shapes per module.

    \b
    Examples:
      inference probe models/smolvla_CupToContainer/step-030000 --depth 4
      inference probe models/vita_CupToContainer/step-060000 --json vita_tree.json
    """
    from inference import probe as probe_mod
    from inference.policies import LeRobotPolicy

    task = yaml.safe_load(resolve_config(config).read_text())["inference"].get("task", "")
    root = probe_mod.trace_policy(LeRobotPolicy(model, device, task))
    click.echo(probe_mod.text(root, depth))
    if out:
        Path(out).write_text(probe_mod.to_json(root))
        click.echo(f"\n{out}")


@main.command()
@click.argument("model")
@click.option("--config", default="baxter.yaml", show_default=True)
@click.option("--device", default="cuda", show_default=True)
def attend(model, config, device):
    """Serve attention maps for recap: JSON requests on stdin, replies on stdout (inference.attention)."""
    import sys

    from inference import attention
    from inference.policies import LeRobotPolicy

    out, sys.stdout = sys.stdout, sys.stderr  # replies only on stdout: library prints go to stderr
    full = yaml.safe_load(resolve_config(config).read_text())
    cfg = full["inference"]
    policy = LeRobotPolicy(model, device, cfg.get("task", ""))
    attention.serve(policy, cfg, full["topics"]["images"], sys.stdin, out)
