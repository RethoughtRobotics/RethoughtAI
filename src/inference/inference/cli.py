from pathlib import Path

import click
import yaml

from inference import __version__


def resolve_config(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    fallback = Path('configs') / p.name
    if fallback.exists():
        return fallback
    raise click.BadParameter(f'Config not found: {path}')


def policy_settings(cfg, model_cfg, label_lead=None):
    """Settings for the checkpoint's type and verified action layout, not its repository name.
    `label_lead`: the training dataset, if its labels already lead the TCP; then no run-time lead."""
    from inference.policies import action_layout

    layout = action_layout(model_cfg)
    kind = model_cfg['type']
    if layout == 'bimanual_absolute':
        if cfg.get('arm') != 'both' or set(cfg.get('arms', {})) != {'left', 'right'}:
            raise ValueError(
                'bimanual absolute poses without grippers require arm: both, both arm '
                'topics and a shared engage/stop gate; use configs/baxter_bimanual.yaml'
            )
        topics = ('tcp', 'target', 'trigger', 'cuff', 'home')
        for name in ('left', 'right'):
            if any(not cfg['arms'][name].get(key) for key in topics):
                raise ValueError(f'bimanual {name} arm needs {", ".join(topics)} topics')
        for key in ('tcp', 'target'):
            if cfg['arms']['left'][key] == cfg['arms']['right'][key]:
                raise ValueError(f'bimanual arms must have distinct {key} topics')
        if not all(cfg.get(key) for key in ('chunk', 'engage', 'engaged')):
            raise ValueError('bimanual runtime needs shared chunk, engage and engaged topics')
        kind = f'{kind}_bimanual'
    elif cfg.get('arm') == 'both':
        raise ValueError('arm: both requires a bimanual 28-D state / 14-D action checkpoint')
    if layout == 'absolute' and f'{kind}_absolute' in cfg.get('policies', {}):
        kind = f'{kind}_absolute'
    if kind not in cfg.get('policies', {}):
        raise click.UsageError(
            f'no inference.policies.{kind} in the config (have: {", ".join(cfg.get("policies", {}))})'
        )
    settings = cfg['policies'][kind]
    if layout in ('absolute', 'bimanual_absolute') and settings.get('rtc'):
        raise ValueError('Absolute-pose checkpoints must not use TCP-relative RTC')
    if label_lead and settings['merge'].get('lead'):
        settings = {**settings, 'merge': {k: v for k, v in settings['merge'].items() if k != 'lead'}}
    return settings


def replay_as(cfg, model_cfg, settings):
    """(runtime config, merge, absolute) to replay a recording as this checkpoint runs: its chunk length,
    a new chunk every `every` steps, its merge, and its goal layout (absolute goals or TCP-relative)."""
    from inference.policies import action_layout, chunk_length

    chunk = chunk_length({**model_cfg, **settings.get('overrides', {})})
    cfg = {**cfg, 'chunk_size': chunk, 'n_action_steps': int(settings.get('every', 1))}
    return cfg, settings['merge'], action_layout(model_cfg) == 'absolute'


def deploy_task(configured, tasks):
    """The prompt to condition on: the dataset's task the checkpoint was trained on (`tasks`), or the
    configured one if the dataset's tasks are unknown or include it. Several need one configured."""
    if not tasks or configured in tasks:
        return configured or ''
    if len(tasks) == 1:
        return tasks[0]
    raise ValueError(f'task {configured!r} is not one the checkpoint was trained on: configure one of {list(tasks)}')


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
    """RAI Inference: deploy trained policies to ROS2 robots."""
    pass


@main.command()
@click.argument('model')
@click.option('--config', default='baxter.yaml', show_default=True)
@click.option('--revision', default=None, help='Pinned Hub branch, tag or commit')
def check(model, config, revision):
    """Check deployment compatibility without loading weights, starting ROS or moving a robot."""
    from inference.policies import action_layout, chunk_length, model_config, trained_lead, trained_tasks
    from inference.runtime import check_merge

    full = yaml.safe_load(resolve_config(config).read_text())
    model_cfg = model_config(model, revision)
    label_lead = trained_lead(model, revision)
    tasks = trained_tasks(model, revision)
    try:
        settings = policy_settings(full['inference'], model_cfg, label_lead)
        layout = action_layout(model_cfg)
        task = deploy_task(full['inference'].get('task'), tasks)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    effective = {**model_cfg, **settings.get('overrides', {})}
    length, every = chunk_length(effective), settings.get('every', 1)
    horizon = min(length, int(settings['merge'].get('horizon') or length))
    if not 0 < every < horizon:
        raise click.ClickException(f'every ({every}) must be below the executable chunk length ({horizon})')
    if model_cfg['type'] == 'vita' and not 1 <= effective['n_action_steps'] <= effective['horizon']:
        raise click.ClickException('VITA n_action_steps must be within its trained horizon')
    if layout == 'absolute' and settings.get('rtc'):
        raise click.ClickException('Absolute-pose checkpoints must not use TCP-relative RTC')
    configured = {c['key'] for c in full['topics']['images']}
    missing = {k for k in model_cfg['input_features'] if k.startswith('observation.images.')} - configured
    if missing:
        raise click.ClickException(f'Missing camera topics: {", ".join(sorted(missing))}')
    click.echo(
        f'Compatible: {model_cfg["type"]}, {layout} poses, arm {full["inference"]["arm"]}; '
        f'{length} targets, predict every {every} frames, merge {settings["merge"]}; '
        + (
            f"no run-time lead: {label_lead}'s labels already lead"
            if label_lead
            else f'pose lead {settings["merge"].get("lead") or 0} s'
        )
        + f', gripper lead {settings["merge"].get("gripper_lead") or 0} s; task {task!r}'
        + ('' if tasks else " (configured: the training dataset's tasks are unreadable)")
    )
    warning = check_merge(settings['merge'], length, every, full['inference']['fps'])
    if warning:
        click.echo(f'Warning: {warning}')


def adapted(policy, adapter):
    """What a loaded adapter changes, for the log: its trainable set (ttt update's) and, for LoRA, its
    layer count per VITA module (flow_net, obs_encoder, action_decoder)."""
    from collections import Counter

    from inference import lora

    meta = lora.meta(adapter)
    trainable = meta.get('trainable', 'lora')
    if meta.get('kind') == 'weights':
        return f'{trainable}: tuned weights'
    if meta.get('kind') in lora.HEADS:
        return f'{trainable}: {meta["kind"]} head, {meta["hidden"]} hidden'
    sites = Counter(name.split('.')[1] for name in lora.adapters(policy.policy))
    return f'{trainable}: ' + ', '.join(f'{n} LoRA layers on {site}' for site, n in sites.items())


@main.command()
@click.argument('config')
@click.option('--model', required=True, help='LeRobot checkpoint: Hub repo (user/name) or local directory')
@click.option('--device', default='cuda', show_default=True)
@click.option('--revision', default=None, help='Hub branch, tag or commit of --model (e.g. step-005000)')
@click.option(
    '--adapter',
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help='Adapter (inference.lora): on VITA LoRA, tuned weights or a head; on the Multi-Task DiT LoRA',
)
@click.option(
    '--alpha', type=float, default=None, help='With --adapter: its weight, 0 the base exactly, 1 the adapted (default)'
)
@click.option(
    '--explore-sigma',
    type=float,
    default=None,
    help='VITA: plan from z0 + this times a seeded offset eps, fixed for the run (inference.explore)',
)
@click.option('--explore-seed', type=int, default=0, show_default=True, help='With --explore-sigma: the seed of eps')
@click.option(
    '--flow-noise', type=float, default=0.0, show_default=True, help='With --explore-sigma: noise per flow step'
)
@click.option(
    '--record',
    type=click.Path(dir_okay=False),
    default=None,
    help='With --explore-sigma: append every plan (latents, log-probabilities) to this file',
)
@click.option(
    '--replay',
    'episode',
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Two-arm config: serve this kinesthetic recording's chunks in place of the network, as --model runs",
)
def serve(config, model, device, revision, adapter, alpha, explore_sigma, explore_seed, flow_noise, record, episode):
    """Run a policy on the robot's topics, publishing its action chunks for `run`.

    \b
    With --replay (a two-arm config), a kinesthetic recording is served in the network's place, with
    --model's chunk length, replanning interval and merge: everything after the chunk topic is the
    chain the checkpoint runs through.

    \b
    Examples:
      inference serve baxter.yaml --model andnetdeboer/ACT_20261007-2113_ckpt010000
      inference serve baxter.yaml --model models/vita_X/step-30000 --adapter adapters/t1 --alpha 0.5
      inference serve baxter_bimanual.yaml --model models/<sweep checkpoint> --replay data/bags/<demo>
    """
    if alpha is not None and not adapter:
        raise click.UsageError('--alpha needs --adapter')
    if (flow_noise or record) and explore_sigma is None:
        raise click.UsageError('--flow-noise and --record need --explore-sigma')
    explore = None
    if explore_sigma is not None:
        explore = {'sigma': explore_sigma, 'seed': explore_seed, 'noise': flow_noise, 'record': record}
    full = yaml.safe_load(resolve_config(config).read_text())
    cfg = full['inference']
    if episode and cfg.get('arm') != 'both':
        raise click.UsageError('serve --replay is for two-arm configs; one arm: inference run --replay')

    from inference.policies import LeRobotPolicy, model_config, trained_lead, trained_tasks
    from inference.runtime import check_merge

    settings = policy_settings(cfg, model_config(model, revision), trained_lead(model, revision))
    task = deploy_task(cfg.get('task'), trained_tasks(model, revision))
    alpha = 1.0 if alpha is None else alpha
    if episode:
        from inference.policies import BimanualReplayPolicy, chunk_length

        model_cfg = model_config(model, revision)
        policy = BimanualReplayPolicy(episode, cfg['fps'], chunk_length({**model_cfg, **settings.get('overrides', {})}))
        start = {arm: [round(float(v), 3) for v in p] for arm, p in policy.start.items()}
        click.echo(
            f'Replaying {episode}: {len(policy)} steps ({len(policy) / cfg["fps"]:.1f} s), starting with the TCPs '
            f'at {start} m (base); scripts/start runs it once both arms are within '
            f'{policy.START_RADIUS * 100:.0f} cm of there'
        )
    else:
        policy = LeRobotPolicy(
            model,
            device,
            task,
            settings.get('rtc'),
            revision,
            settings.get('overrides'),
            adapter=adapter,
            alpha=alpha,
            explore=explore,
        )
    check_merge(settings['merge'], policy.chunk_size, settings.get('every', 1))
    click.echo(
        f'Serving {model} ({policy.type}, chunks of {policy.chunk_size})'
        + (f' with adapter {adapter} ({adapted(policy, adapter)}) at alpha {alpha}' if adapter else '')
        + (
            f', exploring from z0 + {explore_sigma} eps (seed {explore_seed}, flow noise {flow_noise})'
            + (f', recorded to {record}' if record else '')
            if explore
            else ''
        )
        + f' on {device}: every '
        f'{settings.get("every", 1)} step(s){", RTC" if policy.rtc else ""}, task {task!r}, chunks on {cfg["chunk"]}'
    )

    from inference.policy_node import PolicyNode, RecordingLight

    lights, servers = [], []

    def setup(node):
        if policy.bimanual:
            from inference.bimanual_policy_node import BimanualPolicyNode

            servers.append(
                BimanualPolicyNode(
                    node,
                    cfg,
                    full['topics']['images'],
                    policy,
                    settings.get('every', 1),
                    settings.get('inpaint', False),
                )
            )
        else:
            PolicyNode(node, cfg, full['topics']['images'], policy, settings.get('every', 1))
        if full.get('trigger', {}).get('light'):  # lit as in every recorded frame
            lights.append(RecordingLight(node, full['trigger']['light']))

    def shutdown():
        for server in servers:
            server.close()
        for light in lights:
            light.off()

    spin('policy_server', setup, on_exit=shutdown)


@main.command()
@click.argument('config')
@click.option(
    '--replay',
    'episode',
    type=click.Path(exists=True, file_okay=False),
    help='Play a recorded episode back as the policy',
)
@click.option('--model', default=None, help="Execute this checkpoint's chunks from `serve` (its merge settings)")
@click.option(
    '--start-at',
    type=click.Path(exists=True, file_okay=False),
    help="With --model: first move to this recorded episode's start pose (shadow rollout)",
)
@click.option(
    '--player',
    default=None,
    help='With --start-at: the paused `ros2 bag play` node (e.g. /rosbag2_player) to seek to the '
    "episode's start and resume once the arm is there",
)
@click.option('--revision', default=None, help='Hub branch, tag or commit of --model (e.g. step-005000)')
def run(config, episode, model, start_at, player, revision):
    """Run a policy on the robot, through its teleop (the config's `inference:` section).

    \b
    With both --replay and --model the recording is played as that checkpoint runs (its chunk length,
    replanning interval, merge and goal layout): the whole chain a policy runs through, without the
    network.

    \b
    Examples:
      inference run baxter.yaml --replay recorder/data/CupToContainer/episode_042
      inference run baxter.yaml --model andnetdeboer/ACT_20261007-2113_ckpt010000
      inference run baxter.yaml --replay data/bags/<kinesthetic demo> --model models/<checkpoint>
    """
    if episode is None and model is None:
        raise click.UsageError('give --replay, --model, or both')
    if start_at and not model or player and not start_at:
        raise click.UsageError('--start-at needs --model, --player needs --start-at')
    cfg = yaml.safe_load(resolve_config(config).read_text())['inference']

    from adapter.labels import load_episode
    from inference.runtime import Runtime, check_merge

    if episode:
        from inference.policies import ReplayPolicy

        if cfg.get('arm') == 'both':
            raise click.UsageError('Bimanual replay/start-pose loading is not supported; use live cameras')
        merge, absolute = None, False
        if model:
            from inference.policies import model_config, trained_lead

            model_cfg = model_config(model, revision)
            settings = policy_settings(cfg, model_cfg, trained_lead(model, revision))
            cfg, merge, absolute = replay_as(cfg, model_cfg, settings)
        policy = ReplayPolicy(episode, cfg['arm'], cfg['fps'], cfg['chunk_size'], absolute=absolute)
        click.echo(
            f'Replaying {episode}: {len(policy)} steps ({len(policy) / cfg["fps"]:.1f} s), '
            f'chunks of {cfg["chunk_size"]}'
            + (
                f' every {cfg["n_action_steps"]} steps, merge {merge}, {"absolute" if absolute else "relative"} goals'
                f' (as {model})'
                if model
                else f', ensembling {cfg.get("temporal_ensemble_coeff")}'
            )
        )
        start = policy.episode.tcp_p[0]
        click.echo(
            f'Demo starts with the TCP at {start.round(3).tolist()} m (base). Hold the deadman '
            '(trackpad) to run, release to hold; B sends the arm home.'
        )
        spin('policy_runtime', lambda node: Runtime(node, cfg, policy, merge=merge))
        return

    from inference.policies import chunk_length, model_config, trained_lead

    model_cfg = model_config(model, revision)
    settings = policy_settings(cfg, model_cfg, trained_lead(model, revision))
    if cfg.get('arm') == 'both' and start_at:
        raise click.UsageError('Bimanual shadow playback is not supported; use live cameras')
    effective = {**model_cfg, **settings.get('overrides', {})}
    warning = check_merge(settings['merge'], chunk_length(effective), settings.get('every', 1), cfg['fps'])
    if warning:
        click.echo(f'Warning: {warning}')
    cfg = {**cfg, 'chunk_size': chunk_length(effective)}
    start = None
    if start_at:
        recorded = load_episode(start_at, cfg['arm'], cfg['fps'])
        start, first = (recorded.tcp_p[0], recorded.tcp_q[0]), int(recorded.t[0])
        click.echo(
            f'Shadow rollout of {start_at}: moving to its start pose '
            f'{start[0].round(3).tolist()} m (base) first' + (f', then playing {player}' if player else '')
        )
    click.echo(
        f'Executing {model_cfg["type"]} chunks from {cfg["chunk"]}, merge {settings["merge"]}. Engage with '
        f'the deadman or `ros2 service call {cfg["engage"]} std_srvs/srv/SetBool "{{data: true}}"`.'
    )

    def setup(node):
        if cfg.get('arm') == 'both':
            from inference.bimanual_runtime import BimanualRuntime

            BimanualRuntime(node, cfg, merge=settings['merge'])
            return
        on_start = start_player(node, player, first) if player else None
        Runtime(node, cfg, merge=settings['merge'], start=start, on_start=on_start)

    spin('policy_runtime', setup)


def start_player(node, player, t_ns):
    """on_start for a shadow rollout: play the episode's cameras from its first step."""
    from rclpy.time import Time
    from rosbag2_interfaces.srv import Resume, Seek

    seek = node.create_client(Seek, f'{player}/seek')
    resume = node.create_client(Resume, f'{player}/resume')

    def go():
        if not (seek.service_is_ready() and resume.service_is_ready()):
            node.get_logger().error(f'{player} has no seek/resume services: is `ros2 bag play --start-paused` running?')
            return
        future = seek.call_async(Seek.Request(time=Time(nanoseconds=t_ns).to_msg()))
        future.add_done_callback(lambda _: resume.call_async(Resume.Request()))
        node.get_logger().info(f"Playing the episode's cameras from {player}")

    return go


@main.command()
@click.argument('model')
@click.option('--config', default='baxter.yaml', show_default=True, help="Its inference.task is the VLA's prompt")
@click.option('--device', default='cuda', show_default=True)
@click.option('--depth', type=int, default=None, help='Levels of the tree to print (default: all)')
@click.option('--json', 'out', type=click.Path(dir_okay=False), help='Also write the tree as JSON (for recap)')
def probe(model, config, device, depth, out):
    """Print a policy's module tree, traced through one inference: calls and shapes per module.

    \b
    Examples:
      inference probe models/smolvla_CupToContainer/step-030000 --depth 4
      inference probe models/vita_CupToContainer/step-060000 --json vita_tree.json
    """
    from inference import probe as probe_mod
    from inference.policies import LeRobotPolicy

    task = yaml.safe_load(resolve_config(config).read_text())['inference'].get('task', '')
    root = probe_mod.trace_policy(LeRobotPolicy(model, device, task))
    click.echo(probe_mod.text(root, depth))
    if out:
        Path(out).write_text(probe_mod.to_json(root))
        click.echo(f'\n{out}')


@main.command()
@click.argument('model')
@click.option('--config', default='baxter.yaml', show_default=True)
@click.option('--device', default='cuda', show_default=True)
def attend(model, config, device):
    """Serve attention maps for recap: JSON requests on stdin, replies on stdout (inference.attention)."""
    import sys

    from inference import attention
    from inference.policies import LeRobotPolicy

    out, sys.stdout = sys.stdout, sys.stderr  # replies only on stdout: library prints go to stderr
    full = yaml.safe_load(resolve_config(config).read_text())
    cfg = full['inference']
    policy = LeRobotPolicy(model, device, cfg.get('task', ''))
    attention.serve(policy, cfg, full['topics']['images'], sys.stdin, out)
