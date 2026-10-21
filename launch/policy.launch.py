"""A policy on the arm: the policy server and the runtime, which waits for `scripts/start`.

  ros2 launch launch/policy.launch.py model:=models/act_CupToContainer/step-100000
  ros2 launch launch/policy.launch.py model:=... episode:=recorder/data/CupToContainer/episode_004
  ros2 launch launch/policy.launch.py model:=models/vita_X/step-30000 adapter:=adapters/t1 alpha:=0.5
  ros2 launch launch/policy.launch.py model:=models/vita_X/step-30000 explore_sigma:=0.05 record:=p.npy

With `episode` (a shadow rollout), the policy watches that episode's recorded cameras: the arm
first moves to the episode's start pose, then the bag plays from its first step.
Bring the arm and teleop up first (BaxterSDK: scripts/run_sim_teleop.sh --left --policy).
"""

from pathlib import Path

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def processes(context):
    model, episode, config, device, revision, adapter, alpha = (
        LaunchConfiguration(n).perform(context)
        for n in ('model', 'episode', 'config', 'device', 'revision', 'adapter', 'alpha')
    )
    sigma, seed, noise, record = (
        LaunchConfiguration(n).perform(context) for n in ('explore_sigma', 'explore_seed', 'flow_noise', 'record')
    )
    rev = ['--revision', revision] if revision else []
    lora = (['--adapter', adapter] if adapter else []) + (['--alpha', alpha] if alpha else [])
    if sigma:  # seeded exploration (inference.explore)
        lora += ['--explore-sigma', sigma, '--explore-seed', seed or '0']
        lora += (['--flow-noise', noise] if noise else []) + (['--record', record] if record else [])
    shadow = ['--start-at', episode, '--player', '/rosbag2_player'] if episode else []
    actions = [
        # Respawned if it dies (e.g. it exits on a lost GPU context, bimanual_policy_node.device_lost):
        # the runtime holds the arm and resumes on the new server's chunks.
        ExecuteProcess(
            cmd=['uv', 'run', 'inference', 'serve', config, '--model', model, '--device', device, *rev, *lora],
            name='serve',
            output='screen',
            respawn=True,
            respawn_delay=1.0,
        ),
        ExecuteProcess(
            cmd=['uv', 'run', 'inference', 'run', config, '--model', model, *shadow, *rev], name='run', output='screen'
        ),
    ]
    if episode:
        cameras = [c['topic'] for c in yaml.safe_load(Path(config).read_text())['topics']['images']]
        actions.append(
            ExecuteProcess(
                cmd=['ros2', 'bag', 'play', episode, '--start-paused', '--topics', *cameras],
                name='cameras',
                output='screen',
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', description='LeRobot checkpoint: Hub repo or local directory'),
        DeclareLaunchArgument('episode', default_value='', description='recorded episode to shadow (bag directory)'),
        DeclareLaunchArgument('config', default_value='configs/baxter.yaml'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('revision', default_value='', description='Hub branch/tag/commit of the model'),
        DeclareLaunchArgument(
            'adapter', default_value='', description='VITA adapter directory (inference.lora): LoRA or tuned weights'
        ),
        DeclareLaunchArgument(
            'alpha', default_value='', description="the adapter's weight: 0 the base, 1 (default) adapted"
        ),
        DeclareLaunchArgument(
            'explore_sigma', default_value='', description='VITA: plan from z0 + sigma eps (inference.explore)'
        ),
        DeclareLaunchArgument('explore_seed', default_value='0', description='with explore_sigma: the seed of eps'),
        DeclareLaunchArgument('flow_noise', default_value='', description='with explore_sigma: noise per flow step'),
        DeclareLaunchArgument('record', default_value='', description='with explore_sigma: file every plan goes to'),
        # the robot stack's middleware, unless the shell sets one
        SetEnvironmentVariable(
            'RMW_IMPLEMENTATION', EnvironmentVariable('RMW_IMPLEMENTATION', default_value='rmw_zenoh_cpp')
        ),
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),  # logs as they happen, not at exit
        OpaqueFunction(function=processes),
    ])
