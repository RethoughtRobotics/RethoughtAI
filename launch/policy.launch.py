"""A policy on the arm: the policy server and the runtime, which waits for `./start`.

  ros2 launch launch/policy.launch.py model:=models/act_CupToContainer/step-100000
  ros2 launch launch/policy.launch.py model:=... episode:=recorder/data/CupToContainer/episode_004

With `episode` (a shadow rollout), the policy watches that episode's recorded cameras: the arm
first moves to the episode's start pose, then the bag plays from its first step.
Bring the arm and teleop up first (bringbackbaxter: run_sim_teleop.sh --left --policy).
"""

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def processes(context):
    model, episode, config, device, revision = (LaunchConfiguration(n).perform(context)
                                                for n in ("model", "episode", "config", "device", "revision"))
    rev = ["--revision", revision] if revision else []
    shadow = ["--start-at", episode, "--player", "/rosbag2_player"] if episode else []
    actions = [
        ExecuteProcess(cmd=["uv", "run", "inference", "serve", config, "--model", model, "--device", device, *rev],
                       name="serve", output="screen"),
        ExecuteProcess(cmd=["uv", "run", "inference", "run", config, "--model", model, *shadow, *rev],
                       name="run", output="screen"),
    ]
    if episode:
        cameras = [c["topic"] for c in yaml.safe_load(Path(config).read_text())["topics"]["images"]]
        actions.append(ExecuteProcess(cmd=["ros2", "bag", "play", episode, "--start-paused", "--topics", *cameras],
                                      name="cameras", output="screen"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model", description="LeRobot checkpoint: Hub repo or local directory"),
        DeclareLaunchArgument("episode", default_value="", description="recorded episode to shadow (bag directory)"),
        DeclareLaunchArgument("config", default_value="configs/baxter.yaml"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("revision", default_value="", description="Hub branch/tag/commit of the model"),
        # the robot stack's middleware, unless the shell sets one
        SetEnvironmentVariable("RMW_IMPLEMENTATION", EnvironmentVariable("RMW_IMPLEMENTATION", default_value="rmw_zenoh_cpp")),
        SetEnvironmentVariable("PYTHONUNBUFFERED", "1"),  # logs as they happen, not at exit
        OpaqueFunction(function=processes),
    ])
