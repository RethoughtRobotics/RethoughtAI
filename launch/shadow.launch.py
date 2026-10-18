"""Shadow rollout: a policy drives the arm while watching a recorded episode's cameras.

  ros2 launch launch/shadow.launch.py model:=USER/MODEL episode:=recorder/data/SESSION/episode_004

Bring the arm and teleop up first (bringbackbaxter: run_sim_teleop.sh --left --policy --objects),
then engage: ros2 service call /policy/left/engage std_srvs/srv/SetBool "{data: true}"
The arm moves to the episode's start pose, then the bag plays from the episode's first step.
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
    cameras = [c["topic"] for c in yaml.safe_load(Path(config).read_text())["topics"]["images"]]
    return [
        ExecuteProcess(cmd=["uv", "run", "inference", "serve", config, "--model", model, "--device", device, *rev],
                       name="serve", output="screen"),
        ExecuteProcess(cmd=["uv", "run", "inference", "run", config, "--model", model,
                            "--start-at", episode, "--player", "/rosbag2_player", *rev],
                       name="run", output="screen"),
        ExecuteProcess(cmd=["ros2", "bag", "play", episode, "--start-paused", "--topics", *cameras],
                       name="cameras", output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model", description="LeRobot checkpoint: Hub repo or local directory"),
        DeclareLaunchArgument("episode", description="recorded episode (bag directory) to shadow"),
        DeclareLaunchArgument("config", default_value="configs/baxter.yaml"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("revision", default_value="", description="Hub branch/tag/commit of the model"),
        # the robot stack's middleware, unless the shell sets one
        SetEnvironmentVariable("RMW_IMPLEMENTATION", EnvironmentVariable("RMW_IMPLEMENTATION", default_value="rmw_zenoh_cpp")),
        OpaqueFunction(function=processes),
    ])
