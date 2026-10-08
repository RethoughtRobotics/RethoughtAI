# Sourced by run_inference, start, stop: ROS, the Baxter messages, the robot's middleware, and
# ENGAGE, the runtime's engage service (config: $CONFIG, default configs/baxter.yaml).
set +u
source /opt/ros/kilted/setup.bash
source "${BAXTER_WS:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../bringbackbaxter/ros2_ws")}/install/setup.bash"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
ENGAGE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['inference']['engage'])" \
    "${CONFIG:-$(dirname "${BASH_SOURCE[0]}")/../configs/baxter.yaml}")
