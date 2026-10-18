# Sourced by run_inference, start, stop, home: ROS, the Baxter messages, the robot's middleware,
# ENGAGE, the runtime's engage service, and HOME_TOPIC, the teleop's home button the runtime also
# listens to (config: $CONFIG, default configs/baxter.yaml).
set +u
source /opt/ros/kilted/setup.bash
source "${BAXTER_WS:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../bringbackbaxter/ros2_ws")}/install/setup.bash"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
read -r ENGAGE HOME_TOPIC HOME_TOPIC_OTHER < <(python3 -c "import yaml,sys; c = yaml.safe_load(open(sys.argv[1]))['inference']; homes = list(dict.fromkeys(a['home'] for a in c['arms'].values())) if c.get('arm') == 'both' else [c['home']]; print(c['engage'], *homes)" \
    "${CONFIG:-$(dirname "${BASH_SOURCE[0]}")/../configs/baxter.yaml}")
