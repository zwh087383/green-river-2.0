#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot_ws="${ROBOT_WS:-$HOME/wheeled_robot_pkg}"
[[ -f "$robot_ws/install/include/arm_motion.hpp" ]] || { echo 'Vendor ArmMotion headers missing.' >&2; exit 1; }
[[ -f /opt/ros/humble/setup.bash ]] || { echo 'Motion plugin requires ROS Humble; /opt/ros/humble/setup.bash missing.' >&2; exit 1; }
[[ "${ROS_DISTRO:-humble}" == humble ]] || { echo 'Build in a clean ROS Humble environment.' >&2; exit 1; }
python3 "$project_root/scripts/check_motion_sdk.py" --robot-ws "$robot_ws" --inventory-only
set +u
source /opt/ros/humble/setup.bash
source "$robot_ws/install/setup.bash"
set -u
cd "$project_root"
colcon build --base-paths native/motion --packages-select greenriver_native_motion \
    --build-base ros2-build --install-base ros2-install --event-handlers console_direct+ \
    --cmake-args -DROBOT_WS="$robot_ws" -DCMAKE_BUILD_TYPE=RelWithDebInfo
