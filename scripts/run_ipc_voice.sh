#!/usr/bin/env bash
# Run the 4.0 overlay without replacing the vendor SDK or the old Voice checkout.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# All robot entrypoints go through one supervisor. The internal child performs
# the existing environment setup and ROS launch after old-process cleanup.
if [[ "${1:-}" != --runtime-child ]]; then
    exec /usr/bin/python3 "$project_root/scripts/robot_runtime.py" start --project-root "$project_root" --legacy-root "$HOME/Voice" -- "$@"
fi
shift
robot_ws="${ROBOT_WS:-$HOME/wheeled_robot_pkg}"
ros_distro="${ROS_DISTRO:-humble}"
if [[ ! -f "$project_root/ros2-install/setup.bash" ]]; then
    echo "缺少本项目 ROS2 overlay：$project_root/ros2-install/setup.bash" >&2
    exit 1
fi
set +u
source "/opt/ros/$ros_distro/setup.bash"
source "$robot_ws/install/setup.bash"
source "$project_root/ros2-install/setup.bash"
set -u
# The industrial PC's existing arm controller uses loopback-only DDS discovery.
# Apply after the SDK setup, which otherwise exports ROS_LOCALHOST_ONLY=0.
export ROS_LOCALHOST_ONLY="${RAG_ROS_LOCALHOST_ONLY:-1}"
export GREENRIVER_PROJECT_ROOT="$project_root"
export GREENRIVER_MOTION_SOCKET="${GREENRIVER_MOTION_SOCKET:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/greenriver-motion.sock}"
export RAG_SYSTEM_PYTHON="$project_root/.venv-linux/bin/python"
export RAG_DATA_DIR="${RAG_DATA_DIR:-$project_root/data}"
export RAG_SYSTEM_PORT="${RAG_SYSTEM_PORT:-8001}"
export RAG_ROBOT_WAKE_SOURCE="${RAG_ROBOT_WAKE_SOURCE:-hardware}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export TOKENIZERS_PARALLELISM=false
if [[ ! -f "$project_root/ros2-install/greenriver_native_motion/lib/libgreenriver_native_motion.so" ]]; then
    "$project_root/scripts/build_sdk_motion.sh"
fi
"$project_root/scripts/start_linux.sh"
cd "$project_root"
exec ros2 launch voice voice_realtime.launch.py \
    data_dir:="$RAG_DATA_DIR" backend_url:="http://127.0.0.1:$RAG_SYSTEM_PORT" "$@"
