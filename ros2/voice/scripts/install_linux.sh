#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ROBOT_WS="${ROBOT_WS:-${HOME}/wheeled_robot_pkg}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    echo "ROS2 ${ROS_DISTRO} is not installed under /opt/ros" >&2
    exit 1
fi
if [[ ! -f "${ROBOT_WS}/install/setup.bash" ]]; then
    echo "Robot SDK is not installed at ${ROBOT_WS}" >&2
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    echo "Cannot locate the Green River project root: ${PROJECT_ROOT}" >&2
    exit 1
fi

# ROS-generated setup files may inspect optional variables before defining them.
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${ROBOT_WS}/install/setup.bash"
set -u

# Keep ROS2 bindings from the system installation while isolating the complete
# application dependency set from the vendor SDK's Python environment.
bash "${PROJECT_ROOT}/scripts/install_all_linux.sh"
export RAG_SYSTEM_PYTHON="${PROJECT_ROOT}/.venv-linux/bin/python"

cd "${ROBOT_WS}/sample"
colcon build \
    --base-paths "${PROJECT_ROOT}/ros2" \
    --packages-select rag_interfaces voice \
    --cmake-clean-cache \
    --symlink-install

echo
echo "Installation completed. Start with:"
echo "  ${PROJECT_ROOT}/ros2/voice/scripts/run_voice.sh"
