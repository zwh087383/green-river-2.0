#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ROBOT_WS="${ROBOT_WS:-${HOME}/wheeled_robot_pkg}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
FULL_PYTHON="${RAG_SYSTEM_PYTHON:-${PROJECT_ROOT}/.venv-linux/bin/python}"

if [[ ! -f "${ROBOT_WS}/sample/install/setup.bash" ]]; then
    echo "Voice overlay is not built. Run install_linux.sh first." >&2
    exit 1
fi
if [[ ! -x "${FULL_PYTHON}" ]]; then
    echo "Full Linux environment is missing: ${FULL_PYTHON}" >&2
    echo "Run ${PROJECT_ROOT}/scripts/install_all_linux.sh first." >&2
    exit 1
fi

# ROS-generated setup files may inspect optional variables before defining them.
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${ROBOT_WS}/install/setup.bash"
source "${ROBOT_WS}/sample/install/setup.bash"
set -u

export RAG_DATA_DIR="${RAG_DATA_DIR:-${PROJECT_ROOT}/data}"
export RAG_SYSTEM_PYTHON="${FULL_PYTHON}"
mkdir -p "${RAG_DATA_DIR}"

exec ros2 launch voice "${RAG_VOICE_LAUNCH_FILE:-voice_rag.launch.py}" data_dir:="${RAG_DATA_DIR}" "$@"
