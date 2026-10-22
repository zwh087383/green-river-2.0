#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot_ws="${ROBOT_WS:-$HOME/wheeled_robot_pkg}"
worker="$project_root/ros2-install/greenriver_native_motion/lib/libgreenriver_native_motion.so"
[[ -f "$worker" ]] || { echo 'Run scripts/build_sdk_motion.sh first.' >&2; exit 1; }
[[ -f /opt/ros/humble/setup.bash ]] || { echo 'Motion plugin requires ROS Humble.' >&2; exit 1; }
[[ "${ROS_DISTRO:-humble}" == humble ]] || { echo 'Start in a clean ROS Humble environment.' >&2; exit 1; }
# Reject mismatched SDK/config before stopping any existing device owner.
python3 "$project_root/scripts/check_motion_sdk.py" --robot-ws "$robot_ws" \
    --verify-manifest "$project_root/ros2-install/greenriver_native_motion/share/greenriver_native_motion/sdk-contract.json"
export GREENRIVER_MOTION_SOCKET="${GREENRIVER_MOTION_SOCKET:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/greenriver-motion.sock}"
# Serialize takeover as well as ownership; two launch attempts cannot kill or
# replace each other's native process. Lock remains inherited by the worker.
exec 9>"${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/greenriver-motion-launch.lock"
flock -n 9 || { echo 'Native SDK motion launcher already active.' >&2; exit 1; }
if [[ "${1:-}" == --takeover ]]; then
    # Graceful stop of the known vendor arm node only. Chassis and trunk remain.
    # The bundled vendor launch does not respawn this node. Refuse to force-kill
    # or start a second device owner if it does not exit.
    python3 - "$robot_ws" <<'PY'
import os, signal, sys, time
from pathlib import Path
expected = str(Path(sys.argv[1]).resolve() / 'install/lib/arm_control/arm_control_node')
processes = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        command = (p/'cmdline').read_bytes().split(b'\0')[0].decode()
        if command == expected:
            processes.append(int(p.name))
    except (OSError, UnicodeError):
        pass
for pid in processes:
    print(f'Stopping vendor arm controller {pid}; native SDK will take ownership.', flush=True)
    try: os.kill(pid, signal.SIGINT)
    except ProcessLookupError: pass
end = time.monotonic() + 10
while any(Path(f'/proc/{pid}').exists() for pid in processes):
    if time.monotonic() >= end:
        raise SystemExit('Vendor arm controller did not stop; aborting takeover without force-kill.')
    time.sleep(.1)
PY
fi
set +u
source /opt/ros/humble/setup.bash
source "$robot_ws/install/setup.bash"
set -u
set +u
source "$project_root/ros2-install/setup.bash"
set -u
cd "$robot_ws"
exec /usr/bin/python3 "$project_root/scripts/sdk_motion_host.py" \
    --project-root "$project_root" --robot-ws "$robot_ws"
