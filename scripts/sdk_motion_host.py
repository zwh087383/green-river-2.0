"""Start the stock arm driver and load the native SDK socket controller once.

ROS controller services are used only during process startup. Conversation-time
requests go over the Unix socket and call ArmMotion directly inside this host.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import signal
import socket
import subprocess
import time
import threading
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def forward_arm_log(line: str, path: str) -> bool:
    """Forward only genuine arm-driver log records; never a movement command."""
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    match = re.match(r'^\[(INFO|WARN|ERROR|FATAL)\] \[(\d+)\.(\d+)\] '
                     r'\[((?:left|right)_arm\.(?:hardware_interface|arm_controller))\]: (.*)', line)
    if not match:
        return False
    level, seconds, fraction, name, reason = match.groups()
    stamp = int(seconds) * 1_000_000_000 + int(fraction[:9].ljust(9, '0'))
    reason = re.sub(r'\s+', ' ', reason).strip()
    code = re.search(r'\([0-9]+-k[A-Za-z0-9_]+\)', reason)
    if code and len(reason.encode('utf-8')) > 112:
        reason = reason[:60] + ' ... ' + code.group()
    encoded = reason.encode('utf-8')[:112].hex()
    command = f'DRIVER_LOG {stamp} {dict(INFO=20,WARN=30,ERROR=40,FATAL=50)[level]} {name} {encoded}\n'
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(.3)
            stream.connect(path)
            stream.sendall(command.encode('ascii'))
            data = bytearray()
            while b'\n' not in data and len(data) <= 8192:
                part = stream.recv(4096)
                if not part:
                    return False
                data.extend(part)
            return json.loads(data.split(b'\n', 1)[0]).get('status') == 'ok'
    except (OSError, ValueError):
        return False


def forward_sdk_output(stream, path: str) -> None:
    confirmed = False
    for line in stream:
        # One successful INFO record verifies the path without forwarding all
        # normal logging. Every ERROR/FATAL record is forwarded immediately.
        error = '[ERROR]' in line or '[FATAL]' in line
        if error or not confirmed:
            delivered = forward_arm_log(line, path)
            confirmed = confirmed or delivered
        sys.stdout.write(line)
        sys.stdout.flush()


def prepare_controller_config(sdk: Path, side: str, output: Path) -> Path:
    """Generate the vendor URDF for a temporary per-arm description publisher.

    Do not give robot_description directly to controller_manager: the vendor
    registers each manager only after its constructor returns. Hardware loading
    must happen later, through the description topic used by the vendor launch.
    """
    if side not in ("left", "right"):
        raise ValueError("Expected left or right arm")
    source = sdk / f"install/config/controller/{side}_arm_controller.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    output.write_text(yaml.dump(config, Dumper=NoAliasDumper), encoding="utf-8")
    xacro = sdk / "install/share/robot_description/urdf/wheel_robot_control.xacro"
    result = subprocess.run([
        "xacro", str(xacro), f"{side}_arm_controller:={output}",
        f"{side}_arm_plugin:=nerv/arm_hw/ArmRokaeAr5",
    ], check=True, capture_output=True, text=True, timeout=20)
    description = result.stdout
    controls = ET.fromstring(description).findall("ros2_control")
    expected_joints = config[f"/{side}_arm/info"]["ros__parameters"]["arm_joints"]
    if (len(controls) != 1 or controls[0].get("name") != f"{side}_arm"
            or controls[0].get("type") != "system"
            or controls[0].findtext("hardware/plugin") != "nerv/arm_hw/ArmRokaeAr5"
            or [joint.get("name") for joint in controls[0].findall("joint")] != expected_joints):
        raise ValueError(f"Unexpected {side} arm control description")
    description_file = output.with_name(output.stem + '-description.yaml')
    description_file.write_text(yaml.safe_dump({
        '/**': {'ros__parameters': {'robot_description': description}},
    }), encoding="utf-8")
    return description_file


def stop_description_publishers(publishers: list[subprocess.Popen]) -> None:
    for publisher in publishers:
        if publisher.poll() is None:
            publisher.terminate()
    for publisher in publishers:
        try:
            publisher.wait(timeout=3)
        except subprocess.TimeoutExpired:
            publisher.kill()
            publisher.wait(timeout=2)
    publishers.clear()


def preset_target_environment(sdk: Path) -> str:
    """Read the SDK's A/PresetPosition configuration without changing its pose."""
    config = json.loads((sdk/'install/product/config/arm_motion.json').read_text())
    targets = {}
    for step in config.get('A', []):
        for side in ('left_arm', 'right_arm'):
            if side in step:
                targets[side] = step[side]
    values = []
    for side in ('left_arm', 'right_arm'):
        pose = targets.get(side, [])
        if len(pose) != 7 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in pose):
            raise ValueError(f'Invalid SDK preset joint positions: {side}')
        values.extend(math.radians(v) for v in pose)
    return ' '.join(format(v, '.17g') for v in values)


def joint_limit_environment(sdk: Path) -> str:
    model = ET.parse(sdk / 'install/share/robot_description/urdf/wheel_robot.urdf')
    values = []
    for side in ('L', 'R'):
        for index in range(1, 8):
            limit = model.find(f".//joint[@name='AR5_{side}joint_{index}']/limit")
            if limit is None:
                raise ValueError('Vendor joint limits missing')
            low, high = float(limit.attrib['lower']), float(limit.attrib['upper'])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError('Invalid vendor joint limits')
            values.extend((low, high))
    return ' '.join(format(v, '.17g') for v in values)


def trunk_target_environment(sdk: Path) -> str:
    """Use the teaching body pose with upright pitch for the installed neck padding."""
    config = yaml.safe_load((sdk / 'teach_script/teach_arm_config_init_pose.yaml').read_text(encoding='utf-8'))
    sequence = config.get('arm_sequence') if isinstance(config, dict) else None
    if (not isinstance(sequence, list) or len(sequence) != 1
            or not isinstance(sequence[0], dict)):
        raise ValueError('Invalid SDK initial pose sequence')
    targets = sequence[0].get('trunk_position')
    if (not isinstance(targets, list) or len(targets) != 6
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in targets)):
        raise ValueError('Invalid SDK initial trunk pose')
    model=ET.parse(sdk/'install/share/robot_description/urdf/wheel_robot.urdf')
    for name,value in zip(('trunk1_joint','trunk2_joint','trunk3_joint','trunk4_joint','head1_joint','head2_joint'),targets):
        limit=model.find(f".//joint[@name='{name}']/limit")
        if limit is None or not float(limit.attrib['lower'])<=math.radians(value)<=float(limit.attrib['upper']):
            raise ValueError('Trunk home outside vendor limits: '+name)
    # The operator has installed neck padding that obstructs the teaching -22°
    # pitch. Keep the head upright rather than driving into that obstruction.
    targets[-1] = 0.
    pitch_limit = model.find(".//joint[@name='head2_joint']/limit")
    if not float(pitch_limit.attrib['lower']) <= 0. <= float(pitch_limit.attrib['upper']):
        raise ValueError('Upright head pitch outside vendor limits')
    return ' '.join(format(v,'.17g') for v in targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--robot-ws', type=Path, required=True)
    args = parser.parse_args()
    root, sdk = args.project_root, args.robot_ws
    from check_motion_sdk import verify_manifest
    verify_manifest(sdk, root/'ros2-install/greenriver_native_motion/share/greenriver_native_motion/sdk-contract.json')
    from trunk_owner_guard import verify_guard
    verify_guard(root,sdk)
    runtime = root/'data/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    configs = []
    descriptions = []
    # rcl parameter files do not accept YAML aliases; follow the vendor launch's
    # dereference step instead of changing the supplied controller settings.
    for side in ('left', 'right'):
        output = runtime/f'native-{side}-arm-controller.yaml'
        descriptions.append(prepare_controller_config(sdk, side, output))
        configs.append(output)
    plugin = runtime/'native-motion-controller.yaml'
    plugin.write_text(yaml.safe_dump({'/left_arm/controller_manager': {'ros__parameters': {
        'greenriver_native_motion': {'type': 'greenriver/NativePresetController'}}}}))
    command = [str(sdk/'install/lib/arm_control/arm_control_node'), '--ros-args',
               '-r', '__node:=ArmControlNode', '-r', '__ns:=/',
               '-r', '/arm_control/motion:=/greenriver_native/unused/arm_motion',
               '-r', '/arm_control/connect:=/greenriver_native/unused/arm_connect',
               '--params-file', str(sdk/'install/config/arm_control_config.yaml')]
    for path in [*configs, plugin]:
        command.extend(['--params-file', str(path)])
    host = subprocess.Popen(command, cwd=sdk, env=dict(os.environ,
                            GREENRIVER_PRESET_TARGETS=preset_target_environment(sdk),
                            GREENRIVER_JOINT_LIMITS=joint_limit_environment(sdk),
                            GREENRIVER_TRUNK_HOME_TARGETS=trunk_target_environment(sdk),
                            GREENRIVER_MOTION_PRESETS=str(root/'config/motion_presets.json')),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors='replace', bufsize=1)
    log_reader = threading.Thread(target=forward_sdk_output,
                                  args=(host.stdout, os.environ['GREENRIVER_MOTION_SOCKET']),
                                  daemon=True, name='sdk-driver-errors')
    log_reader.start()
    stopping = False
    publishers = []
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if host.poll() is None:
            host.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f'[motion-host] Stock arm SDK host PID {host.pid}', flush=True)
    def run_startup(command):
        process = subprocess.Popen(command)
        deadline = time.monotonic() + 40
        try:
            while process.poll() is None:
                if stopping or host.poll() is not None or any(p.poll() is not None for p in publishers):
                    raise RuntimeError('Arm SDK host stopped during controller setup')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Controller startup timed out')
                time.sleep(.1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=2)

    try:
        # These publishers carry the same per-arm model as the vendor launch.
        # Topic delivery defers hardware initialization until the host spins.
        for side, description in zip(('left', 'right'), descriptions):
            publishers.append(subprocess.Popen([
                str(sdk/'install/lib/wheeled_robot/urdf_publisher'), '--ros-args',
                '-r', '__node:=control_urdf_publisher',
                '-r', f'__ns:=/{side}_arm/controller_manager',
                '--params-file', str(description),
            ], cwd=sdk))
        for side in ('left', 'right'):
            for controllers, inactive in [(['joint_state_broadcaster'], False),
                 (['follow_position_controller', 'follow_velocity_controller',
                   'follow_servoj_controller', 'arm_controller'], True)]:
                if stopping or host.poll() is not None:
                    raise RuntimeError('Arm SDK host stopped during startup')
                spawn = ['ros2', 'run', 'controller_manager', 'spawner', *controllers,
                         '-c', f'/{side}_arm/controller_manager',
                         '--controller-manager-timeout', '20', '--service-call-timeout', '10',
                         '--switch-timeout', '20']
                if inactive:
                    spawn.append('--inactive')
                run_startup(spawn)
        run_startup(['ros2', 'run', 'controller_manager', 'spawner', 'greenriver_native_motion',
                        '-c', '/left_arm/controller_manager', '--controller-manager-timeout', '20',
                        '--service-call-timeout', '10', '--switch-timeout', '20'])
        # Both managers have loaded their models; no external model process is
        # required for normal operation. Always clean these helpers on failure.
        stop_description_publishers(publishers)
        previous = None
        while not stopping:
            if host.poll() is not None:
                raise RuntimeError(f'Arm SDK host exited: {host.returncode}')
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
                    stream.settimeout(.3)
                    stream.connect(os.environ['GREENRIVER_MOTION_SOCKET'])
                    stream.sendall(b'STATUS\n')
                    result = json.loads(stream.recv(4096).split(b'\n')[0])
                state = (result.get('state'), result.get('detail'))
                if state != previous:
                    print('[motion-host] ' + json.dumps(result, ensure_ascii=False), flush=True)
                    previous = state
            except (OSError, ValueError) as exc:
                if previous != ('unavailable', str(exc)):
                    print(f'[motion-host] Socket unavailable: {exc}', flush=True)
                    previous = ('unavailable', str(exc))
            time.sleep(1)
    finally:
        stop_description_publishers(publishers)
        if host.poll() is None:
            host.send_signal(signal.SIGINT)
            try:
                host.wait(timeout=8)
            except subprocess.TimeoutExpired:
                host.terminate()
                try:
                    host.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # A stuck SDK is contained in this dedicated arm host. Do not
                    # leave it behind and start another device owner on restart.
                    host.kill()
                    host.wait(timeout=2)
        log_reader.join(timeout=1)


if __name__ == '__main__':
    main()
