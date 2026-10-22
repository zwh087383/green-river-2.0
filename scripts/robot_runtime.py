"""One complete robot runtime, including detached and orphaned project processes.

Only this user's Green River processes are eligible. PID files are hints, never
kill authority; every signal checks the Linux process start time to avoid PID reuse.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

SESSION_KEYS = ('DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS',
                'XDG_SESSION_TYPE', 'WAYLAND_DISPLAY')
VOICE_ROLES = {
    'realtime_gateway_node': 'voice_gateway', 'display_node': 'display',
    'knowledge_node': 'knowledge', 'arm_motion_node': 'old_motion_adapter',
    'audio_input_node': 'old_audio_input', 'bridge_node': 'old_voice_bridge',
    'answer_node': 'old_answer',
}


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    start: int
    state: str
    uid: int
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str]

    @property
    def command(self) -> str:
        return ' '.join(self.argv)


def read_process(pid: int, proc_root: Path = Path('/proc')) -> Process | None:
    try:
        p = proc_root/str(pid)
        tail = (p/'stat').read_text().rpartition(') ')[2].split()
        args = tuple(x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x)
        # Chromium rewrites child argv into one display string on Linux.
        if len(args) == 1 and args[0].startswith('/opt/google/chrome/chrome '):
            args = tuple(shlex.split(args[0]))
        env = {}
        try:
            for item in (p/'environ').read_bytes().split(b'\0'):
                key, sep, value = item.partition(b'=')
                if sep and key.decode(errors='replace') in (*SESSION_KEYS, 'RAG_DATA_DIR',
                        'RAG_SYSTEM_PYTHON', 'GREENRIVER_PROJECT_ROOT', 'GREENRIVER_INSTANCE_ID'):
                    env[key.decode()] = value.decode(errors='replace')
        except OSError:
            pass
        try: cwd = str((p/'cwd').readlink())
        except OSError: cwd = ''
        return Process(pid, int(tail[1]), int(tail[19]), tail[0], p.stat().st_uid, args, cwd, env)
    except (OSError, ValueError, IndexError):
        return None


def snapshot(proc_root: Path = Path('/proc')) -> dict[int, Process]:
    result = {}
    for entry in proc_root.iterdir():
        if entry.name.isdigit():
            p = read_process(int(entry.name), proc_root)
            if p and p.state != 'Z' and p.argv:
                result[p.pid] = p
    return result


def under(value: str, roots: tuple[Path, ...]) -> bool:
    return any(value == str(root) or value.startswith(str(root) + '/') for root in roots)


def role(p: Process, roots: tuple[Path, ...], uid: int | None = None) -> str | None:
    if p.uid != (os.getuid() if uid is None else uid) or not p.argv:
        return None
    ours = under(p.cwd, roots) or any(under(arg, roots) for arg in p.argv) or any(
        under(p.env.get(k, ''), roots) for k in ('RAG_DATA_DIR', 'RAG_SYSTEM_PYTHON', 'GREENRIVER_PROJECT_ROOT'))
    cmd = p.command
    profile = re.search(r'--user-data-dir=([^\s]+)', cmd)
    if profile and under(profile.group(1), roots) and 'display-browser-profile' in profile.group(1):
        return 'browser_child' if re.search(r'(?:^|\s)--type=', cmd) else 'kiosk'
    if not ours:
        return None
    if '-m' in p.argv:
        index = p.argv.index('-m')
        module = p.argv[index + 1] if index + 1 < len(p.argv) else ''
        if module == 'rag_system.web': return 'web'
        if module == 'rag_system.indexing.embedding_server': return 'embedding'
        if module.startswith('voice.'):
            return VOICE_ROLES.get(module[6:])
    code = p.argv[p.argv.index('-c')+1] if '-c' in p.argv and p.argv.index('-c')+1 < len(p.argv) else cmd
    for module, kind in VOICE_ROLES.items():
        if re.search(r'(?:^|;)\s*from voice\.' + re.escape(module) + r' import main\s*;\s*main\(', code):
            return kind
    names = {Path(arg).name for arg in p.argv if not arg.startswith('-')}
    if 'robot_runtime.py' in names: return 'supervisor'
    if 'sdk_motion_host.py' in names: return 'motion_supervisor'
    if Path(p.argv[0]).name == 'arm_control_node': return 'arm_sdk_host'
    if Path(p.argv[0]).name == 'sdk_motion_worker': return 'obsolete_motion_worker'
    if 'voice_realtime.launch.py' in p.argv or 'voice_rag.launch.py' in p.argv:
        return 'voice_launcher'
    entries = {'realtime_voice_gateway': 'voice_gateway', 'display_node': 'display',
               'rag_knowledge_node': 'knowledge', 'voice_arm_motion_node': 'old_motion_adapter',
               'voice_bridge_node': 'old_voice_bridge', 'rag_answer_node': 'old_answer'}
    if Path(p.argv[0]).name in entries: return entries[Path(p.argv[0]).name]
    if names & {'run_ipc_voice.sh', 'run_full_voice.sh', 'run_voice.sh', 'start_linux.sh',
                'start_embedding_linux.sh', 'run_sdk_motion.sh'}:
        return 'bootstrap'
    return None


def ancestors(pid: int, processes: dict[int, Process]) -> set[int]:
    result = {pid}
    while pid in processes and processes[pid].ppid not in result:
        pid = processes[pid].ppid
        if pid <= 1: break
        result.add(pid)
    return result


def select_processes(processes: dict[int, Process], roots: tuple[Path, ...],
                     protected: set[int], instance: str | None = None) -> dict[int, Process]:
    selected = {pid: p for pid, p in processes.items() if pid not in protected and role(p, roots)
                and (instance is None or p.env.get('GREENRIVER_INSTANCE_ID') == instance)}
    # Include helpers that have no project path in argv, such as ROS spawners.
    changed = True
    while changed:
        changed = False
        for pid, p in processes.items():
            if pid not in protected and pid not in selected and p.uid == os.getuid() and p.ppid in selected:
                selected[pid] = p; changed = True
    return selected


def same_process(p: Process) -> bool:
    current = read_process(p.pid)
    return current is not None and current.state != 'Z' and current.start == p.start and current.uid == p.uid


def signal_process(p: Process, sig: int) -> None:
    if same_process(p):
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            try: handle = os.pidfd_open(p.pid)
            except ProcessLookupError: return
            try:
                if same_process(p): signal.pidfd_send_signal(handle, sig)
            except ProcessLookupError: pass
            finally: os.close(handle)
        else:
            try: os.kill(p.pid, sig)
            except ProcessLookupError: pass


def terminate(processes: dict[int, Process], grace: float = 10) -> None:
    # Signal all selected identities so orphaned rclpy nodes are not left to the
    # launch parent's shutdown behavior. Unrelated processes never receive signals.
    for p in processes.values(): signal_process(p, signal.SIGTERM)
    deadline = time.monotonic() + grace
    remaining = list(processes.values())
    while remaining and time.monotonic() < deadline:
        remaining = [p for p in remaining if same_process(p)]
        if remaining: time.sleep(.1)
    for p in remaining:
        print(f'[runtime] Force stopping residual PID {p.pid}', flush=True)
        signal_process(p, signal.SIGKILL)
    deadline = time.monotonic() + 3
    while any(same_process(p) for p in remaining):
        if time.monotonic() > deadline: raise RuntimeError('Old project processes did not exit; refusing a second runtime')
        time.sleep(.05)


def desktop_environment(processes: dict[int, Process]) -> dict[str, str]:
    # Chromium may erase /proc/environ. SSH also has no DISPLAY; recover the
    # actual logged-in desktop session rather than assuming an X display number.
    desktops = {'gnome-shell', 'gnome-session-binary', 'plasmashell', 'xfce4-session', 'sway', 'weston'}
    for p in processes.values():
        if p.uid == os.getuid() and p.argv and Path(p.argv[0]).name in desktops:
            if p.env.get('DISPLAY') or p.env.get('WAYLAND_DISPLAY'):
                return {k: p.env[k] for k in SESSION_KEYS if k in p.env}
    return {}


def save_browser_session(processes: dict[int, Process], roots: tuple[Path, ...], file: Path,
                         all_processes: dict[int, Process] | None = None) -> dict | None:
    main = next((p for p in processes.values() if role(p, roots) == 'kiosk'), None)
    if main:
        config = {'argv': list(main.argv), 'env': {k: main.env[k] for k in SESSION_KEYS if k in main.env}}
    elif file.exists():
        config = json.loads(file.read_text())
    else:
        return None
    config['env'] = {**desktop_environment(all_processes if all_processes is not None else processes),
                     **{k: os.environ[k] for k in SESSION_KEYS if k in os.environ}, **config['env']}
    file.write_text(json.dumps(config, ensure_ascii=False, indent=2)); file.chmod(0o600)
    return config


def counts(processes: dict[int, Process], roots: tuple[Path, ...]) -> dict[str, int]:
    return dict(Counter(kind for p in processes.values() if (kind := role(p, roots))))


def lock_file(path: Path, timeout: float = 180):
    handle = path.open('a')
    deadline = time.monotonic() + timeout
    while True:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); return handle
        except BlockingIOError:
            if time.monotonic() > deadline:
                handle.close(); raise TimeoutError('Another startup is still in progress')
            time.sleep(.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--legacy-root', type=Path, action='append', default=[])
    command_line = sys.argv[1:]
    if '--' in command_line:
        split = command_line.index('--')
        launch_args = command_line[split+1:]
        command_line = command_line[:split]
    else:
        launch_args = []
    args = parser.parse_args(command_line)
    root = args.project_root.resolve()
    # Old Voice checkouts on this robot were earlier deployments of the same app.
    # Source files are never changed by process cleanup.
    roots = (root, *(path.resolve() for path in args.legacy_root))
    runtime = Path(os.environ.get('RAG_DATA_DIR', str(root/'data')))/'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    initial = snapshot(); protected = ancestors(os.getpid(), initial)
    if args.action == 'status':
        report = {str(p.pid): role(p, roots) for p in initial.values() if role(p, roots) and p.pid not in protected}
        print(json.dumps({'counts': counts({p.pid: p for p in initial.values() if p.pid not in protected}, roots),
                          'processes': report}, ensure_ascii=False, indent=2)); return
    if args.action == 'start':
        for required in (root/'.venv-linux/bin/python', root/'ros2-install/setup.bash', root/'scripts/run_ipc_voice.sh'):
            if not required.exists(): raise RuntimeError(f'Missing runtime prerequisite: {required}')
    instance = uuid.uuid4().hex
    owner = startup = None
    child = browser = None
    stopping = False
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        startup = lock_file(runtime/'startup.lock')
        current = snapshot()
        old = select_processes(current, roots, protected)
        browser_config = save_browser_session(old, roots, runtime/'display-session.json', current)
        print('[runtime] Cleaning old project processes: ' + json.dumps(counts(old, roots)), flush=True)
        terminate(old)
        # Detect an external supervisor that is respawning an old instance.
        time.sleep(.2)
        remaining = select_processes(snapshot(), roots, protected)
        if remaining:
            terminate(remaining, grace=2)
            if select_processes(snapshot(), roots, protected):
                raise RuntimeError('Old instance is being respawned; new startup refused')
        for name in ('web.pid', 'embeddinggemma2.pid', 'ipc-voice-motion.pid', 'display-browser.pid', 'supervisor.pid'):
            (runtime/name).unlink(missing_ok=True)
        if stopping: return
        if args.action == 'stop':
            print('[runtime] All old project processes stopped', flush=True); return
        owner = lock_file(runtime/'instance.lock', timeout=2)
        (runtime/'supervisor.pid').write_text(str(os.getpid())+'\n')
        env = dict(os.environ, GREENRIVER_INSTANCE_ID=instance, GREENRIVER_PROJECT_ROOT=str(root))
        child = subprocess.Popen(['bash', str(root/'scripts/run_ipc_voice.sh'), '--runtime-child', *launch_args],
                                 cwd=root, env=env, start_new_session=True)
        (runtime/'ipc-voice-motion.pid').write_text(str(child.pid)+'\n')
        deadline = time.monotonic()+150
        expected = ('voice_launcher', 'web', 'voice_gateway', 'display', 'knowledge', 'motion_supervisor', 'arm_sdk_host')
        while not stopping:
            if child.poll() is not None: raise RuntimeError(f'Voice launcher exited: {child.returncode}')
            current = snapshot(); owned = select_processes(current, roots, protected, instance)
            summary = counts(owned, roots)
            if any(number > 1 for kind, number in summary.items() if kind not in {'browser_child', 'bootstrap'}):
                raise RuntimeError('Duplicate runtime component detected: '+json.dumps(summary))
            if all(summary.get(kind) == 1 for kind in expected):
                break
            if time.monotonic() > deadline: raise TimeoutError('Runtime did not become ready')
            time.sleep(.2)
        if stopping: return
        if browser_config:
            browser_env = dict(env, **browser_config['env'])
            command = browser_config['argv']
            port = os.environ.get('RAG_SYSTEM_PORT', '8001')
            command = [f'--app=http://127.0.0.1:{port}/display' if x.startswith('--app=') else x for x in command]
            with (runtime/'display-browser.log').open('ab', buffering=0) as log:
                browser = subprocess.Popen(command, cwd=root, env=browser_env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            (runtime/'display-browser.pid').write_text(str(browser.pid)+'\n')
            time.sleep(.5)
            if browser.poll() is not None: raise RuntimeError('Display browser failed to start; see display-browser.log')
        summary = counts(select_processes(snapshot(), roots, protected, instance), roots)
        manifest = {'instance_id': instance, 'supervisor_pid': os.getpid(), 'launcher_pid': child.pid, 'counts': summary}
        (runtime/'runtime-instance.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('[runtime] Single runtime started: '+json.dumps(manifest), flush=True)
        startup.close(); startup = None
        while not stopping:
            if child.poll() is not None: raise RuntimeError(f'Voice launcher exited: {child.returncode}')
            time.sleep(.5)
    finally:
        # Instance IDs prevent an old supervisor from killing a newer startup.
        own = select_processes(snapshot(), roots, protected, instance)
        if own: terminate(own, grace=8)
        if child is not None:
            try: child.wait(timeout=1)
            except subprocess.TimeoutExpired: pass
        if owner is not None: owner.close()
        if startup is not None: startup.close()


if __name__ == '__main__':
    main()
