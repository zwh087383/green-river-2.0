"""Install and verify the reversible vendor trunk guard without restarting chassis."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import signal
import time


def paths(sdk):
    directory=sdk/'install/lib/trunk_control'
    return directory/'trunk_control_node',directory/'trunk_control_node.greenriver-original',directory/'greenriver-trunk-owner-idle.py'


def wrapper_bytes(sdk):
    _,_,guard=paths(sdk)
    return ('#!/bin/sh\n# GREENRIVER_TRUNK_OWNER_GUARD_V1\nexec /usr/bin/python3 "'+str(guard)+'" "$@"\n').encode()


def verify_guard(root:Path,sdk:Path):
    executable,original,guard=paths(sdk)
    if (not original.is_file() or executable.read_bytes()!=wrapper_bytes(sdk)
            or guard.read_bytes()!=(root/'scripts/trunk_owner_idle.py').read_bytes()):
        raise RuntimeError('Trunk owner vendor guard missing or changed; refusing hardware startup')
    old=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            target=(entry/'exe').readlink()
            if target in (executable,original):old.append(entry.name)
        except OSError:pass
    if old:raise RuntimeError('Vendor six-axis trunk owner still running: '+','.join(old))


def install_guard(root:Path,sdk:Path):
    executable,original,guard=paths(sdk)
    if not original.exists():
        if not executable.read_bytes().startswith(b'\x7fELF'):
            raise RuntimeError('Expected original vendor trunk ELF')
        shutil.copy2(executable,original)
    elif not executable.read_bytes().startswith((b'\x7fELF',b'#!/bin/sh')):
        raise RuntimeError('Unexpected vendor executable')
    # Capture only the exact vendor executable identities before replacing its path.
    identities=[]
    from robot_runtime import snapshot,signal_process,same_process
    for process in snapshot().values():
        try:target=Path(f'/proc/{process.pid}/exe').readlink()
        except OSError:continue
        if target in (executable,original) and process.uid==os.getuid():identities.append(process)
    temporary=guard.with_suffix('.tmp');shutil.copy2(root/'scripts/trunk_owner_idle.py',temporary);os.replace(temporary,guard)
    temporary=executable.with_suffix('.tmp');temporary.write_bytes(wrapper_bytes(sdk));temporary.chmod(0o755);os.replace(temporary,executable)
    for process in identities:signal_process(process,signal.SIGTERM)
    deadline=time.monotonic()+15
    while any(same_process(process) for process in identities):
        if time.monotonic()>deadline:raise RuntimeError('Old trunk owner did not stop; refusing takeover')
        time.sleep(.1)
    verify_guard(root,sdk)
    return {'original_backup':str(original),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
