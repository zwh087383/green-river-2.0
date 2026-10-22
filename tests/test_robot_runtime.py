from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('runtime_under_test',ROOT/'scripts/robot_runtime.py')
runtime=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=runtime
SPEC.loader.exec_module(runtime)


def proc(pid=100, *, argv=(), cwd='/robot', env=None, uid=None, start=10, ppid=1):
    return runtime.Process(pid,ppid,start,'S',os.getuid() if uid is None else uid,tuple(argv),cwd,env or {})


def test_scope_protects_unrelated_python_browsers_and_vendor_platform():
    roots=(Path('/robot'),)
    assert runtime.role(proc(argv=['python','-m','rag_system.web']),roots)=='web'
    assert runtime.role(proc(argv=['python','-m','rag_system.web'],cwd='/other'),roots) is None
    assert runtime.role(proc(argv=['python','-c','print("rag_system.web")']),roots) is None
    assert runtime.role(proc(argv=['/vendor/arm_control_node'],cwd='/vendor'),roots) is None
    assert runtime.role(proc(argv=['/vendor/arm_control_node'],cwd='/vendor',env={'GREENRIVER_PROJECT_ROOT':'/robot'}),roots)=='arm_sdk_host'
    assert runtime.role(proc(argv=['python','-m','rag_system.web'],uid=os.getuid()+1),roots) is None
    assert runtime.role(proc(argv=['chrome','--user-data-dir=/other/display-browser-profile']),roots) is None
    assert runtime.role(proc(argv=['chrome','--user-data-dir=/robot/data/runtime/display-browser-profile']),roots)=='kiosk'
    assert runtime.role(proc(argv=['chrome','--type=renderer','--user-data-dir=/robot/data/runtime/display-browser-profile']),roots)=='browser_child'


def test_orphans_and_helpers_are_selected_without_signal_to_protected_ancestors():
    display=proc(pid=10,argv=['python','-c','from voice.display_node import main; main()'])
    helper=proc(pid=11,ppid=10,argv=['/usr/bin/helper'],cwd='/other')
    launcher=proc(pid=12,argv=['bash','/robot/scripts/run_ipc_voice.sh'])
    selected=runtime.select_processes({p.pid:p for p in [display,helper,launcher]},(Path('/robot'),),{12})
    assert set(selected)=={10,11}


def test_shutdown_selects_only_its_own_instance():
    a=proc(pid=10,argv=['python','-m','rag_system.web'],env={'GREENRIVER_INSTANCE_ID':'old'})
    b=proc(pid=11,argv=['python','-m','rag_system.web'],env={'GREENRIVER_INSTANCE_ID':'new'})
    selected=runtime.select_processes({10:a,11:b},(Path('/robot'),),set(),'old')
    assert set(selected)=={10}


def test_reused_pid_is_never_signalled(monkeypatch):
    old=proc(argv=['python','-m','rag_system.web'])
    monkeypatch.setattr(runtime,'read_process',lambda pid:proc(pid,start=11))
    monkeypatch.setattr(os,'kill',lambda *args:pytest.fail('must not signal reused PID'))
    runtime.signal_process(old,signal.SIGKILL)


def test_kiosk_restores_desktop_environment_when_chromium_erases_it(tmp_path,monkeypatch):
    for key in runtime.SESSION_KEYS:monkeypatch.delenv(key,raising=False)
    desktop=proc(pid=10,argv=['/usr/bin/gnome-shell'],cwd='/other',
                 env={'DISPLAY':':0','XAUTHORITY':'/run/user/1000/gdm/Xauthority','PRIVATE_TOKEN':'secret'})
    browser=proc(pid=20,argv=['chrome','--user-data-dir=/robot/data/runtime/display-browser-profile','--kiosk'])
    file=tmp_path/'display-session.json'
    config=runtime.save_browser_session({20:browser},(Path('/robot'),),file,{10:desktop,20:browser})
    assert config['env']=={'DISPLAY':':0','XAUTHORITY':'/run/user/1000/gdm/Xauthority'}
    assert config['argv'][-1]=='--kiosk'
    assert file.stat().st_mode & 0o777==0o600
    restored=runtime.save_browser_session({},(Path('/robot'),),file,{10:desktop})
    assert restored==config


@pytest.fixture
def fake_project(tmp_path):
    root=tmp_path/'robot'
    for path in ['scripts','ros2-install','.venv-linux/bin','fake/voice','fake/rag_system/indexing','data/runtime']:
        (root/path).mkdir(parents=True,exist_ok=True)
    (root/'.venv-linux/bin/python').symlink_to(sys.executable)
    (root/'ros2-install/setup.bash').write_text('')
    for p in [root/'fake/voice/__init__.py',root/'fake/rag_system/__init__.py',root/'fake/rag_system/indexing/__init__.py']:p.write_text('')
    for module in ['display_node','knowledge_node','realtime_gateway_node']:
        (root/f'fake/voice/{module}.py').write_text('import time\ndef main(): time.sleep(120)\n')
    for module in ['web.py','indexing/embedding_server.py']:
        (root/f'fake/rag_system/{module}').write_text('import time\ntime.sleep(120)\n')
    (root/'arm_control_node').symlink_to('/bin/sleep')
    (root/'scripts/sdk_motion_host.py').write_text('''import subprocess,sys,time
from pathlib import Path
subprocess.Popen([str(Path.cwd()/'arm_control_node'),'120'])
time.sleep(120)
''')
    (root/'fake_launcher.py').write_text('''import subprocess,sys,time
from pathlib import Path
for module in ['rag_system.web','rag_system.indexing.embedding_server']:
 subprocess.Popen([sys.executable,'-m',module])
for module in ['realtime_gateway_node','display_node','knowledge_node']:
 subprocess.Popen([sys.executable,'-c',f'from voice.{module} import main; main()'])
subprocess.Popen([sys.executable,str(Path.cwd()/'scripts/sdk_motion_host.py')])
time.sleep(120)
''')
    (root/'scripts/run_ipc_voice.sh').write_text(f'#!/bin/bash\ncd "{root}"\nexec "{sys.executable}" "{root}/fake_launcher.py" voice_realtime.launch.py\n')
    env=dict(os.environ,PYTHONPATH=str(root/'fake'),RAG_DATA_DIR=str(root/'data'))
    yield root,env
    found=runtime.select_processes(runtime.snapshot(),(root,),{os.getpid()})
    runtime.terminate(found,grace=.2)


def wait_manifest(root, pid, timeout=12):
    file=root/'data/runtime/runtime-instance.json'
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if file.exists():
            value=json.loads(file.read_text())
            if value['supervisor_pid']==pid:return value
        time.sleep(.05)
    raise AssertionError('New runtime did not become ready')


def test_repeated_start_cleans_orphans_and_leaves_one_copy(fake_project,tmp_path):
    root,env=fake_project
    orphan=subprocess.Popen([sys.executable,'-c','from voice.display_node import main; main()'],cwd=root,env=env)
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],cwd=tmp_path)
    (root/'data/runtime/web.pid').write_text(str(unrelated.pid))
    command=[sys.executable,str(ROOT/'scripts/robot_runtime.py'),'start','--project-root',str(root)]
    output=(root/'test-runtime.log').open('wb')
    first=second=None
    try:
        first=subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
        wait_manifest(root,first.pid)
        assert orphan.poll() is not None
        second=subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
        wait_manifest(root,second.pid)
        first.wait(timeout=4)
        found=runtime.select_processes(runtime.snapshot(),(root,),{os.getpid()})
        summary=runtime.counts(found,(root,))
        assert summary['supervisor']==1
        for kind in ['web','embedding','voice_launcher','voice_gateway','display','knowledge','motion_supervisor','arm_sdk_host']:
            assert summary[kind]==1,summary
        assert unrelated.poll() is None
        second.terminate();second.wait(timeout=5)
        assert not runtime.select_processes(runtime.snapshot(),(root,),{os.getpid()})
    finally:
        for p in [first,second,orphan,unrelated]:
            if p and p.poll() is None:p.kill();p.wait(timeout=2)
        output.close()


def test_concurrent_starts_are_serialized(fake_project):
    root,env=fake_project
    command=[sys.executable,str(ROOT/'scripts/robot_runtime.py'),'start','--project-root',str(root)]
    processes=[]
    with (root/'concurrent.log').open('wb') as output:
        try:
            processes=[subprocess.Popen(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT) for _ in range(2)]
            deadline=time.monotonic()+14
            while time.monotonic()<deadline:
                alive=[p for p in processes if p.poll() is None]
                if len(alive)==1:
                    manifest=wait_manifest(root,alive[0].pid)
                    summary=runtime.counts(runtime.select_processes(runtime.snapshot(),(root,),{os.getpid()}),(root,))
                    assert summary.get('supervisor')==1 and summary.get('display')==1 and summary.get('knowledge')==1
                    break
                time.sleep(.1)
            else:raise AssertionError('Concurrent startup left more than one supervisor')
        finally:
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2)
