from __future__ import annotations
import json
import math
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
import pytest
from rag_system.robot_motion import MotionClient, exchange

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def binary(tmp_path_factory):
    if not shutil.which('g++'):
        pytest.skip('g++ is required for native worker integration tests')
    output = tmp_path_factory.mktemp('native-motion')/'worker'
    subprocess.run(['g++','-std=c++17','-O1','-Wall','-Wextra','-Werror','-pthread',
                    '-DGREENRIVER_MOTION_FAKE',str(ROOT/'native/motion/sdk_motion_worker.cpp'),
                    '-o',str(output)],check=True,capture_output=True,text=True)
    return output

def wait_status(path, predicate, timeout=3):
    end=time.monotonic()+timeout
    last=None
    while time.monotonic()<end:
        try:
            last=exchange('STATUS',path=str(path))
            if predicate(last): return last
        except OSError: pass
        time.sleep(.01)
    raise AssertionError(last)

@pytest.fixture
def worker(binary, tmp_path):
    processes=[]
    def start(*, timeout=2, startup_home=False, **env):
        path=tmp_path/f'm{len(processes)}.sock'
        log=tmp_path/f'm{len(processes)}.log'
        if not startup_home:
            env['MOTION_TEST_SKIP_STARTUP_HOME']='1'
        process=subprocess.Popen([str(binary),'--socket',str(path),'--timeout',str(timeout)],
            env=dict(os.environ,MOTION_TEST_LOG=str(log),**env),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        processes.append(process)
        wait_status(path,lambda s:s['state'] in {'ready','fault'})
        return path,log
    yield start
    for p in processes:
        p.terminate()
        try:p.communicate(timeout=2)
        except subprocess.TimeoutExpired:p.kill();p.communicate(timeout=2)


def test_native_worker_serializes_and_deduplicates_without_reset(worker):
    path,log=worker()
    assert exchange('PLAY first wake_wave',path=str(path))['status']=='accepted'
    assert exchange('PLAY second answer_gesture',path=str(path))['status']=='busy'
    wait_status(path,lambda s:s['completed']==1)
    assert exchange('PLAY first wake_wave',path=str(path))['status']=='duplicate'
    assert log.read_text().splitlines()==['4']
    assert exchange('PLAY third answer_gesture',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['completed']==2)
    assert log.read_text().splitlines()==['4','5']


def test_manual_fly_kiss_serializes_all_frames_converts_units_and_returns_home(worker):
    path,log=worker(startup_home=True,GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    start=exchange('PRESET kiss1 fly_kiss',path=str(path))
    assert start['status']=='accepted' and start['step_count']==3
    assert exchange('PLAY collision wake_wave',path=str(path))['status']=='busy'
    done=wait_status(path,lambda s:s['state']=='ready' and s['completed']==2,timeout=5)
    assert done['at_preset'] and done['step_index']==3
    lines=log.read_text().splitlines();assert lines[0]=='1'
    frames=[json.loads(line) for line in lines[1:]]
    catalog=json.loads((ROOT/'config/motion_presets.json').read_text())
    configured=next(p for p in catalog['presets'] if p['id']=='fly_kiss')['frames']
    assert [f['scale'] for f in frames]==[f['scale'] for f in configured]
    assert math.isclose(frames[0]['left'][0],math.radians(-14.77))
    assert math.isclose(frames[1]['right'][4],math.radians(86.83))
    assert frames[-1]['home'] is True
    assert exchange('PRESET kiss1 fly_kiss',path=str(path))['status']=='duplicate'


def test_manual_test_requires_home_and_power_and_rejects_unregistered_names(worker):
    path,log=worker(GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    assert exchange('PRESET a fly_kiss',path=str(path))['status']=='not_at_home'
    assert exchange('HOME atstart',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='ready' and s['at_preset'])
    assert exchange('PRESET unknown not_registered',path=str(path))['status']=='invalid_preset'
    assert exchange('PRESET bad ../fly_kiss',path=str(path))['status']=='invalid'
    assert exchange('POWER off off',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='powered_off')
    assert exchange('PRESET asleep fly_kiss',path=str(path))['status']=='unavailable'


@pytest.mark.parametrize('flag,frame_count',[('MOTION_TEST_FRAME_LIMIT_FAIL',0),('MOTION_TEST_FRAME_FAIL',1)])
def test_failed_manual_frame_never_advances_or_returns_home_automatically(worker,flag,frame_count):
    path,log=worker(startup_home=True,GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'),**{flag:'1'})
    assert exchange('PRESET fail fly_kiss',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault')
    lines=log.read_text().splitlines();assert lines[0]=='1' and len(lines)==1+frame_count
    assert exchange('PRESET retry fly_kiss',path=str(path))['status']=='unavailable'


def test_manual_frame_timeout_never_sends_later_frames(worker,tmp_path):
    data=json.loads((ROOT/'config/motion_presets.json').read_text())
    next(p for p in data['presets'] if p['id']=='fly_kiss')['timeout_s']=5
    catalog=tmp_path/'presets.json';catalog.write_text(json.dumps(data))
    path,log=worker(startup_home=True,GREENRIVER_MOTION_PRESETS=str(catalog),MOTION_TEST_FRAME_NOT_REACHED='1')
    assert exchange('PRESET stalled fly_kiss',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault',timeout=7)
    assert len(log.read_text().splitlines())==2


@pytest.mark.parametrize('preset,code',[('invitation','2'),('wave_goodbye','4'),('speech','5')])
def test_sdk_presets_use_existing_sdk_action_then_home(worker,preset,code):
    path,log=worker(startup_home=True,GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    assert exchange('PRESET action '+preset,path=str(path))['status']=='accepted'
    done=wait_status(path,lambda s:s['state']=='ready' and s['completed']==2,timeout=5)
    assert done['at_preset'] and log.read_text().splitlines()==['1',code,'1']


def test_power_off_homes_first_and_power_on_homes_after_connect(worker):
    path, log = worker()
    assert exchange('POWER off1 off', path=str(path))['status'] == 'accepted'
    assert exchange('PLAY wave wake_wave', path=str(path))['status'] != 'accepted'
    assert exchange('HOME reload', path=str(path))['status'] == 'unavailable'
    off = wait_status(path, lambda s: s['state'] == 'powered_off')
    assert off['power'] == {'left': 1, 'right': 1}
    assert log.read_text().splitlines() == ['1', 'POWER_OFF']
    assert exchange('POWER off1 off', path=str(path))['status'] == 'duplicate'
    assert exchange('POWER off2 off', path=str(path))['status'] == 'already_off'
    assert exchange('HOME reload2', path=str(path))['status'] == 'unavailable'
    assert exchange('PLAY asleep wake_wave', path=str(path))['status'] == 'unavailable'
    time.sleep(.15)
    assert exchange('STATUS', path=str(path))['state'] == 'powered_off'
    assert exchange('POWER on1 on', path=str(path))['status'] == 'accepted'
    on = wait_status(path, lambda s: s['state'] == 'ready')
    assert on['at_preset'] and on['power'] == {'left': 0, 'right': 0}
    assert log.read_text().splitlines() == ['1', 'POWER_OFF', 'POWER_ON', '1']
    assert exchange('POWER on2 on', path=str(path))['status'] == 'already_on'


def test_power_controls_reject_busy_and_invalid_requests(worker):
    path, log = worker(MOTION_TEST_DURATION_MS='250')
    for command in ['POWER x maybe', 'POWER x off extra', 'POWER ../x off']:
        assert exchange(command, path=str(path))['status'] == 'invalid'
    assert exchange('PLAY gesture answer_gesture', path=str(path))['status'] == 'accepted'
    assert exchange('POWER off1 off', path=str(path))['status'] == 'busy'
    assert exchange('POWER on1 on', path=str(path))['status'] == 'busy'
    wait_status(path, lambda s: s['state'] == 'ready')
    assert log.read_text().splitlines() == ['5']


def test_arm_restart_is_atomic_deduplicated_and_preserves_trunk(worker,tmp_path):
    trunk_log=tmp_path/'trunk-restart.log'
    path,log=worker(MOTION_TEST_TRUNK_LOG=str(trunk_log))
    before=exchange('STATUS',path=str(path))['trunk']
    ack=exchange('POWER reboot restart',path=str(path))
    assert ack['status']=='accepted' and ack['state']=='restarting_arms'
    assert exchange('POWER reboot restart',path=str(path))['status']=='duplicate'
    assert exchange('POWER another restart',path=str(path))['status']=='busy'
    assert exchange('ALLPOWER collide off',path=str(path))['status']=='busy'
    assert exchange('HOME reload',path=str(path))['status']=='unavailable'
    assert exchange('PLAY wave wake_wave',path=str(path))['status']=='unavailable'
    done=wait_status(path,lambda s:s['state']=='ready' and s['completed']==1,timeout=5)
    assert done['at_preset'] and done['purpose']=='power_restart_home'
    assert done['power']=={'left':0,'right':0} and done['trunk']==before
    assert log.read_text().splitlines()==['1','POWER_OFF','POWER_ON','1']
    assert not trunk_log.exists()


def test_arm_restart_when_already_off_only_enables_and_homes(worker):
    path,log=worker()
    assert exchange('POWER off off',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='powered_off')
    assert exchange('POWER reboot restart',path=str(path))['status']=='accepted'
    done=wait_status(path,lambda s:s['state']=='ready' and s['completed']==2,timeout=5)
    assert done['at_preset']
    assert log.read_text().splitlines()==['1','POWER_OFF','POWER_ON','1']


@pytest.mark.parametrize('env,expected',[
    ({'MOTION_TEST_HOME_MISMATCH':'1'},['1']),
    ({'MOTION_TEST_OFF_UNKNOWN':'1'},['1','POWER_OFF']),
    ({'MOTION_TEST_POWER_ON_HOME_FAIL':'1'},['1','POWER_OFF','POWER_ON','1']),
])
def test_arm_restart_failure_stops_sequence_without_retry(worker,env,expected):
    path,log=worker(**env)
    assert exchange('POWER reboot restart',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault',timeout=5)
    assert log.read_text().splitlines()==expected
    assert exchange('POWER reboot restart',path=str(path))['status']=='duplicate'


def test_explicit_restart_recovers_fault_and_resumes_jobs_without_touching_trunk(worker,tmp_path):
    trunk_log=tmp_path/'recovery-trunk.log'
    path,log=worker(MOTION_TEST_FIRST_PLAY_FAIL='1',MOTION_TEST_TRUNK_LOG=str(trunk_log))
    assert exchange('PLAY fail wake_wave',path=str(path))['status']=='accepted'
    fault=wait_status(path,lambda s:s['state']=='fault')
    before=fault['trunk']
    ack=exchange('POWER recover restart',path=str(path))
    assert ack['status']=='accepted' and ack['restart_pending'] and ack['state']=='fault'
    assert exchange('POWER recover restart',path=str(path))['status']=='duplicate'
    assert exchange('POWER another restart',path=str(path))['status']=='busy'
    done=wait_status(path,lambda s:s['state']=='ready' and s['at_preset'],timeout=5)
    assert done['trunk']==before and done['power']=={'left':0,'right':0}
    assert log.read_text().splitlines()==['4','POWER_OFF','POWER_ON','1']
    assert not trunk_log.exists()
    assert exchange('PLAY resumed wake_wave',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='ready' and s['completed']==2)


def test_restart_waits_for_timed_out_sdk_to_exit_without_releasing_old_fault(worker):
    path,log=worker(timeout=.1,MOTION_TEST_DURATION_MS='600')
    assert exchange('PLAY slow wake_wave',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault')
    ack=exchange('POWER recover restart',path=str(path))
    assert ack['status']=='accepted' and ack['state']=='fault' and ack['restart_pending']
    time.sleep(.1)
    assert log.read_text().splitlines()==['4']
    assert exchange('HOME refresh',path=str(path))['status']=='unavailable'
    done=wait_status(path,lambda s:s['state']=='ready' and s['at_preset'],timeout=5)
    assert done['purpose']=='power_restart_home'
    assert log.read_text().splitlines()==['4','POWER_OFF','POWER_ON','1']


def test_fault_restart_refuses_safety_interlock_without_disconnecting_or_moving(worker):
    path,log=worker(MOTION_TEST_FIRST_PLAY_FAIL='1',MOTION_TEST_UNSAFE_RESTART='1')
    assert exchange('PLAY fail wake_wave',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault')
    assert exchange('POWER recover restart',path=str(path))['status']=='accepted'
    done=wait_status(path,lambda s:s['state']=='fault' and not s['restart_pending'] and 'blocked' in s['detail'])
    assert not done['at_preset'] and log.read_text().splitlines()==['4']


def test_restart_transport_uses_one_distinct_deduplicated_command(monkeypatch):
    import rag_system.robot_motion as module
    commands=[]
    monkeypatch.setattr(module,'exchange',lambda command,**kwargs:commands.append(command) or {'status':'accepted'})
    module.request_arm_restart('restart-1')
    module.request_arm_restart('restart-1')
    assert commands[0]==commands[1] and commands[0].startswith('POWER ') and commands[0].endswith(' restart')


@pytest.mark.parametrize('env', [{'MOTION_TEST_HOME_MISMATCH': '1'}, {'MOTION_TEST_PLAY_FAIL': '1'}])
def test_failed_home_never_calls_power_off_or_retries(worker, env):
    path, log = worker(**env)
    assert exchange('POWER off1 off', path=str(path))['status'] == 'accepted'
    state = wait_status(path, lambda s: s['state'] == 'fault')
    assert not state['at_preset']
    assert log.read_text().splitlines() == ['1']
    assert exchange('POWER recover on', path=str(path))['status'] == 'unavailable'
    assert exchange('POWER again off', path=str(path))['status'] == 'unavailable'


def test_power_off_home_watchdog_never_disconnects_after_late_return(worker):
    path, log = worker(timeout=.1, MOTION_TEST_DURATION_MS='600')
    assert exchange('POWER off1 off', path=str(path))['status'] == 'accepted'
    wait_status(path, lambda s: s['state'] == 'fault')
    time.sleep(.7)
    assert log.read_text().splitlines() == ['1']
    assert exchange('STATUS', path=str(path))['state'] == 'fault'


def test_power_on_home_failure_latches_without_gestures(worker):
    path, log = worker(MOTION_TEST_POWER_ON_HOME_FAIL='1')
    assert exchange('POWER off1 off', path=str(path))['status'] == 'accepted'
    wait_status(path, lambda s: s['state'] == 'powered_off')
    assert exchange('POWER on1 on', path=str(path))['status'] == 'accepted'
    wait_status(path, lambda s: s['state'] == 'fault')
    assert log.read_text().splitlines() == ['1', 'POWER_OFF', 'POWER_ON', '1']
    assert exchange('PLAY wave wake_wave', path=str(path))['status'] == 'unavailable'


def test_unknown_power_feedback_is_reported_without_reenabling_motion(worker):
    path, log = worker(MOTION_TEST_OFF_UNKNOWN='1')
    assert exchange('POWER off1 off', path=str(path))['status'] == 'accepted'
    state = wait_status(path, lambda s: s['state'] == 'powered_off')
    assert state['power'] == {'left': -1, 'right': -1}
    assert exchange('HOME reload', path=str(path))['status'] == 'unavailable'
    assert log.read_text().splitlines() == ['1', 'POWER_OFF']


def test_blocked_sdk_keeps_socket_responsive_and_fault_latched(worker):
    path,log=worker(timeout=.1,MOTION_TEST_DURATION_MS='600')
    assert exchange('PLAY first wake_wave',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault')
    start=time.monotonic()
    assert exchange('PLAY second answer_gesture',path=str(path))['status']=='unavailable'
    assert time.monotonic()-start<.2
    time.sleep(.7)
    result=exchange('STATUS',path=str(path))
    assert result['state']=='fault' and result['completed']==0
    assert log.read_text().splitlines()==['4']


@pytest.mark.parametrize('env',[{'MOTION_TEST_CONNECT_FAIL':'1'},{'MOTION_TEST_HEALTH_FAIL':'1'},
                                 {'MOTION_TEST_MOVING':'1'}])
def test_unready_hardware_does_not_accept_gestures(worker,env):
    path,log=worker(**env)
    assert exchange('PLAY first wake_wave',path=str(path))['status']=='unavailable'
    assert not log.exists()


def test_sdk_error_latches_without_retry_or_reset(worker):
    path,log=worker(MOTION_TEST_PLAY_FAIL='1')
    assert exchange('PLAY first closing_wave',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault')
    assert exchange('PLAY other wake_wave',path=str(path))['status']=='unavailable'
    assert log.read_text().splitlines()==['4']


def test_malformed_and_slow_requests_cannot_stall_worker(worker):
    path,log=worker()
    for command in ['PLAY x reset','PLAY x wake_wave unexpected','STATUS extra','PLAY ../x wake_wave']:
        assert exchange(command,path=str(path))['status']=='invalid'
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as slow:
        slow.connect(str(path));slow.sendall(b'P')
        start=time.monotonic()
        assert exchange('STATUS',path=str(path))['state']=='ready'
        assert time.monotonic()-start<.3
    assert not log.exists()


def test_second_owner_cannot_unlink_live_socket(worker,binary):
    path,_=worker()
    p=subprocess.run([str(binary),'--socket',str(path)],capture_output=True,text=True,timeout=2)
    assert p.returncode!=0
    assert exchange('STATUS',path=str(path))['state']=='ready'


def test_client_is_nonblocking_drops_busy_and_stale_without_retries(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();calls=[]
    def blocked(command,**kwargs):
        calls.append(command);entered.set();release.wait(2);return {'status':'accepted'}
    monkeypatch.setattr('rag_system.robot_motion.exchange',blocked)
    client=MotionClient(path=str(tmp_path/'missing.sock'))
    start=time.monotonic()
    assert client.request('first',1,'wake_wave')
    assert time.monotonic()-start<.1
    assert entered.wait(1)
    assert not client.request('second',1,'answer_gesture')
    client.advance(2)
    assert not client.request('stale',1,'wake_wave')
    release.set();client.close()
    assert len(calls)==1
    assert not client.request('closed',3,'closing_wave')


def test_client_failure_is_reported_without_blocking_voice(tmp_path):
    reports=[]
    client=MotionClient(path=str(tmp_path/'absent.sock'),report=reports.append)
    assert client.request('first',1,'wake_wave')
    client._thread.join(1)
    client.close()
    assert len(reports)==1 and 'skipped' in reports[0]


@pytest.mark.parametrize('closing', ['wave_goodbye', 'heart', 'fly_kiss'])
def test_voice_gestures_return_home_and_random_closing_uses_existing_presets(worker, monkeypatch, closing):
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    choices = []
    def choose(options):
        choices.append(options)
        if options == ('wave_goodbye', 'heart', 'fly_kiss'):
            return closing
        assert set(options)=={'open_palms','raise_hand','point_far','flex_muscles','speech'}
        return 'open_palms'
    monkeypatch.setattr('rag_system.robot_motion.random.choice', choose)
    client = MotionClient(path=str(path))
    for count, (purpose, preset) in enumerate([
        ('wake_wave', 'wave_goodbye'), ('answer_gesture', 'open_palms'),
        ('answer_invite', 'open_palms'), ('answer_open_palms', 'open_palms'), ('closing_gesture', closing)
    ], 2):
        assert client.request('turn-' + str(count), count, purpose)
        client._thread.join(1)
        state = wait_status(path, lambda s: s['state']=='ready' and s['completed']==count, timeout=6)
        assert state['at_preset'] and state['purpose']=='preset:' + preset
    client.close()
    assert len(choices)==4
    assert log.read_text().splitlines()[:3] == ['1', '4', '1']


def test_removed_scratch_head_is_rejected_by_native_catalog(worker):
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    assert exchange('PRESET removed scratch_head', path=str(path))['status']=='invalid_preset'
    assert log.read_text().splitlines() == ['1']


def test_sdk_host_dereferences_yaml_anchors_for_ros_parameters():
    import importlib.util
    yaml=pytest.importorskip('yaml')
    spec=importlib.util.spec_from_file_location('_motion_host_config',ROOT/'scripts/sdk_motion_host.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source='info:\n  joints: &joints [j1, j2]\ncontroller:\n  ros__parameters:\n    joints: *joints\n'
    loaded=yaml.safe_load(source)
    output=yaml.dump(loaded,Dumper=module.NoAliasDumper)
    assert yaml.safe_load(output)==loaded
    assert not any(isinstance(token,yaml.tokens.AliasToken) for token in yaml.scan(output))


def test_process_start_homes_once_before_accepting_gestures(worker):
    path,log=worker(startup_home=True)
    state=exchange('STATUS',path=str(path))
    assert state['state']=='ready' and state['completed']==1
    assert state['purpose']=='startup_home' and state['at_preset']
    assert log.read_text().splitlines()==['1']
    time.sleep(.2)
    assert log.read_text().splitlines()==['1']


def test_refresh_waits_for_current_gesture_and_coalesces_reloads(worker):
    path,log=worker(MOTION_TEST_DURATION_MS='200')
    assert exchange('PLAY gesture answer_gesture',path=str(path))['status']=='accepted'
    assert exchange('HOME refresh1',path=str(path))['status']=='queued'
    assert exchange('HOME refresh2',path=str(path))['status']=='coalesced'
    assert exchange('HOME refresh1',path=str(path))['status']=='duplicate'
    assert exchange('PLAY later wake_wave',path=str(path))['status']=='busy'
    wait_status(path,lambda s:s['state']=='ready' and s['completed']==2)
    assert log.read_text().splitlines()==['5','1']
    assert not exchange('STATUS',path=str(path))['home_pending']


def test_refresh_while_homing_does_not_send_another_preset(worker):
    path,log=worker()
    assert exchange('HOME refresh1',path=str(path))['status']=='accepted'
    assert exchange('STATUS',path=str(path))['state']=='homing'
    assert exchange('HOME refresh2',path=str(path))['status']=='coalesced'
    assert exchange('PLAY gesture wake_wave',path=str(path))['status']=='busy'
    wait_status(path,lambda s:s['state']=='ready' and s['completed']==1)
    assert log.read_text().splitlines()==['1']


def test_fault_drops_pending_home_without_recovery_retry(worker):
    path,log=worker(timeout=.1,MOTION_TEST_DURATION_MS='500')
    assert exchange('PLAY gesture answer_gesture',path=str(path))['status']=='accepted'
    assert exchange('HOME refresh',path=str(path))['status']=='queued'
    wait_status(path,lambda s:s['state']=='fault')
    assert exchange('HOME another',path=str(path))['status']=='unavailable'
    time.sleep(.6)
    assert log.read_text().splitlines()==['5']
    assert not exchange('STATUS',path=str(path))['home_pending']


def test_startup_home_failure_keeps_gestures_disabled(worker):
    path,log=worker(startup_home=True,MOTION_TEST_PLAY_FAIL='1')
    assert exchange('STATUS',path=str(path))['state']=='fault'
    assert exchange('PLAY gesture wake_wave',path=str(path))['status']=='unavailable'
    assert log.read_text().splitlines()==['1']


def test_preset_feedback_mismatch_does_not_report_home_success(worker):
    path,log=worker(startup_home=True,MOTION_TEST_HOME_MISMATCH='1')
    state=exchange('STATUS',path=str(path))
    assert state['state']=='fault' and not state['at_preset'] and state['completed']==0
    assert log.read_text().splitlines()==['1']


def test_preset_targets_follow_vendor_config_and_validate_both_arms(tmp_path):
    import importlib.util
    import math
    pytest.importorskip('yaml')
    spec=importlib.util.spec_from_file_location('_motion_host_preset',ROOT/'scripts/sdk_motion_host.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'install/product/config/arm_motion.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'A':[{'left_arm':[0,90,0,0,0,0,0], 'right_arm':[0,-90,0,0,0,0,0]}]}))
    targets=list(map(float,module.preset_target_environment(tmp_path).split()))
    assert len(targets)==14 and targets[1]==pytest.approx(math.pi/2) and targets[8]==pytest.approx(-math.pi/2)
    path.write_text(json.dumps({'A':[{'left_arm':[0]*7}]}))
    with pytest.raises(ValueError):module.preset_target_environment(tmp_path)


def test_all_power_homes_both_groups_and_retains_independent_arm_buttons(worker,tmp_path):
    trunk_log=tmp_path/'trunk.log'
    path,arm_log=worker(startup_home=True,MOTION_TEST_TRUNK_LOG=str(trunk_log))
    assert trunk_log.read_text().splitlines()==['HOME']
    assert exchange('POWER armoff off',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='powered_off')
    assert trunk_log.read_text().splitlines()==['HOME']
    assert exchange('STATUS',path=str(path))['trunk']['power']==[0]*6
    assert exchange('ALLPOWER full-off off',path=str(path))['status']=='accepted'
    off=wait_status(path,lambda s:s['state']=='powered_off' and s['purpose']=='all_power_off_home')
    assert off['trunk']['power']==[1]*6 and off['trunk']['brakes'][:4]==[0]*4
    assert 'POWER_ON' not in arm_log.read_text()  # Already-off arms are not re-enabled by all-off.
    assert trunk_log.read_text().splitlines()==['HOME','HOME','DISABLE']
    assert exchange('ALLPOWER full-off off',path=str(path))['status']=='duplicate'
    assert exchange('POWER arms-only on',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='ready')
    assert exchange('STATUS',path=str(path))['trunk']['power']==[1]*6
    assert exchange('ALLPOWER full-on on',path=str(path))['status']=='accepted'
    ready=wait_status(path,lambda s:s['state']=='ready' and s['purpose']=='all_power_on_home')
    assert ready['power']=={'left':0,'right':0} and ready['trunk']['power']==[0]*6
    assert ready['at_preset'] and ready['trunk']['at_home']
    assert trunk_log.read_text().splitlines()[-2:]==['ENABLE','HOME']


def test_all_power_serializes_and_rejects_refresh_during_operation(worker):
    path,log=worker(startup_home=True,MOTION_TEST_DURATION_MS='350')
    assert exchange('ALLPOWER transaction off',path=str(path))['status']=='accepted'
    assert exchange('POWER conflict off',path=str(path))['status']=='busy'
    assert exchange('ALLPOWER conflict on',path=str(path))['status']=='busy'
    assert exchange('HOME no_override',path=str(path))['status']=='unavailable'
    assert exchange('PRESET no_override fly_kiss',path=str(path))['status']=='busy'
    wait_status(path,lambda s:s['state']=='powered_off')
    assert log.read_text().splitlines()==['1','1','POWER_OFF']


def test_failed_trunk_home_never_disables_either_group(worker,tmp_path):
    trunk_log=tmp_path/'trunk.log'
    path,arm_log=worker(MOTION_TEST_TRUNK_LOG=str(trunk_log),MOTION_TEST_TRUNK_HOME_FAIL='1')
    assert exchange('ALLPOWER fail off',path=str(path))['status']=='accepted'
    result=wait_status(path,lambda s:s['state']=='fault')
    assert 'trunk home failed' in result['detail']
    assert 'POWER_OFF' not in arm_log.read_text()
    assert not trunk_log.exists()
    assert exchange('ALLPOWER retry on',path=str(path))['status']=='unavailable'


def test_trunk_disable_failure_does_not_disconnect_arms(worker):
    path,log=worker(MOTION_TEST_TRUNK_DISABLE_FAIL='1')
    assert exchange('ALLPOWER fail off',path=str(path))['status']=='accepted'
    wait_status(path,lambda s:s['state']=='fault')
    assert 'POWER_OFF' not in log.read_text()
