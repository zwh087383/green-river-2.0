import time
from test_native_motion import binary, worker, wait_status, ROOT
from rag_system.robot_motion import exchange


def test_first_driver_error_stops_builtin_home_and_preserves_original_cause(worker):
    path,log=worker(startup_home=True,GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'),MOTION_TEST_DURATION_MS='450')
    assert exchange('PRESET wave wave_goodbye',path=str(path))['status']=='accepted'
    end=time.monotonic()+1
    while log.read_text().splitlines()!=['1','4'] and time.monotonic()<end:time.sleep(.01)
    exchange('TEST_DRIVER_ERROR first-trajectory-failure',path=str(path))
    first=wait_status(path,lambda s:s['state']=='fault')
    exchange('TEST_DRIVER_ERROR secondary-timeout',path=str(path))
    time.sleep(.6)
    done=exchange('STATUS',path=str(path))
    assert 'first-trajectory-failure' in done['detail']==first['detail']
    assert log.read_text().splitlines()==['1','4']
    assert exchange('PRESET next speech',path=str(path))['status']=='unavailable'


def test_first_driver_error_cancels_later_custom_frames_and_queued_home(worker):
    path,log=worker(startup_home=True,GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    assert exchange('PRESET kiss fly_kiss',path=str(path))['status']=='accepted'
    end=time.monotonic()+1
    while len(log.read_text().splitlines())<2 and time.monotonic()<end:time.sleep(.01)
    assert exchange('HOME queued',path=str(path))['status']=='queued'
    exchange('TEST_DRIVER_ERROR first-frame-driver-failure',path=str(path))
    done=wait_status(path,lambda s:s['state']=='fault')
    time.sleep(.5)
    assert not done['home_pending'] and len(log.read_text().splitlines())==2


def test_power_off_fault_refreshes_actual_arm_power_without_automatic_reenable(worker):
    path,log=worker(MOTION_TEST_FIRST_PLAY_POWER_OFF='1')
    assert exchange('PLAY fail wake_wave',path=str(path))['status']=='accepted'
    done=wait_status(path,lambda s:s['state']=='fault' and s['power']=={'left':1,'right':1})
    assert 'not ready: 1' in done['detail'] and done['power_feedback_age_ms']<2000
    assert log.read_text().splitlines()==['4']


def test_real_driver_log_protocol_filters_old_records_and_stops_dispatch(worker):
    path,log=worker(MOTION_TEST_DURATION_MS='450')
    assert exchange('PLAY wave wake_wave',path=str(path))['status']=='accepted'
    message='10-kCommandJointVelocityLimitsViolation'.encode().hex()
    assert exchange(f'DRIVER_LOG 1 40 right_arm.hardware_interface {message}',path=str(path))['state']=='running'
    failed=exchange(f'DRIVER_LOG {time.time_ns()} 40 right_arm.hardware_interface {message}',path=str(path))
    assert failed['state']=='fault' and 'kCommandJointVelocity' in failed['detail']
    time.sleep(.6)
    assert log.read_text().splitlines()==['4']
