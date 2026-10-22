from __future__ import annotations
import time
from pathlib import Path
import pytest
from rag_system.robot_motion import exchange, request_result
from test_native_motion import binary, worker, wait_status


def test_catalog_result_is_retained_after_the_next_request_and_pinned_to_instance(worker):
    path,_=worker(startup_home=True,GREENRIVER_MOTION_PRESETS=str(Path(__file__).parents[1]/'config/motion_presets.json'))
    ack=exchange('PRESET first wave_goodbye',path=str(path))
    assert ack['status']=='accepted'
    wait_status(path,lambda s:s['state']=='ready' and s['completed']==2)
    result=request_result('first',instance_id=ack['instance_id'],path=str(path))
    assert result['result']['result_state']=='verified_completed'
    assert result['result']['target_verified'] and result['result']['process_verified']
    exchange('PRESET second speech',path=str(path))
    assert request_result('first',path=str(path))['result']['result_state']=='verified_completed'
    assert exchange('PRESET first speech',path=str(path))['status']=='conflict'
    changed=request_result('first',instance_id='different_host',path=str(path))
    assert changed['status']=='instance_changed' and changed['result'] is None
    assert request_result('missing',path=str(path))['status']=='not_found'


def test_queued_and_coalesced_home_requests_share_the_original_result(worker):
    path,_=worker()
    exchange('PLAY gesture wake_wave',path=str(path))
    assert exchange('HOME first_home',path=str(path))['status']=='queued'
    assert exchange('HOME alias_home',path=str(path))['status']=='coalesced'
    wait_status(path,lambda s:s['state']=='ready' and s['completed']==2)
    alias=request_result('alias_home',path=str(path))['result']
    assert alias['result_state']=='verified_completed'
    assert alias['canonical_request_id']=='first_home'


def test_timed_out_call_never_becomes_verified_after_its_late_return(worker):
    path,_=worker(timeout=.1,MOTION_TEST_DURATION_MS='500')
    exchange('PLAY slow answer_gesture',path=str(path))
    wait_status(path,lambda s:s['state']=='fault')
    assert request_result('slow',path=str(path))['result']['result_state']=='unknown'
    time.sleep(.6)
    assert request_result('slow',path=str(path))['result']['result_state']=='unknown'


def test_explicit_recovery_has_a_new_result_and_keeps_old_failure(worker):
    path,_=worker(MOTION_TEST_FIRST_PLAY_FAIL='1')
    exchange('PLAY failed wake_wave',path=str(path))
    wait_status(path,lambda s:s['state']=='fault')
    assert request_result('failed',path=str(path))['result']['result_state']=='failed'
    exchange('POWER recover restart',path=str(path))
    wait_status(path,lambda s:s['state']=='ready')
    assert request_result('recover',path=str(path))['result']['result_state']=='verified_completed'
    assert request_result('failed',path=str(path))['result']['result_state']=='failed'


def test_result_api_rejects_injected_tokens_before_transport(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('transport must not be called')
    monkeypatch.setattr('rag_system.robot_motion.exchange',forbidden)
    with pytest.raises(ValueError):request_result('x\nPLAY')
