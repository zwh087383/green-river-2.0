"""Playback-scoped native integration: repetition, fencing and safe completion."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rag_system import robot_motion as motion
from test_native_motion import binary, worker, wait_status

ROOT = Path(__file__).parents[1]


def opened(path):
    state = motion.exchange('STATUS', path=str(path))
    instance = state['instance_id']
    ack = motion.exchange(f'OPEN {instance}', path=str(path))
    assert ack['status'] == 'accepted'
    return ack['scope_id'], instance


def test_scope_close_fences_late_gestures_and_isolates_manual_and_next_scope(worker):
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    old, instance = opened(path)
    assert motion.exchange(f'END {old} {instance}', path=str(path))['status'] == 'closed'
    assert motion.exchange(f'KEEP {old} {instance}', path=str(path))['status'] == 'closed'
    assert motion.exchange(f'GESTURE stale speech {old} {instance}', path=str(path))['status'] == 'closed'
    new, _ = opened(path)
    assert old != new
    assert motion.exchange(f'END {old} {instance}', path=str(path))['status'] == 'closed'
    assert motion.exchange(f'GESTURE current speech {new} wrong_instance', path=str(path))['status'] == 'instance_changed'
    assert motion.exchange(f'GESTURE current speech {new} {instance}', path=str(path))['status'] == 'accepted'
    wait_status(path, lambda s: s['state'] == 'ready' and s['completed'] == 2, timeout=5)
    assert motion.request_result('current', path=str(path))['result']['result_state'] == 'verified_completed'
    assert motion.exchange('PRESET manual open_palms', path=str(path))['status'] == 'accepted'
    motion.exchange(f'END {new} {instance}', path=str(path))
    wait_status(path, lambda s: s['state'] == 'ready' and s['completed'] == 3, timeout=5)
    assert motion.request_result('manual', path=str(path))['result']['result_state'] == 'verified_completed'


def test_mid_sequence_close_skips_next_pose_and_retraces_then_verifies_home(worker, tmp_path):
    data = json.loads((ROOT/'config/motion_presets.json').read_text())
    preset = next(p for p in data['presets'] if p['id'] == 'fly_kiss')
    preset['frames'][0]['wait_ms'] = 4000
    catalog = tmp_path/'catalog.json'
    catalog.write_text(json.dumps(data))
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(catalog))
    scope, instance = opened(path)
    motion.exchange(f'GESTURE answer fly_kiss {scope} {instance}', path=str(path))
    wait_status(path, lambda s: s['step_index'] == 1)
    # Give the first submitted segment time to settle, then cancel its hold.
    time.sleep(.4)
    motion.exchange(f'END {scope} {instance}', path=str(path))
    wait_status(path, lambda s: s['state'] == 'ready' and s['completed'] == 2, timeout=4)
    result = motion.request_result('answer', path=str(path))['result']
    assert result['result_state'] == 'cancelled_home'
    assert result['home_verified'] and result['target_verified']
    frames = [json.loads(row) for row in log.read_text().splitlines() if row.startswith('{')]
    assert len(frames) == 2  # First outward pose and captured home; no second kiss pose.
    assert frames[0]['home'] is False and frames[1]['home'] is True
    assert log.read_text().splitlines()[-1] == '1'


def test_scope_lease_expiry_requests_normal_finishing_without_power_reset(worker, tmp_path):
    data = json.loads((ROOT/'config/motion_presets.json').read_text())
    preset = next(p for p in data['presets'] if p['id'] == 'fly_kiss')
    for frame in preset['frames']:
        frame['wait_ms'] = 5000
    catalog = tmp_path/'catalog.json'
    catalog.write_text(json.dumps(data))
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(catalog))
    scope, instance = opened(path)
    motion.exchange(f'GESTURE orphan fly_kiss {scope} {instance}', path=str(path))
    state = wait_status(path, lambda s: s['state'] == 'ready' and s['completed'] == 2, timeout=15)
    assert state['at_preset'] and state['power'] == {'left': 0, 'right': 0}
    assert motion.request_result('orphan', path=str(path))['result']['result_state'] == 'cancelled_home'
    assert 'POWER_OFF' not in log.read_text()


def test_closed_scope_ids_never_reopen_after_other_scopes_are_removed(worker):
    path, _ = worker()
    old, instance = opened(path)
    motion.exchange(f'END {old} {instance}', path=str(path))
    for _ in range(70):
        scope, _ = opened(path)
        motion.exchange(f'END {scope} {instance}', path=str(path))
    assert motion.exchange(f'KEEP {old} {instance}', path=str(path))['status'] == 'closed'
    assert motion.exchange(f'GESTURE old speech {old} {instance}', path=str(path))['status'] == 'closed'


def test_answer_repeats_random_choices_only_while_audio_active(worker, monkeypatch):
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'))
    # Force alternating real pool choices while retaining the no-immediate-repeat rule.
    monkeypatch.setattr(motion.random, 'choice', lambda options: options[0])
    client = motion.MotionClient(path=str(path))
    try:
        assert client.start_playback('audio', 1, 'answer_gesture', 'test')
        deadline = time.monotonic()+10
        while len([r for r in client.results() if r['result_state'] == 'verified_completed']) < 3:
            assert time.monotonic() < deadline, client.results()
            time.sleep(.05)
        client.finish_playback('audio', 1, 'test')
        deadline = time.monotonic()+5
        while client._playbacks:
            assert time.monotonic() < deadline
            time.sleep(.05)
        results = client.results()
        assert len(results) >= 3
        assert all(a['preset'] != b['preset'] for a,b in zip(results, results[1:]))
        rows = log.read_text()
        time.sleep(.3)
        assert log.read_text() == rows
        assert all(r['result_state'] in {'verified_completed', 'cancelled_home'} for r in results)
        assert not client._inflight
    finally:
        client.close()


def test_audio_end_while_open_ack_pending_never_submits_gesture(monkeypatch):
    opening, release = threading.Event(), threading.Event()
    commands = []
    def exchange(command, **kwargs):
        commands.append(command)
        if command == 'STATUS':
            return {'status':'ok', 'instance_id':'host', 'state':'ready', 'at_preset':True}
        if command.startswith('OPEN '):
            opening.set()
            assert release.wait(2)
            return {'status':'accepted', 'instance_id':'host', 'scope_id':'1'}
        if command.startswith('END '):
            return {'status':'closed'}
        raise AssertionError(command)
    monkeypatch.setattr(motion, 'exchange', exchange)
    client = motion.MotionClient()
    client.start_playback('audio', 1, 'answer_gesture')
    assert opening.wait(1)
    client.finish_playback('audio', 1)
    release.set()
    deadline = time.monotonic()+2
    while client._playbacks:
        assert time.monotonic() < deadline
        time.sleep(.01)
    client.close()
    assert commands == ['STATUS', 'OPEN host', 'END 1 host']


def test_lost_gesture_ack_never_replays_or_starts_another_gesture(monkeypatch):
    commands = []
    def exchange(command, **kwargs):
        commands.append(command)
        if command == 'STATUS':
            return {'status':'ok', 'instance_id':'host', 'state':'ready', 'at_preset':True}
        if command.startswith('OPEN '):
            return {'status':'accepted', 'instance_id':'host', 'scope_id':'1'}
        if command.startswith('KEEP '): return {'status':'ok'}
        if command.startswith('GESTURE '): raise TimeoutError('lost ACK')
        if command.startswith('END '): return {'status':'closed'}
        if command.startswith('RESULT '): return {'status':'not_found'}
        raise AssertionError(command)
    monkeypatch.setattr(motion, 'exchange', exchange)
    client = motion.MotionClient()
    client.start_playback('audio', 1, 'answer_gesture')
    deadline = time.monotonic()+2
    while client._playbacks:
        assert time.monotonic() < deadline
        time.sleep(.01)
    client.close()
    assert sum(c.startswith('GESTURE ') for c in commands) == 1
    assert any(c.startswith('END ') for c in commands)
    assert client.results()[0]['result_state'] == 'unknown'


def test_builtin_stops_at_internal_segment_boundary_and_returns_without_more_segments(worker, tmp_path):
    segment_log = tmp_path/'segments.log'
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'),
                       MOTION_TEST_VENDOR_SEGMENTS='4', MOTION_TEST_SEGMENT_LOG=str(segment_log),
                       MOTION_TEST_DURATION_MS='350')
    scope, instance = opened(path)
    motion.exchange(f'GESTURE speech speech {scope} {instance}', path=str(path))
    deadline = time.monotonic()+2
    while not segment_log.exists():
        assert time.monotonic() < deadline
        time.sleep(.01)
    motion.exchange(f'END {scope} {instance}', path=str(path))
    done = wait_status(path, lambda s: s['state'] == 'ready' and s['completed'] == 2, timeout=5)
    assert done['at_preset']
    assert segment_log.read_text().splitlines() == ['0']
    result = motion.request_result('speech', path=str(path))['result']
    assert result['result_state'] == 'cancelled_home'
    assert result['home_verified'] and not result['process_verified']
    assert 'POWER_OFF' not in log.read_text()


def test_driver_fault_during_finishing_is_latched_and_never_marks_cancelled_home(worker):
    path, _ = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'),
                     MOTION_TEST_DURATION_MS='600')
    scope, instance = opened(path)
    motion.exchange(f'GESTURE speech speech {scope} {instance}', path=str(path))
    wait_status(path, lambda s: s['step_index'] == 1)
    motion.exchange(f'END {scope} {instance}', path=str(path))
    motion.exchange('TEST_DRIVER_ERROR failed_during_return', path=str(path))
    wait_status(path, lambda s: s['state'] == 'fault')
    time.sleep(.7)
    assert motion.request_result('speech', path=str(path))['result']['result_state'] == 'failed'
    assert motion.exchange('STATUS', path=str(path))['state'] == 'fault'


def test_audio_end_while_wake_busy_does_not_dispatch_after_ready(monkeypatch):
    commands = []
    busy = threading.Event()
    def exchange(command, **kwargs):
        commands.append(command)
        if command == 'STATUS':
            busy.set()
            return {'status':'ok', 'instance_id':'host', 'state':'running', 'at_preset':False}
        if command.startswith('OPEN '): return {'status':'accepted', 'instance_id':'host', 'scope_id':'1'}
        if command.startswith('KEEP '): return {'status':'ok'}
        if command.startswith('END '): return {'status':'closed'}
        raise AssertionError(command)
    monkeypatch.setattr(motion, 'exchange', exchange)
    client = motion.MotionClient()
    client.start_playback('audio', 1, 'answer_gesture')
    assert busy.wait(1)
    client.finish_playback('audio', 1)
    client.close()
    assert not any(command.startswith('GESTURE ') for command in commands)


def wait_client(client, predicate, timeout=6):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError(client.results())


def test_farewell_audio_drain_completes_entire_gesture_once(worker, tmp_path, monkeypatch):
    data = json.loads((ROOT/'config/motion_presets.json').read_text())
    next(p for p in data['presets'] if p['id']=='fly_kiss')['frames'][0]['wait_ms'] = 600
    catalog = tmp_path/'farewell.json'
    catalog.write_text(json.dumps(data))
    path, log = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(catalog))
    monkeypatch.setattr(motion.random, 'choice', lambda options: 'fly_kiss')
    client = motion.MotionClient(path=str(path))
    try:
        client.start_playback('bye', 1, 'closing_gesture')
        wait_client(client, lambda: bool(client.results()) and client.results()[0]['result_state']=='running')
        client.finish_playback('bye', 1, reason='closing_completed')
        client.finish_playback('bye', 1, reason='closing_completed')  # gateway finally is idempotent
        wait_client(client, lambda: not client._playbacks)
        assert len(client.results()) == 1
        assert client.results()[0]['result_state'] == 'verified_completed'
        frames = [json.loads(row) for row in log.read_text().splitlines() if row.startswith('{')]
        assert len(frames) == 3 and [frame['home'] for frame in frames] == [False,False,True]
    finally:
        client.close()


def test_late_closing_classification_completes_current_action_without_repeating(worker, monkeypatch):
    path, _ = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'),
                     MOTION_TEST_DURATION_MS='500')
    monkeypatch.setattr(motion, 'select_motion_preset', lambda purpose, previous=None: 'speech')
    client = motion.MotionClient(path=str(path))
    try:
        client.start_playback('late_bye', 1, 'answer_gesture')
        wait_client(client, lambda: bool(client.results()) and client.results()[0]['result_state']=='running')
        client.finish_playback('late_bye', 1, reason='closing_completed')
        wait_client(client, lambda: not client._playbacks)
        assert len(client.results()) == 1
        assert client.results()[0]['result_state'] == 'verified_completed'
    finally:
        client.close()


@pytest.mark.parametrize('reason', ['playback_failed', 'interrupted'])
def test_farewell_failure_or_interruption_still_finishes_safely(worker, monkeypatch, reason):
    path, _ = worker(startup_home=True, GREENRIVER_MOTION_PRESETS=str(ROOT/'config/motion_presets.json'),
                     MOTION_TEST_DURATION_MS='500', MOTION_TEST_VENDOR_SEGMENTS='3')
    monkeypatch.setattr(motion, 'select_motion_preset', lambda purpose, previous=None: 'wave_goodbye')
    client = motion.MotionClient(path=str(path))
    try:
        client.start_playback('bye', 1, 'closing_gesture')
        wait_client(client, lambda: bool(client.results()) and client.results()[0]['result_state']=='running')
        client.finish_playback('bye', 1, reason=reason)
        wait_client(client, lambda: not client._playbacks)
        assert client.results()[0]['result_state'] == 'cancelled_home'
    finally:
        client.close()
