from __future__ import annotations

import importlib.util
import json
import queue
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def gateway(monkeypatch):
    for name in ("rclpy", "rclpy.node", "std_msgs", "std_msgs.msg", "rag_interfaces", "rag_interfaces.msg"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["rclpy.node"].Node = object
    sys.modules["std_msgs.msg"].String = object
    sys.modules["rag_interfaces.msg"].VoiceControl = object
    sys.modules["rag_interfaces.msg"].VoicePlayback = object
    path = Path(__file__).parents[1] / 'ros2/voice/voice/realtime_gateway_node.py'
    spec = importlib.util.spec_from_file_location('gateway_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.RealtimeVoiceGateway.__new__(module.RealtimeVoiceGateway)
    node._output_lock = threading.Lock()
    node._lock = threading.RLock()
    node._send_lock = threading.RLock()
    node._capture_lock = threading.Lock()
    node._capture_epoch = 0
    node._capture = threading.Event()
    node._overflow = threading.Event()
    node._audio = queue.Queue(maxsize=50)
    node._interrupt_pending = threading.Event()
    node._last_hardware_sequence = 0
    node._playback_guard_s = 0
    node._cancel = threading.Event()
    node.wake_source = 'hardware'
    node.interrupt_source = 'hardware'
    node._wake_phase = 'answer'
    node._stop = threading.Event()
    node._generation = 4
    node._output = node._response = None
    node._motion_client = Mock()
    node.base = 'http://localhost:8000'
    node.conversation_id = 'robot-test'
    node.session_id = 'robot'
    node._device = lambda name: None
    node.get_logger = lambda: SimpleNamespace(error=lambda message: None, info=lambda message: None)
    actions = []
    node._show = lambda text: actions.append(('display', text))
    node._feedback = lambda r, g, state: actions.append(('playback', state))
    node._gesture = lambda *args: None
    node._send = lambda payload: actions.append(('control', payload['type']))
    return module, node, actions


def test_closed_gate_discards_pcm_and_invalidates_dequeued_frame(gateway):
    _, node, actions = gateway
    node._set_capture(True)
    node._callback(b'old', 1, None, None)
    dequeued = node._audio.get_nowait()
    node._callback(b'queued', 1, None, None)
    node._set_capture(False)
    node._callback(b'echo', 1, None, None)
    assert node._audio.empty()
    node._set_capture(True)
    assert not node._send_audio_frame(dequeued)
    assert not actions
    sent = []
    node._send = sent.append
    node._callback(b'new-user-audio', 1, None, None)
    assert node._send_audio_frame(node._audio.get_nowait())
    assert sent == [b'new-user-audio']


def test_serial_wake_wins_over_old_continue_and_duplicate_is_ignored(gateway):
    _, node, actions = gateway
    sent = []
    node._connection_id = 'active-connection'
    node._socket = SimpleNamespace(send=lambda data: sent.append(json.loads(data)))
    node._hardware_wake({'sequence': 1})
    node._hardware_wake({'sequence': 2})
    assert node._interrupt_pending.is_set()
    assert not node._finish_turn('continue', 4, node._cancel, played=False)
    assert len(sent) == 1 and sent[0]['type'] == 'hardware_wake'
    assert not actions


def test_guard_is_interruptible_and_does_not_hold_output_or_send_lock(gateway):
    _, node, actions = gateway
    node._playback_guard_s = 1.5
    node._connection_id = 'active-connection'
    sent = []
    node._socket = SimpleNamespace(send=lambda data: sent.append(json.loads(data)))
    waiting = threading.Event()
    class Cancel:
        def is_set(self): return False
        def wait(self, timeout):
            waiting.set()
            return node._interrupt_pending.wait(timeout)
    cancel = Cancel()
    result = []
    worker = threading.Thread(target=lambda: result.append(node._finish_turn('continue', 4, cancel, played=True)))
    worker.start()
    assert waiting.wait(1)
    assert node._output_lock.acquire(timeout=.1)
    node._output_lock.release()
    node._hardware_wake({'sequence': 1})
    worker.join(.5)
    assert not worker.is_alive() and result == [False]
    assert len(sent) == 1 and not actions


def test_guard_drops_echo_before_continue_without_reopening_capture(gateway):
    module, node, actions = gateway
    node._playback_guard_s = .3
    now = [0.]
    module_clock = SimpleNamespace(monotonic=lambda: now[0])
    module.time = module_clock
    class Cancel:
        def is_set(self): return False
        def wait(self, seconds):
            assert node._wake_phase == 'answer'
            node._callback(b'echo', 1, None, None)
            now[0] += seconds
    assert node._finish_turn('continue', 4, Cancel(), played=True)
    assert now[0] >= .3
    assert actions == [('control', 'continue')]
    assert node._audio.empty() and not node._capture.is_set()


@pytest.mark.parametrize('control', ['continue', 'resume', 'prompt_finished'])
def test_cancelled_worker_cannot_advance_session(gateway, control):
    _, node, actions = gateway
    assert not node._finish_turn(control, 3, node._cancel, played=True)
    assert not actions


@pytest.mark.parametrize('source,played', [('software', True), ('hardware', False)])
def test_software_kws_and_no_audio_turn_do_not_wait(gateway, source, played):
    _, node, actions = gateway
    node.wake_source = source
    node._playback_guard_s = 1.5
    cancel = Mock()
    cancel.is_set.return_value = False
    assert node._finish_turn('continue', 4, cancel, played=played)
    cancel.wait.assert_not_called()
    assert actions == [('control', 'continue')]


@pytest.mark.parametrize('query', ['好', '是', '对', '不', '量。', '请介绍一下长江源'])
def test_short_text_is_not_filtered(query):
    from rag_system.generation.service import local_turn_control
    assert local_turn_control(query) == 'answer'


@pytest.mark.parametrize('source', ['hardware', 'software'])
def test_event_flow_keeps_kws_but_isolates_hardware_answer_pcm(gateway, source):
    _, node, _ = gateway
    node.wake_source = source
    node.interrupt_source = source
    node._hardware = None
    node._workers = Mock()
    node._handle_voice_event({'type': 'armed'})
    assert node._capture.is_set() == (source == 'software')
    node._handle_voice_event({'type': 'listening'})
    assert node._capture.is_set()
    node._callback(b'user', 1, None, None)
    node._handle_voice_event({'type': 'final', 'text': '好'})
    assert node._capture.is_set() == (source == 'software')
    assert node._wake_phase == 'answer'
    assert node._workers.submit.call_args.args[1]['text'] == '好'
    if source == 'hardware':
        node._callback(b'robot-answer', 1, None, None)
        assert node._audio.empty()


def test_hybrid_wake_keeps_initial_serial_and_routes_answer_pcm_to_software(gateway):
    _, node, _ = gateway
    node.interrupt_source = 'software'
    node._hardware = None
    node._workers = Mock()
    node._connection_id = 'active-connection'
    sent = []
    node._socket = SimpleNamespace(send=lambda data: sent.append(json.loads(data)))
    node._handle_voice_event({'type': 'armed'})
    assert not node._capture.is_set()
    node._hardware_wake({'sequence': 1})
    assert len(sent) == 1
    node._handle_voice_event({'type': 'wake_detected'})
    node._handle_voice_event({'type': 'listening'})
    node._callback(b'question-tail', 1, None, None)
    old = node._audio.get_nowait()
    node._handle_voice_event({'type': 'final', 'text': 'hello'})
    assert node._capture.is_set() and not node._send_audio_frame(old)
    node._hardware_wake({'sequence': 2})
    assert len(sent) == 1 and not node._interrupt_pending.is_set()
    node._callback(b'answer-kws', 1, None, None)
    assert node._audio.get_nowait()[1] == b'answer-kws'
    old_cancel = node._cancel
    node._handle_voice_event({'type': 'wake_detected', 'reason': 'interrupt'})
    assert old_cancel.is_set() and not node._capture.is_set()
    node._handle_voice_event({'type': 'armed'})
    node._hardware_wake({'sequence': 3})
    assert len(sent) == 2


@pytest.mark.parametrize('stage', ['generating', 'playing', 'guard'])
def test_hardware_interrupt_cancels_old_reply_and_starts_ack(gateway, stage):
    _, node, actions = gateway
    node._hardware = None
    node._workers = Mock()
    node._connection_id = 'active-connection'
    wire = []
    node._socket = SimpleNamespace(send=lambda data: wire.append(json.loads(data)))
    old_cancel = node._cancel
    if stage == 'playing':
        node._output = SimpleNamespace(abort=lambda: actions.append(('device', 'aborted')))
        node._response = SimpleNamespace(close=lambda: actions.append(('http', 'closed')))
    node._hardware_wake({'sequence': 1})
    assert not node._finish_turn('continue', 4, old_cancel, played=False)
    node._handle_voice_event({'type': 'wake_detected', 'reason': 'interrupt'})
    assert old_cancel.is_set() and node._generation == 5
    assert not node._interrupt_pending.is_set() and not node._capture.is_set()
    args = node._workers.submit.call_args.args
    assert args[0] == node._prompt and args[1] is True
    assert not node._finish_turn('continue', 4, old_cancel, played=False)
    assert node._finish_turn('prompt_finished', 5, node._cancel, played=False)
    node._handle_voice_event({'type': 'listening'})
    assert node._capture.is_set()
    assert len(wire) == 1
    if stage == 'playing':
        assert ('device', 'aborted') in actions and ('http', 'closed') in actions


def test_sender_rechecks_epoch_after_waiting_for_send_lock(gateway):
    _, node, _ = gateway
    node._set_capture(True)
    node._callback(b'old', 1, None, None)
    frame = node._audio.get_nowait()
    sent, result = [], []
    node._send = sent.append
    with node._send_lock:
        worker = threading.Thread(target=lambda: result.append(node._send_audio_frame(frame)))
        worker.start()
        node._set_capture(False)
        node._set_capture(True)
    worker.join(1)
    assert not worker.is_alive() and result == [False] and not sent


def test_generation_invalidation_is_serialized_with_final_control(gateway):
    _, node, actions = gateway
    entered, finished = threading.Event(), threading.Event()
    old_cancel = node._cancel
    def invalidate():
        entered.set()
        node._advance()
        finished.set()
    with node._send_lock:
        worker = threading.Thread(target=invalidate)
        worker.start()
        assert entered.wait(1)
        assert not finished.wait(.02)
        assert node._generation == 4
    worker.join(1)
    assert finished.is_set() and old_cancel.is_set()
    assert not node._finish_turn('continue', 4, old_cancel, played=False)
    assert not actions


def test_gateway_rearming_clears_stale_listening_display(gateway):
    _, node, actions = gateway
    node._wake_phase = 'answer'
    node._hardware = SimpleNamespace(state={})
    node._update_voice_phase('armed')
    assert node._hardware.state['gateway_phase'] == 'armed'
    assert actions == [('display', '[idle] 待唤醒，请说小绿小绿')]
    node._update_voice_phase('armed')
    assert len(actions) == 1
    node._update_voice_phase('partial')
    assert node._wake_phase == 'armed'
    node._update_voice_phase('listening')
    assert node._hardware.state['gateway_phase'] == 'listening'
    assert actions[-1] == ('display', '[query] 正在聆听，请说话')


@pytest.mark.parametrize('mode', ['complete', 'closing', 'late_closing', 'invite', 'options', 'cancelled', 'truncated', 'missing_audio'])
def test_gateway_completion_requires_provider_final_and_audio_drain(gateway, monkeypatch, mode):
    module, node, actions = gateway
    node._motion_client.start_playback.side_effect = lambda request_id, generation, purpose, session: actions.append(('gesture', purpose))
    node._motion_client.finish_playback.side_effect = lambda *args: actions.append(('motion', 'ended'))
    cancel = threading.Event()
    class Output:
        def __init__(self, **kwargs):
            assert kwargs['samplerate'] == 24000
        def __enter__(self): return self
        def __exit__(self, *args): actions.append(('device', 'closed'))
        def write(self, pcm):
            actions.append(('device', 'write'))
            assert pcm == b'\0\0'
            if mode == 'cancelled': cancel.set()
        def stop(self): actions.append(('device', 'drained'))
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(RawOutputStream=Output))
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): actions.append(('http', 'closed'))
        def raise_for_status(self): pass
        def iter_lines(self):
            yield json.dumps({'type':'meta','retrieval':{}})
            yield json.dumps({'type':'emotion','tone':'shy'})
            if mode == 'closing':
                yield json.dumps({'type':'meta','retrieval':{'reason':'end_session'}})
            if mode in {'invite', 'options'}:
                text = '欢迎一起参观我们的展厅。' if mode == 'invite' else '有两种方案可以选择。'
                yield json.dumps({'type':'delta','text':text})
            if mode != 'missing_audio':
                yield json.dumps({'type':'audio','audio':'AAA=','sample_rate':24000,'channels':1,'format':'pcm16'})
                yield json.dumps({'type':'delta','text':'欢迎一起参观。现在有两种方案。'})
                yield json.dumps({'type':'emotion','tone':'confused'})
                yield json.dumps({'type':'audio','audio':'AAA=','sample_rate':24000,'channels':1,'format':'pcm16'})
            if mode != 'truncated':
                yield json.dumps({'type':'final','result':{'answer':'hello', 'retrieval':{'reason':'end_session' if mode == 'late_closing' else 'answer'}}})
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def stream(self, method, url, json):
            assert json['conversation_id'] == 'robot-test'
            return Response()
    monkeypatch.setattr(module.httpx, 'Client', Client)
    node._reply({'text':'question'},4,cancel)
    assert ('http','closed') in actions
    assert ('display', '[tone] shy') in actions
    if mode in {'complete', 'closing', 'late_closing', 'invite', 'options'}:
        purpose = {'closing':'closing_gesture', 'invite':'answer_invite', 'options':'answer_open_palms'}.get(mode, 'answer_gesture')
        gesture = ('gesture', purpose)
        assert actions.count(gesture) == 1
        assert len([a for a in actions if a[0]=='gesture']) == 1
        assert actions.index(('device','write')) < actions.index(gesture) < actions.index(('device','drained'))
        assert actions.index(('device','drained')) < actions.index(('playback','completed'))
        assert actions.index(('device','drained')) < actions.index(('motion','ended')) < actions.index(('playback','completed'))
        assert ('control','resume' if mode in {'closing', 'late_closing'} else 'continue') in actions
        expected_reason = 'closing_completed' if mode in {'closing', 'late_closing'} else 'playback_completed'
        assert node._motion_client.finish_playback.call_args.args[-1] == expected_reason
        assert actions[-1] == ('motion','ended')
    else:
        assert ('playback','completed') not in actions
        assert ('control','continue') not in actions
    if mode in {'truncated', 'missing_audio'}:
        assert ('playback','failed') in actions
    if mode in {'cancelled', 'missing_audio'}:
        assert not any(a[0] == 'gesture' for a in actions)


def test_gateway_interrupt_cancels_http_and_aborts_pcm(gateway):
    module, node, actions = gateway
    old_cancel = threading.Event()
    node._cancel = old_cancel
    node._output = SimpleNamespace(abort=lambda: actions.append(('device','aborted')))
    node._response = SimpleNamespace(close=lambda: actions.append(('http','closed')))
    generation, new_cancel = node._advance()
    assert old_cancel.is_set() and not new_cancel.is_set()
    assert generation == 5
    node._motion_client.advance.assert_called_once_with(5)
    assert ('device','aborted') in actions and ('http','closed') in actions


@pytest.mark.parametrize('source', ['hardware', 'software'])
def test_echo_cancellation_is_kept_for_both_interrupt_sources(gateway, monkeypatch, source):
    _, node, _ = gateway
    node.interrupt_source = source
    node.get_parameter = lambda name: SimpleNamespace(value=True)
    route = {'source':'greenriver_voice_aec_source','sink':'greenriver_voice_aec_sink'}
    prepare = Mock(return_value=route)
    monkeypatch.setattr('rag_system.voice.echo_cancel.prepare_echo_cancel', prepare)
    assert node._prepare_audio_route() == route
    prepare.assert_called_once_with()


def test_disabled_aec_preserves_direct_hardware_audio_route(gateway, monkeypatch):
    _, node, _ = gateway
    node.get_parameter = lambda name: SimpleNamespace(value=False)
    monkeypatch.setenv('PULSE_SOURCE', 'physical_iflytek')
    monkeypatch.setenv('PULSE_SINK', 'physical_speaker')
    prepare = Mock()
    monkeypatch.setattr('rag_system.voice.echo_cancel.prepare_echo_cancel', prepare)
    assert node._prepare_audio_route() is None
    prepare.assert_not_called()
    import os
    assert os.environ['PULSE_SOURCE'] == 'physical_iflytek'
    assert os.environ['PULSE_SINK'] == 'physical_speaker'


def test_gateway_gesture_calls_native_client_and_interrupt_only_invalidates(gateway):
    module, node, _ = gateway
    node.session_id = 'robot'
    module.RealtimeVoiceGateway._gesture(node, 10, 'wake_wave')
    module.RealtimeVoiceGateway._gesture(node, 11, 'answer_gesture')
    module.RealtimeVoiceGateway._gesture(node, 12, 'interrupt', True)
    calls = node._motion_client.start_playback.call_args_list
    assert [(c.args[1], c.args[2], c.args[3]) for c in calls] == [
        (10, 'wake_wave', 'robot'), (11, 'answer_gesture', 'robot')]
    assert all(c.args[0] for c in calls)
    node._motion_client.advance.assert_called_once_with(12)


@pytest.mark.parametrize('phase,expected', [('armed', 1), ('answer', 1), ('prompt', 0),
                                         ('listening', 0), ('disconnected', 0)])
def test_hardware_wake_forwarding_is_scoped_to_active_connection(gateway, phase, expected):
    _, node, _ = gateway
    sent = []
    node._wake_phase, node._connection_id = phase, 'connection'
    node._send_lock = threading.Lock()
    node._socket = SimpleNamespace(send=lambda data: sent.append(json.loads(data)))
    node.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    node._hardware_wake({'sequence': 9})
    assert len(sent) == expected
    if expected:
        assert sent == [{'type': 'hardware_wake', 'connection_id': 'connection', 'sequence': 9}]
    node._connection_id = ''
    node._hardware_wake({'sequence': 10})
    assert len(sent) == expected
