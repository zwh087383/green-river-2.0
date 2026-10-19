from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    node._stop = threading.Event()
    node._generation = 4
    node._output = node._response = None
    node.base = 'http://localhost:8000'
    node.conversation_id = 'robot-test'
    node._device = lambda name: None
    node.get_logger = lambda: SimpleNamespace(error=lambda message: None)
    actions = []
    node._show = lambda text: actions.append(('display', text))
    node._feedback = lambda r, g, state: actions.append(('playback', state))
    node._gesture = lambda *args: None
    node._send = lambda payload: actions.append(('control', payload['type']))
    return module, node, actions


@pytest.mark.parametrize('mode', ['complete', 'cancelled', 'truncated', 'missing_audio'])
def test_gateway_completion_requires_provider_final_and_audio_drain(gateway, monkeypatch, mode):
    module, node, actions = gateway
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
            if mode != 'missing_audio':
                yield json.dumps({'type':'audio','audio':'AAA=','sample_rate':24000,'channels':1,'format':'pcm16'})
            if mode != 'truncated':
                yield json.dumps({'type':'final','result':{'answer':'hello'}})
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
    if mode == 'complete':
        assert actions.index(('device','drained')) < actions.index(('playback','completed'))
        assert actions[-1] == ('control','continue')
    else:
        assert ('playback','completed') not in actions
        assert ('control','continue') not in actions
    if mode in {'truncated', 'missing_audio'}:
        assert ('playback','failed') in actions


def test_gateway_interrupt_cancels_http_and_aborts_pcm(gateway):
    module, node, actions = gateway
    old_cancel = threading.Event()
    node._cancel = old_cancel
    node._output = SimpleNamespace(abort=lambda: actions.append(('device','aborted')))
    node._response = SimpleNamespace(close=lambda: actions.append(('http','closed')))
    generation, new_cancel = node._advance()
    assert old_cancel.is_set() and not new_cancel.is_set()
    assert generation == 5
    assert ('device','aborted') in actions and ('http','closed') in actions
