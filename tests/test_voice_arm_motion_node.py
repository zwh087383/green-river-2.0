from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def node(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'ros2/voice'))
    rclpy = ModuleType('rclpy')
    rclnode = ModuleType('rclpy.node'); rclnode.Node = object
    messages = ModuleType('rag_interfaces.msg'); messages.VoiceControl = object
    monkeypatch.setitem(sys.modules, 'rclpy', rclpy)
    monkeypatch.setitem(sys.modules, 'rclpy.node', rclnode)
    monkeypatch.setitem(sys.modules, 'rag_interfaces.msg', messages)
    spec = importlib.util.spec_from_file_location('_native_arm_adapter_test', ROOT/'ros2/voice/voice/arm_motion_node.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    node = module.VoiceArmMotionNode.__new__(module.VoiceArmMotionNode)
    node._enabled = True; node._session_id = 'robot'; node._client = Mock()
    return node

def message(**kwargs):
    return SimpleNamespace(**dict(dict(command='play_motion', session_id='robot', request_id='r1',
                                      generation=2, purpose='wake_wave'), **kwargs))

def test_legacy_voice_request_uses_native_client(node):
    node._on_motion_request(message())
    node._client.request.assert_called_once_with('r1', 2, 'wake_wave', 'robot')

@pytest.mark.parametrize('kwargs', [{'session_id':'other'}, {'command':'unrelated'}])
def test_foreign_request_is_ignored(node, kwargs):
    node._on_motion_request(message(**kwargs)); node._client.request.assert_not_called()

def test_disabled_request_is_ignored(node):
    node._enabled = False; node._on_motion_request(message()); node._client.request.assert_not_called()

@pytest.mark.parametrize('command', ['interrupt', 'return_to_wakeup'])
def test_interruption_never_sends_another_motion(node, command):
    node._on_control(message(command=command))
    node._client.advance.assert_called_once_with(2)
    node._client.request.assert_not_called()
