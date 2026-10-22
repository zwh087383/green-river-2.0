from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def desktop(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("desktop_runtime_under_test", scripts / "desktop_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_does_not_restart_healthy_existing_robot(desktop, monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "project_counts", lambda _: dict.fromkeys(desktop.EXPECTED, 1))
    monkeypatch.setattr(desktop, "healthy", lambda: True)
    launch = Mock(side_effect=AssertionError("Must not restart the running robot"))
    monkeypatch.setattr(desktop.subprocess, "Popen", launch)
    desktop.start(tmp_path)
    launch.assert_not_called()


def test_duplicate_or_partial_runtime_is_not_complete(desktop):
    summary = dict.fromkeys(desktop.EXPECTED, 1)
    assert desktop.complete({**summary, "browser_child": 8})
    assert not desktop.complete({**summary, "voice_gateway": 2})
    assert not desktop.complete({**summary, "web": 0})
    assert not desktop.complete({**summary, "old_voice_bridge": 1})


def test_new_runtime_is_detached_from_the_desktop_terminal(desktop, monkeypatch, tmp_path):
    runtime = tmp_path / "data/runtime"
    runtime.mkdir(parents=True)
    summaries = iter([{}, dict.fromkeys(desktop.EXPECTED, 1)])
    monkeypatch.setattr(desktop, "project_counts", lambda _: next(summaries))
    monkeypatch.setattr(desktop, "healthy", lambda: True)
    def launch(command, **kwargs):
        assert command == ["/bin/bash", str(tmp_path / "scripts/run_ipc_voice.sh")]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == desktop.subprocess.DEVNULL
        assert kwargs["env"]["RAG_ROBOT_WAKE_SOURCE"] == "hardware"
        assert "GREENRIVER_INSTANCE_ID" not in kwargs["env"]
        (runtime / "runtime-instance.json").write_text(json.dumps({"supervisor_pid": 12345}))
        return SimpleNamespace(pid=12345, poll=lambda: None)
    monkeypatch.setattr(desktop.subprocess, "Popen", launch)
    desktop.start(tmp_path)


def test_stop_delegates_to_scoped_runtime_and_checks_residuals(desktop, monkeypatch, tmp_path):
    command = Mock()
    monkeypatch.setattr(desktop.subprocess, "run", command)
    monkeypatch.setattr(desktop, "project_counts", lambda _: {})
    desktop.stop(tmp_path)
    args = command.call_args.args[0]
    assert args[:3] == ["/usr/bin/python3", str(tmp_path / "scripts/robot_runtime.py"), "stop"]
    assert args[3:] == ["--project-root", str(tmp_path), "--legacy-root", str(Path.home() / "Voice")]
    assert command.call_args.kwargs["check"] is True
    monkeypatch.setattr(desktop, "project_counts", lambda _: {"web": 1})
    with pytest.raises(RuntimeError, match="残留"):
        desktop.stop(tmp_path)
