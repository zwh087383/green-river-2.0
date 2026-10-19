from __future__ import annotations

import json
import shutil
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_system.agent.cli import main as agent_main
from rag_system.module_test import cli
from rag_system.module_test.doctor import build_doctor_report
from rag_system.rag_cloud.cli import main as rag_cloud_main
from rag_system.rag_edge.cli import main as rag_edge_main
from rag_system.vad.cli import main as vad_main
from rag_system.wakeup.cli import main as wakeup_main


def test_module_test_list_is_simple_machine_readable_json() -> None:
    output = StringIO()

    exit_code = cli.main(["list"], stdout=output)
    payload = json.loads(output.getvalue())

    assert exit_code == 0
    assert payload["ok"] is True
    assert [item["command"] for item in payload["modules"]] == [
        "wakeup",
        "vad",
        "asr",
        "tts",
        "rag-cloud",
        "rag-edge",
        "agent",
    ]


def test_module_test_dispatches_remaining_arguments(monkeypatch) -> None:
    calls: list[list[str]] = []
    fake_cli = SimpleNamespace(main=lambda args: calls.append(list(args)) or 7)
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: fake_cli)

    exit_code = cli.main(["asr", "question.wav"])

    assert exit_code == 7
    assert calls == [["question.wav"]]


def test_module_test_unknown_module_returns_json_error() -> None:
    output = StringIO()

    exit_code = cli.main(["unknown"], stdout=output)
    payload = json.loads(output.getvalue())

    assert exit_code == 1
    assert payload["ok"] is False
    assert "asr" in payload["available"]


def test_module_test_doctor_outputs_one_json_line_and_returns_readiness(monkeypatch) -> None:
    output = StringIO()
    report = {
        "ok": True,
        "command": "doctor",
        "offline": True,
        "modules": {"asr": {"ready": True}},
    }
    monkeypatch.setattr(cli, "build_doctor_report", lambda: report)

    exit_code = cli.main(["doctor"], stdout=output)

    assert exit_code == 0
    assert output.getvalue().count("\n") == 1
    assert json.loads(output.getvalue()) == report


def test_module_test_doctor_returns_one_when_not_ready(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        cli,
        "build_doctor_report",
        lambda: {"ok": False, "command": "doctor", "offline": True},
    )

    exit_code = cli.main(["doctor"], stdout=output)

    assert exit_code == 1
    assert json.loads(output.getvalue())["ok"] is False


def test_doctor_is_read_only_and_never_exposes_key_values(tmp_path) -> None:
    settings_dir = tmp_path / "data"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(
        json.dumps(
            {
                "dashscope_api_key": "saved-dashscope-secret",
                "deepseek_api_key": "saved-deepseek-secret",
                "asr_provider": "dashscope_realtime",
            }
        ),
        encoding="utf-8",
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    report = build_doctor_report(
        project_root=tmp_path,
        dependency_finder=lambda _name: None,
        environ={"DASHSCOPE_API_KEY": "environment-secret"},
    )

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    serialized = json.dumps(report, ensure_ascii=False)
    assert before == after
    assert report["offline"] is True
    assert report["checks"]["configuration"]["dashscope_api_key"] is True
    assert report["checks"]["configuration"]["dashscope_api_key_source"] == "env"
    assert "saved-dashscope-secret" not in serialized
    assert "saved-deepseek-secret" not in serialized
    assert "environment-secret" not in serialized


def test_doctor_rejects_extra_arguments_as_one_json_line() -> None:
    output = StringIO()

    exit_code = cli.main(["doctor", "unexpected"], stdout=output)

    assert exit_code == 1
    assert output.getvalue().count("\n") == 1
    assert json.loads(output.getvalue())["error_type"] == "InputError"


@pytest.mark.parametrize(
    ("module_main", "module"),
    [
        (wakeup_main, "wakeup"),
        (vad_main, "vad"),
        (rag_cloud_main, "rag-cloud"),
        (rag_edge_main, "rag-edge"),
        (agent_main, "agent"),
    ],
)
def test_every_module_usage_error_is_one_json_line(
    module_main,
    module: str,
    capsys,
) -> None:
    exit_code = module_main([])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["module"] == module
    assert captured.err == ""


@pytest.mark.skipif(shutil.which("powershell") is None, reason="PowerShell is unavailable")
def test_powershell_wrapper_forwards_arguments_and_exit_code() -> None:
    project_root = Path(__file__).resolve().parents[1]
    wrapper = project_root / "module-test.ps1"

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-File", str(wrapper), "unknown"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["module"] == "unknown"
