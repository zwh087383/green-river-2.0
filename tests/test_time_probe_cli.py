from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_probe(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    python_paths = [str(tmp_path), str(PROJECT_ROOT / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "rag_system.time_probe", *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_time_probe_cli_preserves_target_stdout_and_writes_default_report(
    tmp_path: Path,
) -> None:
    completed = _run_probe(
        tmp_path,
        "--quiet",
        "--",
        "rag_system.module_test",
        "list",
    )

    assert completed.returncode == 0
    assert len(completed.stdout.splitlines()) == 1
    target_payload = json.loads(completed.stdout)
    assert target_payload["ok"] is True
    assert target_payload["modules"]
    assert "[time-probe]" not in completed.stdout
    report_path = tmp_path / "data" / "runtime" / "time_probe" / "latest.json"
    assert completed.stderr == f"[time-probe] report={report_path}\n"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["target"] == {
        "module": "rag_system.module_test",
        "argument_count": 1,
        "exit_code": 0,
        "status": "ok",
    }
    assert report["summary"]["calls"] > 0
    assert report["summary"]["module_count"] > 0
    assert all(
        not row["module"].startswith("rag_system.time_probe")
        for row in report["functions"]
    )


def test_time_probe_cli_writes_a_report_when_target_returns_nonzero(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "reports" / "unknown-module.json"

    completed = _run_probe(
        tmp_path,
        "--output",
        str(report_path),
        "--include",
        "rag_system.module_test",
        "--quiet",
        "--",
        "rag_system.module_test",
        "unknown",
    )

    assert completed.returncode == 1
    assert len(completed.stdout.splitlines()) == 1
    target_payload = json.loads(completed.stdout)
    assert target_payload["ok"] is False
    assert target_payload["module"] == "unknown"
    assert completed.stderr == f"[time-probe] report={report_path}\n"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["target"] == {
        "module": "rag_system.module_test",
        "argument_count": 1,
        "exit_code": 1,
        "status": "failed",
    }
    assert report["summary"]["include_modules"] == ["rag_system.module_test"]
    assert report["functions"]


def test_time_probe_cli_forwards_arguments_and_arbitrary_exit_code(
    tmp_path: Path,
) -> None:
    package = tmp_path / "probe_fixture_target"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "from .worker import run\n"
        "raise SystemExit(run())\n",
        encoding="utf-8",
    )
    (package / "worker.py").write_text(
        "import sys\n"
        "\n"
        "def run():\n"
        "    assert sys.argv[1:] == ['--mode', 'private-argument']\n"
        "    print('target-only')\n"
        "    return 7\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "custom-target.json"

    completed = _run_probe(
        tmp_path,
        "--output",
        str(report_path),
        "--include",
        "probe_fixture_target",
        "--quiet",
        "--",
        "probe_fixture_target",
        "--mode",
        "private-argument",
    )

    assert completed.returncode == 7
    assert completed.stdout == "target-only\n"
    assert completed.stderr == f"[time-probe] report={report_path}\n"

    serialized_report = report_path.read_text(encoding="utf-8")
    report = json.loads(serialized_report)
    assert report["target"] == {
        "module": "probe_fixture_target",
        "argument_count": 2,
        "exit_code": 7,
        "status": "failed",
    }
    assert "private-argument" not in serialized_report
    assert any(
        row["qualified_name"] == "probe_fixture_target.worker.run"
        for row in report["functions"]
    )
