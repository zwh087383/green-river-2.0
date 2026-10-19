from __future__ import annotations

import json

from rag_system.voice.models import ModelVerification


def test_model_setup_parser_accepts_wakeup_component() -> None:
    from rag_system.model_setup import _build_parser

    args = _build_parser().parse_args(["verify", "--component", "wakeup"])

    assert args.action == "verify"
    assert args.component == "wakeup"


def test_model_setup_routes_wakeup_verification(monkeypatch, tmp_path, capsys) -> None:
    from rag_system import model_setup

    target = tmp_path / "wakeup"
    expected = ModelVerification(
        model_id="test/wakeup",
        revision="test",
        path=str(target),
        ready=True,
        size_bytes=123,
        sha256="abc",
        errors=[],
    )
    monkeypatch.setattr(model_setup, "verify_wakeup_model", lambda path: expected)

    exit_code = model_setup.main(
        ["verify", "--component", "wakeup", "--target", str(target)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {"wakeup": expected.to_dict()}
