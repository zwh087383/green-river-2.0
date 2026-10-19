from __future__ import annotations

import json
from dataclasses import dataclass

from rag_system.contracts import ModuleResult, pcm16_to_wav_bytes
from rag_system.voice.wakeup import WakeWordEngine as LegacyWakeWordEngine
from rag_system.wakeup import WakeWordEngine, WakeupNode
from rag_system.wakeup.cli import main


@dataclass
class FakeWakeEvent:
    keyword: str


class FakeWakeSession:
    def __init__(self, *, detect_on_call: int = 2) -> None:
        self.detect_on_call = detect_on_call
        self.accepted: list[bytes] = []
        self.closed = False

    def accept_pcm16(self, pcm: bytes):
        self.accepted.append(pcm)
        if len(self.accepted) == self.detect_on_call:
            return FakeWakeEvent("小绿小绿")
        return None

    def close(self) -> None:
        self.closed = True


class FakeWakeEngine:
    sample_rate = 16_000

    def __init__(self, session: FakeWakeSession) -> None:
        self.session = session
        self.create_calls = 0

    def create_session(self) -> FakeWakeSession:
        self.create_calls += 1
        return self.session


def test_wakeup_package_reexports_existing_streaming_engine() -> None:
    assert WakeWordEngine is LegacyWakeWordEngine


def test_wakeup_node_accepts_wav_bytes_and_streams_chunks() -> None:
    session = FakeWakeSession(detect_on_call=2)
    node = WakeupNode(engine=FakeWakeEngine(session), chunk_ms=100)
    pcm = _pcm(800, samples=4_800)

    result = node.detect(pcm16_to_wav_bytes(pcm))

    assert result.detected is True
    assert result.keyword == "小绿小绿"
    assert result.audio_ms == 300
    assert result.processed_ms == 200
    assert [len(chunk) for chunk in session.accepted] == [3_200, 3_200]
    assert session.closed is True


def test_wakeup_node_file_run_returns_uniform_module_result(tmp_path) -> None:
    session = FakeWakeSession(detect_on_call=1)
    node = WakeupNode(engine=FakeWakeEngine(session))
    wav_path = tmp_path / "wake.wav"
    wav_path.write_bytes(pcm16_to_wav_bytes(_pcm(500, samples=1_600)))

    result = node.run(wav_path)

    assert result == ModuleResult.success(
        "wakeup",
        {"detected": True, "keyword": "小绿小绿"},
        metrics={"sample_rate": 16_000, "audio_ms": 100, "processed_ms": 100},
    )


def test_wakeup_node_invalid_audio_is_a_failure_result() -> None:
    node = WakeupNode(engine=FakeWakeEngine(FakeWakeSession()))

    result = node.run(b"not a wav")

    assert result.module == "wakeup"
    assert result.ok is False
    assert "valid WAV" in (result.error or "")


def test_wakeup_cli_prints_one_json_result_and_supports_strict_exit(capsys) -> None:
    class FakeNode:
        def __init__(self, **options) -> None:
            self.options = options

        def run(self, _path):
            return ModuleResult.success(
                "wakeup",
                {"detected": False, "keyword": None},
                metrics={"audio_ms": 100},
            )

        def close(self) -> None:
            pass

    exit_code = main(
        ["--input", "wake.wav", "--require-detected"],
        node_factory=FakeNode,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["module"] == "wakeup"
    assert payload["ok"] is False
    assert payload["output"]["detected"] is False
    assert payload["error"] == "wake word was not detected"


def _pcm(value: int, *, samples: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * samples

