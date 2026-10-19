from __future__ import annotations

import json

from rag_system.contracts import ModuleResult, load_wav_pcm16, pcm16_to_wav_bytes
from rag_system.vad import FsmnVadEngine, VadEvent, VadNode
from rag_system.vad.cli import main
from rag_system.voice.vad import FsmnVadEngine as LegacyFsmnVadEngine


class FakeVadSession:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, bool]] = []
        self.finished = False
        self.all_pcm = bytearray()

    def feed_pcm16(self, pcm: bytes, *, is_final: bool = False):
        self.calls.append((pcm, is_final))
        self.all_pcm.extend(pcm)
        if len(self.calls) == 1 and not is_final:
            return [VadEvent(type="speech_start", start_ms=0)]
        if is_final:
            self.finished = True
            return [
                VadEvent(
                    type="speech_end",
                    start_ms=0,
                    end_ms=150,
                    reason="final",
                    audio_pcm=bytes(self.all_pcm),
                )
            ]
        return []


class FakeVadEngine:
    def __init__(self, session: FakeVadSession) -> None:
        self.session = session
        self.configs = []

    def create_session(self, config):
        self.configs.append(config)
        return self.session


def test_vad_package_reexports_existing_streaming_engine() -> None:
    assert FsmnVadEngine is LegacyFsmnVadEngine


def test_vad_node_accepts_wav_bytes_and_keeps_binary_out_of_json() -> None:
    session = FakeVadSession()
    engine = FakeVadEngine(session)
    node = VadNode(engine=engine, end_silence_ms=800)
    pcm = _pcm(900, samples=2_400)

    result = node.segment(pcm16_to_wav_bytes(pcm))

    assert result.speech_detected is True
    assert result.audio_ms == 150
    assert result.utterance_pcm == pcm
    assert result.to_dict() == {
        "speech_detected": True,
        "events": [
            {"type": "speech_start", "start_ms": 0},
            {
                "type": "speech_end",
                "start_ms": 0,
                "end_ms": 150,
                "reason": "final",
            },
        ],
    }
    assert engine.configs[0].end_silence_ms == 800
    assert [len(chunk) for chunk, _ in session.calls] == [3_200, 1_600]
    assert [is_final for _, is_final in session.calls] == [False, True]


def test_vad_node_file_run_can_write_detected_utterance(tmp_path) -> None:
    pcm = _pcm(700, samples=2_400)
    input_path = tmp_path / "question.wav"
    output_path = tmp_path / "utterance.wav"
    input_path.write_bytes(pcm16_to_wav_bytes(pcm))
    node = VadNode(engine=FakeVadEngine(FakeVadSession()))

    result = node.run(input_path, output_wav=output_path)

    assert result.ok is True
    assert result.module == "vad"
    assert result.output["speech_detected"] is True
    assert result.artifacts == {"utterance_wav": str(output_path.resolve())}
    assert load_wav_pcm16(output_path).pcm16 == pcm


def test_vad_node_invalid_audio_is_a_failure_result() -> None:
    node = VadNode(engine=FakeVadEngine(FakeVadSession()))

    result = node.run(b"not a wav")

    assert result.module == "vad"
    assert result.ok is False
    assert "valid WAV" in (result.error or "")


def test_vad_cli_prints_one_json_result_and_supports_strict_exit(capsys) -> None:
    class FakeNode:
        def __init__(self, **options) -> None:
            self.options = options

        def run(self, _path, **_options):
            return ModuleResult.success(
                "vad",
                {"speech_detected": False, "events": []},
                metrics={"audio_ms": 100},
            )

        def close(self) -> None:
            pass

    exit_code = main(
        ["--input", "speech.wav", "--require-speech"],
        node_factory=FakeNode,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["module"] == "vad"
    assert payload["ok"] is False
    assert payload["output"]["speech_detected"] is False
    assert payload["error"] == "speech was not detected"


def _pcm(value: int, *, samples: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * samples

