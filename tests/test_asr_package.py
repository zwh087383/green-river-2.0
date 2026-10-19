from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from rag_system.asr import (
    ASRFacade,
    ASRNode,
    ASRProvider,
    DASHSCOPE_REALTIME_ASR_MODEL,
    DashScopeRealtimeASR,
    RealtimeASRTransport,
    RealtimeASRResult,
)
from rag_system.asr.cli import main
from rag_system.contracts import pcm16_to_wav_bytes


class FakeTransport:
    connected = True

    def __init__(self) -> None:
        self.closed = False

    async def append_audio(self, pcm16: bytes) -> None:
        del pcm16

    async def receive_event(self):
        return None

    async def events(self):
        if False:
            yield None

    async def finish(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeASRProvider:
    def __init__(self, text: str = "介绍一下绿色江河") -> None:
        self.text = text
        self.calls: list[tuple[bytes, str]] = []
        self.transport = FakeTransport()
        self.open_intervals: list[int] = []
        self.released: list[FakeTransport] = []

    def speech_to_text(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "recording.wav",
    ) -> str:
        self.calls.append((audio_bytes, filename))
        return self.text

    async def open_realtime_asr(self, *, end_silence_ms: int = 400, force: bool = False):
        del force
        self.open_intervals.append(end_silence_ms)
        return self.transport

    async def release_realtime_asr(self, transport: FakeTransport) -> None:
        self.released.append(transport)
        await transport.close()


def test_asr_node_accepts_wav_bytes_and_file(tmp_path) -> None:
    provider = FakeASRProvider()
    node = ASRNode(voice_service=provider)
    memory_wav = pcm16_to_wav_bytes(b"\x00\x00" * 800)
    file_wav = pcm16_to_wav_bytes(b"\x01\x00" * 800)

    assert node.transcribe(memory_wav, filename="memory.wav") == "介绍一下绿色江河"
    wav_path = tmp_path / "question.wav"
    wav_path.write_bytes(file_wav)
    assert node.transcribe_file(wav_path) == "介绍一下绿色江河"

    assert [filename for _audio, filename in provider.calls] == [
        "memory.wav",
        "question.wav",
    ]
    assert all(audio.startswith(b"RIFF") for audio, _filename in provider.calls)


def test_asr_node_exposes_realtime_lifecycle() -> None:
    provider = FakeASRProvider()
    node = ASRNode(provider)

    async def exercise() -> None:
        transport = await node.open_realtime(end_silence_ms=800)
        assert transport is provider.transport
        assert isinstance(transport, RealtimeASRTransport)
        await node.release_realtime(transport)

    asyncio.run(exercise())
    assert provider.open_intervals == [800]
    assert provider.released == [provider.transport]
    assert provider.transport.closed is True


def test_asr_package_keeps_stable_legacy_exports() -> None:
    assert ASRFacade is ASRNode
    assert isinstance(FakeASRProvider(), ASRProvider)
    assert DASHSCOPE_REALTIME_ASR_MODEL
    assert DashScopeRealtimeASR.__module__ == "rag_system.voice.dashscope_asr"


def test_asr_cli_prints_json_and_returns_zero(tmp_path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(pcm16_to_wav_bytes(b"\x00\x00" * 800))
    provider = FakeASRProvider(text="识别成功")
    output = io.StringIO()

    code = main(
        [str(wav_path)],
        node_factory=lambda: ASRNode(provider),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload == {
        "ok": True,
        "module": "asr",
        "provider": "local",
        "input": str(wav_path),
        "text": "识别成功",
    }


def test_asr_cli_prints_json_and_returns_nonzero_on_error(tmp_path) -> None:
    output = io.StringIO()

    code = main([str(tmp_path / "missing.wav")], stdout=output)

    payload = json.loads(output.getvalue())
    assert code == 1
    assert payload["ok"] is False
    assert payload["module"] == "asr"
    assert payload["error_type"] == "AudioInputError"


def test_asr_cli_usage_error_is_json() -> None:
    output = io.StringIO()

    code = main([], stdout=output)

    payload = json.loads(output.getvalue())
    assert code == 1
    assert payload["ok"] is False
    assert payload["error_type"] == "_CLIUsageError"


def test_asr_node_can_feed_one_wav_through_realtime_provider() -> None:
    class StreamingTransport:
        connected = True

        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.finished = False
            self.closed = False

        async def append_audio(self, pcm16: bytes) -> None:
            self.audio.append(pcm16)

        async def receive_event(self):
            return None

        async def events(self):
            yield SimpleNamespace(type="partial", text="绿色")
            while not self.finished:
                await asyncio.sleep(0)
            yield SimpleNamespace(type="completed", text="绿色江河")

        async def finish(self) -> None:
            self.finished = True

        async def close(self) -> None:
            self.closed = True

    class StreamingProvider:
        def __init__(self) -> None:
            self.transport = StreamingTransport()
            self.intervals: list[int] = []

        async def open_realtime_asr(
            self,
            *,
            end_silence_ms: int = 400,
            force: bool = False,
        ):
            self.intervals.append(end_silence_ms if force else -end_silence_ms)
            return self.transport

        async def release_realtime_asr(self, transport) -> None:
            await transport.close()

        def postprocess_asr_text(self, text: str) -> str:
            return text.strip()

    provider = StreamingProvider()
    node = ASRNode(provider)
    wav = pcm16_to_wav_bytes(b"\x00\x00" * 3_200)

    result = asyncio.run(
        node.transcribe_realtime(wav, chunk_ms=100, end_silence_ms=800, pace=False)
    )

    assert result == RealtimeASRResult(
        text="绿色江河",
        partials=("绿色",),
        event_count=2,
    )
    assert provider.intervals == [800]
    assert [len(chunk) for chunk in provider.transport.audio] == [3_200, 3_200]
    assert provider.transport.closed is True


def test_asr_cli_qwen_mode_is_still_one_json_command(tmp_path) -> None:
    wav_path = tmp_path / "qwen.wav"
    wav_path.write_bytes(pcm16_to_wav_bytes(b"\x00\x00" * 1_600))
    output = io.StringIO()

    class FakeRealtimeNode:
        async def transcribe_realtime_file(self, _path, **_options):
            return RealtimeASRResult("绿色江河", ("绿色",), 2)

    code = main(
        [str(wav_path), "--provider", "qwen"],
        node_factory=FakeRealtimeNode,
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["provider"] == "qwen"
    assert payload["text"] == "绿色江河"
    assert payload["partials"] == ["绿色"]


def test_asr_cli_qwen_stream_prints_live_json_lines(tmp_path) -> None:
    wav_path = tmp_path / "qwen-stream.wav"
    wav_path.write_bytes(pcm16_to_wav_bytes(b"\x00\x00" * 1_600))
    output = io.StringIO()

    class FakeRealtimeNode:
        async def transcribe_realtime_file(self, _path, **options):
            options["on_event"]({"type": "partial", "text": "绿色"})
            options["on_event"]({"type": "completed", "text": "绿色江河"})
            return RealtimeASRResult("绿色江河", ("绿色",), 2)

    code = main(
        [str(wav_path), "--provider", "qwen", "--stream"],
        node_factory=FakeRealtimeNode,
        stdout=output,
    )

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert code == 0
    assert [line["event"] for line in lines] == ["partial", "completed", "summary"]


def test_asr_node_does_not_mark_empty_text_as_success() -> None:
    node = ASRNode(FakeASRProvider(text="  "))
    wav = pcm16_to_wav_bytes(b"\x00\x00" * 800)

    with pytest.raises(RuntimeError, match="no recognized text"):
        node.transcribe(wav)
