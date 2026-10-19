from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from rag_system.tts import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_AUDIO_SUFFIX,
    DEFAULT_MEDIA_TYPE,
    DashScopeRealtimeTTS,
    TTSFacade,
    TTSNode,
    TTSProvider,
)
from rag_system.tts.cli import main


class FakeTTSProvider:
    def __init__(self, audio: bytes = b"ID3-fake-mp3") -> None:
        self.audio = audio
        self.texts: list[str] = []
        self.stream_texts: list[str] = []
        self.warmed = False
        self.closed = False

    def text_to_speech(self, text: str) -> bytes:
        self.texts.append(text)
        return self.audio

    async def _stream(self, text: str):
        self.stream_texts.append(text)
        yield b"ID3"
        yield b"-stream"

    def stream_text_to_speech(self, text: str):
        return self._stream(text)

    async def warm_tts_connection(self) -> bool:
        self.warmed = True
        return True

    async def close_tts_connection(self) -> None:
        self.closed = True


def test_tts_node_returns_mp3_bytes_and_file(tmp_path) -> None:
    provider = FakeTTSProvider()
    node = TTSNode(voice_service=provider)

    assert node.synthesize("你好") == b"ID3-fake-mp3"
    output = node.synthesize_to_file("绿色江河", tmp_path / "answer")

    assert output == tmp_path / "answer.mp3"
    assert output.read_bytes() == b"ID3-fake-mp3"
    assert provider.texts == ["你好", "绿色江河"]


def test_tts_node_creates_the_customer_output_directory(tmp_path) -> None:
    node = TTSNode(FakeTTSProvider())

    output = node.synthesize_to_file("你好", tmp_path / "nested" / "answer.mp3")

    assert output.is_file()


def test_tts_node_does_not_mark_empty_audio_as_success() -> None:
    node = TTSNode(FakeTTSProvider(audio=b""))

    with pytest.raises(RuntimeError, match="no audio"):
        node.synthesize("你好")


def test_tts_node_rejects_a_misleading_wav_suffix(tmp_path) -> None:
    node = TTSNode(FakeTTSProvider())

    with pytest.raises(ValueError, match="currently outputs MP3"):
        node.synthesize_to_file("你好", tmp_path / "answer.wav")


def test_tts_node_exposes_stream_and_provider_lifecycle() -> None:
    provider = FakeTTSProvider()
    node = TTSNode(provider)

    async def exercise() -> bytes:
        chunks = [chunk async for chunk in node.stream("流式测试")]
        assert await node.warm() is True
        await node.close()
        return b"".join(chunks)

    assert asyncio.run(exercise()) == b"ID3-stream"
    assert provider.stream_texts == ["流式测试"]
    assert provider.warmed is True
    assert provider.closed is True


def test_tts_package_has_explicit_format_and_stable_exports() -> None:
    assert TTSFacade is TTSNode
    assert isinstance(FakeTTSProvider(), TTSProvider)
    assert DEFAULT_AUDIO_FORMAT == "mp3"
    assert DEFAULT_AUDIO_SUFFIX == ".mp3"
    assert DEFAULT_MEDIA_TYPE == "audio/mpeg"
    assert DashScopeRealtimeTTS.__module__ == "rag_system.voice.dashscope_tts"


def test_tts_cli_writes_mp3_and_prints_json(tmp_path) -> None:
    output_path = tmp_path / "spoken.mp3"
    output = io.StringIO()
    provider = FakeTTSProvider(audio=b"mp3-content")

    code = main(
        ["甲方测试", "--output", str(output_path)],
        node_factory=lambda: TTSNode(provider),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert output_path.read_bytes() == b"mp3-content"
    assert payload == {
        "ok": True,
        "module": "tts",
        "output": str(output_path),
        "format": "mp3",
        "media_type": "audio/mpeg",
        "bytes": 11,
    }


def test_tts_cli_can_exercise_the_streaming_provider(tmp_path) -> None:
    output_path = tmp_path / "streamed.mp3"
    output = io.StringIO()
    provider = FakeTTSProvider()

    code = main(
        ["流式测试", "--stream", "--output", str(output_path)],
        node_factory=lambda: TTSNode(provider),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["streaming"] is True
    assert output_path.read_bytes() == b"ID3-stream"
    assert provider.closed is True


def test_tts_cli_can_require_qwen_without_hiding_edge_fallback(tmp_path) -> None:
    output_path = tmp_path / "fallback.mp3"
    output = io.StringIO()

    class EdgeFallbackProvider(FakeTTSProvider):
        def status(self):
            return SimpleNamespace(tts_provider_active="edge")

    code = main(
        [
            "流式测试",
            "--stream",
            "--require-provider",
            "qwen",
            "--output",
            str(output_path),
        ],
        node_factory=lambda: TTSNode(EdgeFallbackProvider()),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 2
    assert payload["ok"] is False
    assert payload["provider"] == "edge"
    assert payload["error_type"] == "ProviderMismatch"


def test_tts_cli_prints_json_and_returns_nonzero_on_error(tmp_path) -> None:
    output = io.StringIO()

    code = main(
        ["甲方测试", "--output", str(tmp_path / "misleading.wav")],
        node_factory=lambda: TTSNode(FakeTTSProvider()),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert code == 1
    assert payload["ok"] is False
    assert payload["module"] == "tts"
    assert payload["format"] == "mp3"
    assert payload["error_type"] == "ValueError"


def test_tts_cli_usage_error_is_json() -> None:
    output = io.StringIO()

    code = main([], stdout=output)

    payload = json.loads(output.getvalue())
    assert code == 1
    assert payload["ok"] is False
    assert payload["error_type"] == "_CLIUsageError"
