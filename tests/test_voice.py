from __future__ import annotations

import asyncio
import math
import sys
from types import SimpleNamespace
import wave
from io import BytesIO
from pathlib import Path

import pytest

import rag_system.voice.service as voice_service_module
from rag_system.settings import DEFAULT_STT_MODEL, SettingsStore
from rag_system.voice import VoiceDependencyError, VoiceInputError, VoiceService
from rag_system.voice.dashscope_tts import (
    DASHSCOPE_REALTIME_TTS_MODEL,
    DashScopeTTSError,
)
from rag_system.voice.models import (
    ASR_MODEL_ID,
    ASR_MODEL_NAME,
    ASR_MODEL_REVISION,
    default_asr_model_dir,
    resolve_asr_model_dir,
    verify_asr_model,
)


class FakeWhisperModel:
    def __init__(self) -> None:
        self.audio_length = 0
        self.decode_options = {}

    def transcribe(
        self,
        audio,
        *,
        language: str,
        fp16: bool,
        temperature,
        beam_size: int,
        best_of: int,
        patience: float,
    ):
        self.audio_length = len(audio)
        self.decode_options = {
            "language": language,
            "fp16": fp16,
            "temperature": temperature,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
        }
        return {"text": f"{language}:{fp16}:{self.audio_length}"}


class TraditionalChineseWhisperModel(FakeWhisperModel):
    def transcribe(self, audio, **kwargs):
        super().transcribe(audio, **kwargs)
        return {"text": "綠色江和是個什麼樣的組織"}


class FakeDashScopeTTS:
    voice = "Cherry"
    model = DASHSCOPE_REALTIME_TTS_MODEL

    def __init__(self, events=None) -> None:
        self.events = list([b"dashscope-mp3"] if events is None else events)
        self.connected = False
        self.warm_calls = []
        self.stream_calls = []
        self.close_calls = 0

    async def warm(self, api_key: str) -> None:
        self.warm_calls.append(api_key)
        self.connected = True

    def stream(self, text: str, api_key: str):
        self.stream_calls.append((text, api_key))
        self.connected = True

        async def generate():
            for event in self.events:
                if isinstance(event, BaseException):
                    self.connected = False
                    raise event
                yield event

        return generate()

    async def close(self) -> None:
        self.close_calls += 1
        self.connected = False


def test_voice_service_transcribes_wav_with_injected_whisper(tmp_path) -> None:
    fake_model = FakeWhisperModel()
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        whisper_model_loader=lambda _model: fake_model,
        tts_synthesizer=lambda _text, _voice, _rate: b"mp3",
    )

    text = service.speech_to_text(_wav_bytes(), filename="recording.wav")

    assert text.startswith("zh:False:")
    assert fake_model.audio_length > 0
    assert fake_model.decode_options == {
        "language": "zh",
        "fp16": False,
        "temperature": (0.0,),
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
    }


def test_voice_service_converts_traditional_asr_text_to_simplified(tmp_path) -> None:
    fake_model = TraditionalChineseWhisperModel()
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        whisper_model_loader=lambda _model: fake_model,
    )

    text = service.speech_to_text(_wav_bytes(), filename="recording.wav")

    assert text == "绿色江河是个什么样的组织"


def test_voice_service_uses_saved_custom_asr_correction_rules(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_asr_corrections(
        True,
        [{"wrong": "绿涩江河", "correct": "绿色江河"}],
    )
    service = VoiceService(settings_store=store)

    assert service.postprocess_asr_text("介绍一下绿涩江河") == "介绍一下绿色江河"


def test_voice_service_respects_saved_asr_correction_enabled_switch(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_asr_corrections(
        False,
        [{"wrong": "绿色江和", "correct": "绿色长河"}],
    )
    service = VoiceService(settings_store=store)

    # OpenCC remains active, while both built-in and custom correction rules
    # are bypassed by the persisted switch.
    assert service.postprocess_asr_text("綠色江和") == "绿色江和"


def test_voice_service_rejects_invalid_audio(tmp_path) -> None:
    service = VoiceService(settings_store=SettingsStore(tmp_path / "settings.json"))

    with pytest.raises(VoiceInputError, match="valid WAV"):
        service.speech_to_text(b"not-wav", filename="bad.wav")


def test_voice_service_synthesizes_with_injected_tts(tmp_path) -> None:
    calls = []
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        tts_synthesizer=lambda text, voice, rate: calls.append((text, voice, rate)) or b"fake-mp3",
    )

    audio = service.text_to_speech("  你好   世界  ")

    assert audio == b"fake-mp3"
    assert calls == [("你好 世界", "zh-CN-XiaoxiaoNeural", "+0%")]


def test_voice_service_marks_zang_in_tibetan_antelope_for_tts(tmp_path) -> None:
    calls = []
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        tts_synthesizer=lambda text, voice, rate: calls.append((text, voice, rate)) or b"fake-mp3",
    )

    service.text_to_speech("藏羚羊生活在青藏高原")

    assert calls == [("葬羚羊生活在青藏高原", "zh-CN-XiaoxiaoNeural", "+0%")]


def test_voice_service_distinguishes_quantities_from_calendar_years(tmp_path) -> None:
    calls = []
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        tts_synthesizer=lambda text, voice, rate: calls.append((text, voice, rate)) or b"fake-mp3",
    )

    service.text_to_speech("经过近30年的保护，该保护站于1940年建立，位于G109国道附近")

    assert calls == [
        (
            "经过近三十年的保护，该保护站于一九四零年建立，位于G幺零九国道附近",
            "zh-CN-XiaoxiaoNeural",
            "+0%",
        )
    ]


def test_voice_service_streams_with_injected_tts(tmp_path) -> None:
    calls = []

    async def synthesize(text: str, voice: str, rate: str):
        calls.append((text, voice, rate))
        yield b"first"
        yield b""
        yield b"second"

    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        tts_stream_synthesizer=synthesize,
    )

    async def collect() -> list[bytes]:
        return [chunk async for chunk in service.stream_text_to_speech("  你好   世界  ")]

    assert asyncio.run(collect()) == [b"first", b"second"]
    assert calls == [("你好 世界", "zh-CN-XiaoxiaoNeural", "+0%")]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("   ", "TTS text cannot be empty"),
        ("字" * 2001, "TTS text is too long; max 2000 characters"),
        ("！？……—", "TTS text must contain at least one letter or number"),
        ("[1]", "TTS text must contain at least one letter or number"),
    ],
)
def test_voice_service_streaming_rejects_invalid_text(tmp_path, text, message) -> None:
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        tts_stream_synthesizer=lambda _text, _voice, _rate: None,
        tts_ssml_enabled=True,
    )

    with pytest.raises(VoiceInputError, match=message):
        service.stream_text_to_speech(text)


def test_voice_service_streaming_reports_missing_dependency_before_iteration(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setitem(sys.modules, "edge_tts", None)
    service = VoiceService(settings_store=SettingsStore(tmp_path / "settings.json"))

    with pytest.raises(VoiceDependencyError, match="edge-tts is not installed"):
        service.stream_text_to_speech("你好")


def test_voice_service_streaming_falls_back_to_sync_injected_tts(tmp_path) -> None:
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        tts_synthesizer=lambda _text, _voice, _rate: b"complete-mp3",
    )

    async def collect() -> list[bytes]:
        return [chunk async for chunk in service.stream_text_to_speech("你好")]

    assert asyncio.run(collect()) == [b"complete-mp3"]


def test_voice_service_streams_edge_audio_events_and_closes_on_cancel(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    state = {"closed": False}

    class FakeEdgeStream:
        def __init__(self) -> None:
            self.events = iter(
                [
                    {"type": "SentenceBoundary", "text": "你好"},
                    {"type": "audio", "data": b"mp3-chunk"},
                    {"type": "audio", "data": b"another-chunk"},
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def aclose(self) -> None:
            state["closed"] = True

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, *, rate: str) -> None:
            calls.append((text, voice, rate))

        def stream(self):
            return FakeEdgeStream()

    monkeypatch.setitem(
        sys.modules,
        "edge_tts",
        SimpleNamespace(Communicate=FakeCommunicate),
    )
    service = VoiceService(settings_store=SettingsStore(tmp_path / "settings.json"))

    async def consume_first_chunk() -> bytes:
        stream = service.stream_text_to_speech("你好")
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    assert asyncio.run(consume_first_chunk()) == b"mp3-chunk"
    assert calls == [("你好", "zh-CN-XiaoxiaoNeural", "+0%")]
    assert state["closed"] is True


def test_voice_service_warms_dashscope_and_reports_ready_female_provider(tmp_path) -> None:
    api_key = "sk-dashscope-warm-12345678"
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key(api_key)
    fake_dashscope = FakeDashScopeTTS()
    service = VoiceService(settings_store=store, dashscope_tts=fake_dashscope)

    assert asyncio.run(service.warm_tts_connection()) is True
    status = service.status()

    assert fake_dashscope.warm_calls == [api_key]
    assert status.tts_available is True
    assert status.tts_provider == "dashscope_realtime"
    assert status.tts_provider_active == "dashscope_realtime"
    assert status.tts_connection_state == "ready"
    assert status.tts_voice == "Cherry"
    assert status.tts_model == DASHSCOPE_REALTIME_TTS_MODEL


def test_voice_service_prefers_dashscope_for_streaming_tts(monkeypatch, tmp_path) -> None:
    api_key = "sk-dashscope-stream-12345678"
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key(api_key)
    fake_dashscope = FakeDashScopeTTS([b"dash-one", b"", b"dash-two"])
    edge_calls = []

    async def edge_stream(text: str, *, voice: str, rate: str):
        edge_calls.append((text, voice, rate))
        yield b"unexpected-edge"

    monkeypatch.setattr(voice_service_module, "_edge_tts_stream_mp3", edge_stream)
    service = VoiceService(settings_store=store, dashscope_tts=fake_dashscope)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in service.stream_text_to_speech("你好")]

    assert asyncio.run(collect()) == [b"dash-one", b"dash-two"]
    assert fake_dashscope.stream_calls == [("你好", api_key)]
    assert edge_calls == []


def test_voice_service_falls_back_to_edge_before_first_dashscope_chunk(
    monkeypatch,
    tmp_path,
) -> None:
    api_key = "sk-dashscope-fallback-12345678"
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key(api_key)
    fake_dashscope = FakeDashScopeTTS([DashScopeTTSError("DashScope unavailable")])
    edge_calls = []

    async def edge_stream(text: str, *, voice: str, rate: str):
        edge_calls.append((text, voice, rate))
        yield b"edge-fallback"

    monkeypatch.setattr(voice_service_module, "_edge_tts_stream_mp3", edge_stream)
    service = VoiceService(settings_store=store, dashscope_tts=fake_dashscope)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in service.stream_text_to_speech("你好")]

    assert asyncio.run(collect()) == [b"edge-fallback"]
    assert fake_dashscope.stream_calls == [("你好", api_key)]
    assert edge_calls == [("你好", "zh-CN-XiaoxiaoNeural", "+0%")]
    assert service.status().tts_provider_error == "DashScope unavailable"


def test_voice_service_does_not_mix_edge_after_dashscope_emits_audio(
    monkeypatch,
    tmp_path,
) -> None:
    api_key = "sk-dashscope-partial-12345678"
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key(api_key)
    fake_dashscope = FakeDashScopeTTS(
        [b"dash-partial", DashScopeTTSError("DashScope dropped")]
    )
    edge_calls = []

    async def edge_stream(text: str, *, voice: str, rate: str):
        edge_calls.append((text, voice, rate))
        yield b"must-not-be-mixed"

    monkeypatch.setattr(voice_service_module, "_edge_tts_stream_mp3", edge_stream)
    service = VoiceService(settings_store=store, dashscope_tts=fake_dashscope)

    async def consume() -> list[bytes]:
        chunks = []
        with pytest.raises(DashScopeTTSError, match="DashScope dropped"):
            async for chunk in service.stream_text_to_speech("你好"):
                chunks.append(chunk)
        return chunks

    assert asyncio.run(consume()) == [b"dash-partial"]
    assert edge_calls == []


def test_voice_service_closes_dashscope_connection(tmp_path) -> None:
    fake_dashscope = FakeDashScopeTTS()
    fake_dashscope.connected = True
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        dashscope_tts=fake_dashscope,
    )

    asyncio.run(service.close_tts_connection())

    assert fake_dashscope.close_calls == 1
    assert fake_dashscope.connected is False


def test_voice_service_opens_tracks_and_releases_realtime_asr(tmp_path) -> None:
    api_key = "sk-dashscope-asr-12345678"
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key(api_key)
    created = []

    class FakeASRTransport:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.connect_calls = []
            self.close_calls = 0

        async def connect(self, candidate: str) -> None:
            self.connect_calls.append(candidate)

        async def close(self) -> None:
            self.close_calls += 1

    def factory(**kwargs):
        transport = FakeASRTransport(**kwargs)
        created.append(transport)
        return transport

    service = VoiceService(
        settings_store=store,
        dashscope_asr_factory=factory,
    )

    async def exercise():
        transport = await service.open_realtime_asr(end_silence_ms=400)
        assert transport is not None
        await service.release_realtime_asr(transport)
        return transport

    transport = asyncio.run(exercise())

    assert created == [transport]
    assert transport.kwargs == {
        "model": "qwen3-asr-flash-realtime",
        "silence_duration_ms": 400,
    }
    assert transport.connect_calls == [api_key]
    assert transport.close_calls == 1
    status = service.status()
    assert status.asr_provider == "dashscope_realtime"
    assert status.asr_provider_active == "dashscope_realtime"
    assert status.asr_model == "qwen3-asr-flash-realtime"
    assert status.asr_streaming is True


def test_voice_service_can_force_qwen_for_an_explicit_module_test(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key("sk-dashscope-asr-force-12345678")
    store.save_asr("local_whisper")
    created = []

    class FakeASRTransport:
        async def connect(self, _candidate: str) -> None:
            return None

        async def close(self) -> None:
            return None

    def factory(**_kwargs):
        transport = FakeASRTransport()
        created.append(transport)
        return transport

    service = VoiceService(settings_store=store, dashscope_asr_factory=factory)

    async def exercise():
        assert await service.open_realtime_asr() is None
        transport = await service.open_realtime_asr(force=True)
        assert transport is created[0]
        await service.release_realtime_asr(transport)

    asyncio.run(exercise())


def test_voice_service_uses_local_asr_after_cloud_connect_failure(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key("sk-dashscope-asr-fail-12345678")

    class FailingASRTransport:
        async def connect(self, _api_key: str) -> None:
            raise RuntimeError("cloud ASR unavailable")

        async def close(self) -> None:
            return None

    service = VoiceService(
        settings_store=store,
        whisper_model_loader=lambda _model: FakeWhisperModel(),
        dashscope_asr_factory=lambda **_kwargs: FailingASRTransport(),
    )

    assert asyncio.run(service.open_realtime_asr()) is None
    status = service.status()

    assert status.stt_available is True
    assert status.asr_provider == "dashscope_realtime"
    assert status.asr_provider_active == "local_whisper"
    assert status.asr_streaming is False
    assert status.asr_provider_error == "cloud ASR unavailable"


def test_voice_service_status_uses_injected_providers(tmp_path) -> None:
    service = VoiceService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        whisper_model_loader=lambda _model: FakeWhisperModel(),
        tts_synthesizer=lambda _text, _voice, _rate: b"mp3",
    )

    status = service.status()

    assert status.voice_available is True
    assert status.stt_available is True
    assert status.tts_available is True
    assert status.stt_model == DEFAULT_STT_MODEL
    assert status.tts_voice == "zh-CN-XiaoxiaoNeural"


def test_default_asr_model_dir_can_be_overridden(monkeypatch, tmp_path) -> None:
    target = tmp_path / "models" / "asr"
    monkeypatch.setenv("RAG_SYSTEM_ASR_MODEL_DIR", str(target))

    assert default_asr_model_dir() == target.resolve()


def test_default_local_asr_model_is_medium() -> None:
    assert DEFAULT_STT_MODEL == "medium"
    assert ASR_MODEL_NAME == "medium"
    assert ASR_MODEL_ID == "Systran/faster-whisper-medium"
    assert default_asr_model_dir().name == "faster-whisper-medium"
    assert resolve_asr_model_dir("medium") == default_asr_model_dir()
    assert resolve_asr_model_dir("base").name == "faster-whisper-base"


def test_verify_asr_model_reports_missing_files(tmp_path) -> None:
    result = verify_asr_model(tmp_path / "missing")

    assert result.ready is False
    assert result.model_id == ASR_MODEL_ID
    assert result.revision == ASR_MODEL_REVISION
    assert result.size_bytes == 0
    assert "missing file: model.bin" in result.errors


def _wav_bytes(sample_rate: int = 16000, duration_seconds: float = 0.05) -> bytes:
    frames = bytearray()
    for index in range(int(sample_rate * duration_seconds)):
        sample = int(math.sin(index / 8) * 12000)
        frames.extend(sample.to_bytes(2, "little", signed=True))
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()
