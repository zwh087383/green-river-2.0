from __future__ import annotations

import wave
from io import BytesIO

import numpy as np
import pytest

from rag_system.voice import (
    FsmnVadEngine,
    VadConfig,
    VadEvent,
    pcm16_to_wav_bytes,
)


class ScriptedVadBackend:
    def __init__(self, outputs=None) -> None:
        self.outputs = list(outputs or [])
        self.calls = []
        self.max_end_sil = 999

    def __call__(self, *, audio_in, param_dict):
        self.calls.append(
            {
                "audio": audio_in.copy(),
                "is_final": param_dict.get("is_final"),
                "param_id": id(param_dict),
                "cache_id": id(param_dict["in_cache"]),
                "max_end_sil": self.max_end_sil,
            }
        )
        return self.outputs.pop(0) if self.outputs else []


def test_vad_config_defaults_to_400_ms_end_silence() -> None:
    config = VadConfig()

    assert config.end_silence_ms == 400
    assert config.frame_ms == 100
    assert config.frame_samples == 1600
    assert config.frame_bytes == 3200
    assert config.pre_pad_ms == 300
    assert config.post_pad_ms == 200


@pytest.mark.parametrize("end_silence_ms", [99, 450, 3100])
def test_vad_config_rejects_unsupported_end_silence(end_silence_ms) -> None:
    with pytest.raises(ValueError):
        VadConfig(end_silence_ms=end_silence_ms)


def test_vad_event_payload_never_serializes_audio() -> None:
    event = VadEvent(
        type="speech_end",
        start_ms=100,
        end_ms=900,
        reason="silence",
        audio_pcm=b"secret binary audio",
    )

    assert event.to_dict() == {
        "type": "speech_end",
        "start_ms": 100,
        "end_ms": 900,
        "reason": "silence",
    }


def test_stream_reassembles_fragments_and_emits_padded_utterance() -> None:
    backend = ScriptedVadBackend(
        outputs=[[[100, -1]], [], [[-1, 200]]],
    )
    engine = FsmnVadEngine(backend=backend)
    config = VadConfig(pre_pad_ms=100, post_pad_ms=100)
    session = engine.create_session(config)
    frame = _pcm_frame(1200)
    all_audio = frame * 3

    assert session.feed_pcm16(all_audio[:1000]) == []
    first_events = session.feed_pcm16(all_audio[1000:3200])
    terminal_events = session.feed_pcm16(all_audio[3200:])

    assert [event.type for event in first_events] == ["speech_start"]
    assert first_events[0].start_ms == 100
    assert [event.type for event in terminal_events] == ["speech_end"]
    assert terminal_events[0].start_ms == 100
    assert terminal_events[0].end_ms == 200
    assert terminal_events[0].audio_pcm == all_audio
    assert len(backend.calls) == 3
    assert all(call["audio"].shape == (1600,) for call in backend.calls)
    assert all(call["audio"].dtype == np.float32 for call in backend.calls)


def test_final_full_frame_adds_nonempty_silent_flush() -> None:
    backend = ScriptedVadBackend()
    session = FsmnVadEngine(backend=backend).create_session()

    events = session.feed_pcm16(_pcm_frame(800), is_final=True)

    assert [event.type for event in events] == ["no_speech"]
    assert events[0].end_ms == 100
    assert [call["is_final"] for call in backend.calls] == [False, True]
    assert np.count_nonzero(backend.calls[-1]["audio"]) == 0
    assert backend.calls[-1]["audio"].shape == (1600,)


def test_final_partial_frame_is_zero_padded_and_preserves_real_duration() -> None:
    backend = ScriptedVadBackend()
    session = FsmnVadEngine(backend=backend).create_session()
    partial = _pcm_samples(900, 800)

    events = session.feed_pcm16(partial, is_final=True)

    assert [event.type for event in events] == ["no_speech"]
    assert events[0].end_ms == 50
    assert len(backend.calls) == 1
    assert backend.calls[0]["is_final"] is True
    assert np.count_nonzero(backend.calls[0]["audio"][:800]) == 800
    assert np.count_nonzero(backend.calls[0]["audio"][800:]) == 0


def test_manual_final_closes_active_speech_with_final_reason() -> None:
    backend = ScriptedVadBackend(outputs=[[[0, -1]], [[-1, 100]]])
    session = FsmnVadEngine(backend=backend).create_session()

    start_events = session.feed_pcm16(_pcm_frame(800))
    final_events = session.feed_pcm16(b"", is_final=True)

    assert [event.type for event in start_events] == ["speech_start"]
    assert [event.type for event in final_events] == ["speech_end"]
    assert final_events[0].reason == "final"
    assert final_events[0].audio_pcm == _pcm_frame(800)


def test_initial_silence_timeout_emits_no_speech_once() -> None:
    backend = ScriptedVadBackend()
    session = FsmnVadEngine(backend=backend).create_session(
        VadConfig(initial_silence_ms=200)
    )

    events = session.feed_pcm16(_pcm_frame(0) * 2)

    assert [event.to_dict() for event in events] == [
        {"type": "no_speech", "end_ms": 200, "reason": "initial_timeout"}
    ]
    assert session.finished is True
    assert session.feed_pcm16(_pcm_frame(0), is_final=True) == []


def test_max_utterance_emits_forced_end_with_audio() -> None:
    backend = ScriptedVadBackend(outputs=[[[0, -1]], []])
    session = FsmnVadEngine(backend=backend).create_session(
        VadConfig(max_utterance_ms=200, pre_pad_ms=0, post_pad_ms=0)
    )
    audio = _pcm_frame(1000) * 2

    events = session.feed_pcm16(audio)

    assert [event.type for event in events] == ["speech_start", "forced_end"]
    assert events[-1].to_dict() == {
        "type": "forced_end",
        "start_ms": 0,
        "end_ms": 200,
        "reason": "max_utterance",
    }
    assert events[-1].audio_pcm == audio


def test_sessions_have_independent_backend_cache_and_configuration() -> None:
    backend = ScriptedVadBackend()
    engine = FsmnVadEngine(backend=backend)
    first = engine.create_session(VadConfig(end_silence_ms=400))
    second = engine.create_session(VadConfig(end_silence_ms=800))

    first.feed_pcm16(_pcm_frame(0))
    second.feed_pcm16(_pcm_frame(0))

    assert backend.calls[0]["param_id"] != backend.calls[1]["param_id"]
    assert backend.calls[0]["cache_id"] != backend.calls[1]["cache_id"]
    assert [call["max_end_sil"] for call in backend.calls] == [400, 800]
    assert backend.max_end_sil == 999


def test_stream_rejects_odd_pcm_byte_count() -> None:
    session = FsmnVadEngine(backend=ScriptedVadBackend()).create_session()

    with pytest.raises(ValueError, match="even number of bytes"):
        session.feed_pcm16(b"\x00")


def test_pcm16_to_wav_bytes_builds_mono_16k_wav() -> None:
    pcm = _pcm_samples(400, 160)

    wav_bytes = pcm16_to_wav_bytes(pcm)

    with wave.open(BytesIO(wav_bytes), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        assert handle.readframes(handle.getnframes()) == pcm


def _pcm_frame(value: int) -> bytes:
    return _pcm_samples(value, 1600)


def _pcm_samples(value: int, count: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * count
