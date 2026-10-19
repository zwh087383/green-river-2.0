from __future__ import annotations

import json
import math
import wave
from io import BytesIO

import pytest

from rag_system.contracts import (
    AudioBuffer,
    AudioInputError,
    ModuleResult,
    iter_pcm_chunks,
    load_wav_pcm16,
)


def _wav_bytes(*, sample_rate: int = 8_000, channels: int = 2, duration_ms: int = 100) -> bytes:
    frame_count = sample_rate * duration_ms // 1_000
    samples = bytearray()
    for index in range(frame_count):
        value = round(2_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for _channel in range(channels):
            samples.extend(int(value).to_bytes(2, "little", signed=True))
    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))
    return output.getvalue()


def test_load_wav_normalizes_stereo_and_sample_rate() -> None:
    audio = load_wav_pcm16(_wav_bytes())

    assert audio.spec.sample_rate == 16_000
    assert audio.spec.channels == 1
    assert audio.duration_ms == 100
    assert len(audio.pcm16) == 3_200


def test_audio_chunks_have_customer_friendly_fixed_size() -> None:
    audio = AudioBuffer(b"\x00\x00" * 3_200)

    chunks = list(iter_pcm_chunks(audio, chunk_ms=100))

    assert [len(chunk) for chunk in chunks] == [3_200, 3_200]


def test_load_wav_rejects_invalid_input() -> None:
    with pytest.raises(AudioInputError, match="valid WAV"):
        load_wav_pcm16(b"not-a-wav")


def test_module_result_is_plain_json_data() -> None:
    result = ModuleResult.success("asr", {"text": "绿色江河"}, metrics={"latency_ms": 12})

    payload = result.to_dict()

    assert json.loads(json.dumps(payload, ensure_ascii=False))["output"]["text"] == "绿色江河"


def test_module_failure_keeps_a_flat_error_contract() -> None:
    payload = ModuleResult.failure("vad", ValueError("bad audio")).to_dict()

    assert payload["ok"] is False
    assert payload["error"] == "bad audio"
    assert payload["error_type"] == "ValueError"
