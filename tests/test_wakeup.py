from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from rag_system.voice.wakeup import (
    DEFAULT_KEYWORD_SPEC,
    DEFAULT_KEYWORDS_FILENAME,
    WakeWordEngine,
    WakeWordModelError,
    _keyword_from_result,
    _resolve_keywords_file,
    keyword_spec_for_text,
)
from rag_system.voice.wakeup_models import WAKEUP_TOKENS_FILE


class FakeStream:
    def __init__(self) -> None:
        self.accepted = []

    def accept_waveform(self, sample_rate, samples) -> None:
        self.accepted.append((sample_rate, samples.copy()))


class ScriptedKwsBackend:
    def __init__(self, results=None) -> None:
        self.results = list(results or [])
        self.streams = []
        self.decode_calls = 0
        self.reset_calls = 0

    def create_stream(self):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream) -> bool:
        return bool(self.results)

    def decode_stream(self, stream) -> None:
        self.decode_calls += 1

    def get_result(self, stream):
        return self.results.pop(0)

    def reset_stream(self, stream) -> None:
        self.reset_calls += 1


def test_default_wake_word_detects_once_until_session_reset() -> None:
    backend = ScriptedKwsBackend(["小绿小绿"])
    session = WakeWordEngine(backend=backend).create_session()
    pcm = _pcm_samples([-32_768, 0, 32_767])

    event = session.accept_pcm16(pcm)

    assert event is not None
    assert event.keyword == "小绿小绿"
    assert event.to_dict() == {"type": "wakeup", "keyword": "小绿小绿"}
    assert session.detected is True
    assert backend.reset_calls == 1
    sample_rate, samples = backend.streams[0].accepted[0]
    assert sample_rate == 16_000
    assert samples.dtype == np.float32
    assert samples.tolist() == pytest.approx([-1.0, 0.0, 32_767 / 32_768])

    assert session.accept_pcm16(pcm) is None
    assert len(backend.streams[0].accepted) == 1

    session.reset()
    backend.results.append("小绿小绿")
    assert session.accept_pcm16(pcm).keyword == "小绿小绿"
    assert backend.reset_calls == 2


def test_sessions_keep_independent_streaming_state() -> None:
    backend = ScriptedKwsBackend()
    engine = WakeWordEngine(backend=backend)
    first = engine.create_session()
    second = engine.create_session()

    first.accept_pcm16(_pcm_samples([100]))
    second.accept_pcm16(_pcm_samples([200]))

    assert backend.streams[0] is not backend.streams[1]
    assert backend.streams[0].accepted[0][1].tolist() == pytest.approx([100 / 32_768])
    assert backend.streams[1].accepted[0][1].tolist() == pytest.approx([200 / 32_768])


def test_session_validates_pcm_and_close_is_idempotent() -> None:
    session = WakeWordEngine(backend=ScriptedKwsBackend()).create_session()

    with pytest.raises(TypeError, match="bytes-like"):
        session.accept_pcm16("not bytes")
    with pytest.raises(ValueError, match="even number of bytes"):
        session.accept_pcm16(b"\x00")
    assert session.accept_pcm16(b"") is None

    session.close()
    session.close()
    assert session.closed is True
    with pytest.raises(RuntimeError, match="session is closed"):
        session.accept_pcm16(b"")


def test_engine_close_prevents_new_or_existing_session_use() -> None:
    backend = ScriptedKwsBackend()
    engine = WakeWordEngine(backend=backend)
    session = engine.create_session()

    engine.close()

    with pytest.raises(RuntimeError, match="engine is closed"):
        engine.create_session()
    with pytest.raises(RuntimeError, match="engine is closed"):
        session.accept_pcm16(_pcm_samples([1]))


def test_retired_engine_keeps_existing_session_alive_until_it_closes() -> None:
    backend = ScriptedKwsBackend(["小绿小绿"])
    engine = WakeWordEngine(backend=backend)
    session = engine.create_session()

    engine.retire()

    with pytest.raises(RuntimeError, match="engine is retired"):
        engine.create_session()
    assert session.accept_pcm16(_pcm_samples([1])).keyword == "小绿小绿"
    assert engine._closed is False

    session.close()

    assert engine._closed is True


def test_default_keyword_file_is_utf8_and_uses_ppinyin_tokens(tmp_path) -> None:
    tokens = ["x", "iǎo", "l", "ǜ"]
    (tmp_path / WAKEUP_TOKENS_FILE).write_text(
        "\n".join(f"{token} {index}" for index, token in enumerate(tokens)),
        encoding="utf-8",
    )

    path = _resolve_keywords_file(
        tmp_path,
        keywords_file=None,
        keyword_specs=None,
    )

    assert path == tmp_path / DEFAULT_KEYWORDS_FILENAME
    assert path.read_text(encoding="utf-8") == f"{DEFAULT_KEYWORD_SPEC}\n"


def test_keyword_spec_validation_reports_unknown_token(tmp_path) -> None:
    (tmp_path / WAKEUP_TOKENS_FILE).write_text("x 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from tokens.txt"):
        _resolve_keywords_file(
            tmp_path,
            keywords_file=None,
            keyword_specs=("x unknown @测试",),
        )


def test_keyword_file_rejects_an_explicit_empty_spec_list(tmp_path) -> None:
    (tmp_path / WAKEUP_TOKENS_FILE).write_text("x 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one keyword"):
        _resolve_keywords_file(
            tmp_path,
            keywords_file=None,
            keyword_specs=(),
        )


def test_keyword_spec_for_text_converts_custom_chinese_word(tmp_path) -> None:
    tokens = ["n", "ǐ", "h", "ǎo", "x", "iǎo", "l", "ǜ"]
    tokens_file = tmp_path / WAKEUP_TOKENS_FILE
    tokens_file.write_text(
        "\n".join(f"{token} {index}" for index, token in enumerate(tokens)),
        encoding="utf-8",
    )

    assert keyword_spec_for_text(" 你好小绿 ", tokens_file) == (
        "n ǐ h ǎo x iǎo l ǜ @你好小绿"
    )


def test_keyword_spec_for_text_rejects_non_chinese_word(tmp_path) -> None:
    tokens_file = tmp_path / WAKEUP_TOKENS_FILE
    tokens_file.write_text("x 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Chinese wake words only"):
        keyword_spec_for_text("hello", tokens_file)


def test_missing_model_has_actionable_error(tmp_path) -> None:
    with pytest.raises(WakeWordModelError) as caught:
        WakeWordEngine(tmp_path / "missing")

    message = str(caught.value)
    assert "local wake-up model is not ready" in message
    assert "download_wakeup_model()" in message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("小绿小绿", "小绿小绿"),
        ('{"keyword": "小绿小绿"}', "小绿小绿"),
        ("", ""),
    ],
)
def test_keyword_result_compatibility(raw, expected) -> None:
    assert _keyword_from_result(raw) == expected


@dataclass
class StructuredResult:
    keyword: str


def test_keyword_result_accepts_structured_binding_result() -> None:
    assert _keyword_from_result(StructuredResult("小绿小绿")) == "小绿小绿"


def _pcm_samples(values: list[int]) -> bytes:
    return b"".join(value.to_bytes(2, "little", signed=True) for value in values)
