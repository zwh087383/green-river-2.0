from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "ros2" / "voice"


class _WakeSession:
    def __init__(self) -> None:
        self.closed = False

    def accept_pcm16(self, pcm16: bytes) -> object | None:
        if pcm16 == b"wake":
            return SimpleNamespace(keyword="wake")
        return None

    def close(self) -> None:
        self.closed = True


class _WakeEngine:
    def create_session(self) -> _WakeSession:
        return _WakeSession()


class _VadSession:
    def __init__(self) -> None:
        self.closed = False

    def feed_pcm16(self, pcm16: bytes, *, is_final: bool) -> list[object]:
        assert not is_final
        if pcm16 == b"speech":
            return [SimpleNamespace(type="speech_start")]
        if pcm16 == b"done":
            return [SimpleNamespace(type="speech_end", audio_pcm=b"\x01\x00")]
        if pcm16 == b"timeout":
            return [SimpleNamespace(type="no_speech", reason="initial_timeout")]
        return []

    def close(self) -> None:
        self.closed = True


class _VadEngine:
    def create_session(self, _config: object) -> _VadSession:
        return _VadSession()


def test_audio_pipeline_generation_advances_across_new_incarnations() -> None:
    sys.path.insert(0, str(VOICE))
    try:
        import voice.audio_pipeline as audio_pipeline

        with patch.object(audio_pipeline.time, "time_ns", side_effect=[1_000, 2_000]):
            first = audio_pipeline.LocalAudioPipeline(
                wake_engine=_WakeEngine(),
                vad_engine=_VadEngine(),
                vad_config=object(),
            )
            second = audio_pipeline.LocalAudioPipeline(
                wake_engine=_WakeEngine(),
                vad_engine=_VadEngine(),
                vad_config=object(),
            )

        assert second.generation > first.generation
    finally:
        sys.path.remove(str(VOICE))


def test_audio_pipeline_wakes_segments_and_resets() -> None:
    sys.path.insert(0, str(VOICE))
    try:
        from voice.audio_pipeline import LocalAudioPipeline

        pipeline = LocalAudioPipeline(
            wake_engine=_WakeEngine(),
            vad_engine=_VadEngine(),
            vad_config=object(),
        )
        assert pipeline.state == "armed"
        assert pipeline.feed(b"noise") == ()
        assert pipeline.feed(b"wake")[0].type == "wake"
        assert pipeline.state == "listening"
        assert pipeline.feed(b"wake") == ()
        assert pipeline.feed(b"speech")[0].type == "speech_start"
        utterance = pipeline.feed(b"done")[0]
        assert utterance.type == "utterance"
        assert utterance.pcm16 == b"\x01\x00"
        assert pipeline.state == "transcribing"

        pipeline.reset()
        assert pipeline.state == "armed"
        pipeline.close()
        assert pipeline.state == "closed"
    finally:
        sys.path.remove(str(VOICE))


def test_audio_pipeline_can_run_without_wakeup() -> None:
    sys.path.insert(0, str(VOICE))
    try:
        from voice.audio_pipeline import LocalAudioPipeline

        pipeline = LocalAudioPipeline(
            wake_engine=None,
            vad_engine=_VadEngine(),
            vad_config=object(),
        )
        assert pipeline.state == "listening"
        timeout = pipeline.feed(b"timeout")[0]
        assert timeout.type == "timeout"
        assert pipeline.state == "listening"
    finally:
        sys.path.remove(str(VOICE))


def test_audio_pipeline_resumes_listening_after_ignored_turn() -> None:
    sys.path.insert(0, str(VOICE))
    try:
        from voice.audio_pipeline import LocalAudioPipeline

        pipeline = LocalAudioPipeline(
            wake_engine=_WakeEngine(),
            vad_engine=_VadEngine(),
            vad_config=object(),
        )
        pipeline.feed(b"wake")
        pipeline.feed(b"done")
        assert pipeline.state == "transcribing"

        pipeline.resume_listening()

        assert pipeline.state == "listening"
        assert pipeline.feed(b"speech")[0].type == "speech_start"
    finally:
        sys.path.remove(str(VOICE))


def test_audio_pipeline_does_not_run_kws_while_question_vad_is_listening() -> None:
    class CountingWakeSession(_WakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.accept_calls = 0

        def accept_pcm16(self, pcm16: bytes) -> object | None:
            self.accept_calls += 1
            return super().accept_pcm16(pcm16)

    class CountingWakeEngine:
        def __init__(self) -> None:
            self.session = CountingWakeSession()

        def create_session(self) -> CountingWakeSession:
            return self.session

    sys.path.insert(0, str(VOICE))
    try:
        from voice.audio_pipeline import LocalAudioPipeline

        wake_engine = CountingWakeEngine()
        pipeline = LocalAudioPipeline(
            wake_engine=wake_engine,
            vad_engine=_VadEngine(),
            vad_config=object(),
        )
        assert pipeline.feed(b"wake")[0].type == "wake"
        assert pipeline.state == "listening"
        assert wake_engine.session.accept_calls == 1

        # The same PCM is routed only to VAD after the user-speaking phase has
        # started, so it cannot create a competing wake event.
        assert pipeline.feed(b"wake") == ()
        assert wake_engine.session.accept_calls == 1
    finally:
        sys.path.remove(str(VOICE))


def test_audio_pipeline_keeps_kws_hot_and_interrupts_busy_turn() -> None:
    sys.path.insert(0, str(VOICE))
    try:
        from voice.audio_pipeline import LocalAudioPipeline

        pipeline = LocalAudioPipeline(
            wake_engine=_WakeEngine(),
            vad_engine=_VadEngine(),
            vad_config=object(),
        )
        pipeline.feed(b"wake")
        pipeline.feed(b"done")
        old_generation = pipeline.generation
        assert pipeline.begin_transcribing(old_generation)

        # VAD is closed while ASR is active, but KWS still sees this block.
        assert pipeline.feed(b"speech") == ()
        interrupt = pipeline.feed(b"wake")[0]
        assert interrupt.type == "interrupt"
        assert interrupt.purpose == "interrupt"
        assert interrupt.generation > old_generation
        assert pipeline.state == "interrupt_ack"

        assert pipeline.playback_started(
            purpose="interrupt_ack", generation=interrupt.generation
        )
        assert pipeline.state == "interrupt_ack"
        assert pipeline.playback_finished(
            purpose="interrupt_ack", generation=interrupt.generation
        )
        assert pipeline.state == "listening"
        assert pipeline.feed(b"speech")[0].type == "speech_start"
    finally:
        sys.path.remove(str(VOICE))


def test_audio_pipeline_interrupts_answering_and_playing_states() -> None:
    sys.path.insert(0, str(VOICE))
    try:
        from voice.audio_pipeline import LocalAudioPipeline

        for target_state in ("answering", "playing"):
            pipeline = LocalAudioPipeline(
                wake_engine=_WakeEngine(),
                vad_engine=_VadEngine(),
                vad_config=object(),
            )
            pipeline.feed(b"wake")
            pipeline.feed(b"done")
            assert pipeline.begin_answering(pipeline.generation)
            if target_state == "playing":
                assert pipeline.playback_started(
                    purpose="answer", generation=pipeline.generation
                )
            assert pipeline.state == target_state
            assert pipeline.feed(b"speech") == ()
            assert pipeline.feed(b"wake")[0].type == "interrupt"
            pipeline.close()
    finally:
        sys.path.remove(str(VOICE))


def test_audio_pipeline_does_not_feed_vad_during_interrupt_ack() -> None:
    class CountingVadSession(_VadSession):
        def __init__(self) -> None:
            super().__init__()
            self.feed_calls: list[bytes] = []

        def feed_pcm16(self, pcm16: bytes, *, is_final: bool) -> list[object]:
            self.feed_calls.append(pcm16)
            return super().feed_pcm16(pcm16, is_final=is_final)

    class CountingVadEngine:
        def __init__(self) -> None:
            self.sessions: list[CountingVadSession] = []

        def create_session(self, _config: object) -> CountingVadSession:
            session = CountingVadSession()
            self.sessions.append(session)
            return session

    sys.path.insert(0, str(VOICE))
    try:
        from voice.audio_pipeline import LocalAudioPipeline

        vad_engine = CountingVadEngine()
        pipeline = LocalAudioPipeline(
            wake_engine=_WakeEngine(),
            vad_engine=vad_engine,
            vad_config=object(),
        )
        pipeline.feed(b"wake")
        pipeline.feed(b"done")
        assert vad_engine.sessions[0].feed_calls == [b"done"]

        interrupt = pipeline.feed(b"wake")[0]
        assert interrupt.type == "interrupt"
        assert pipeline.state == "interrupt_ack"
        assert pipeline.feed(b"speech") == ()

        assert len(vad_engine.sessions) == 1
        assert vad_engine.sessions[0].feed_calls == [b"done"]
    finally:
        sys.path.remove(str(VOICE))
