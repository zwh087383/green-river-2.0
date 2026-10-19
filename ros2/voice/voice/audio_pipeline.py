from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AudioPipelineEvent:
    """An event emitted by :class:`LocalAudioPipeline`.

    ``generation`` is copied by the audio node into its asynchronous work
    items.  A result from an older generation is never allowed to change the
    current microphone state after a wake-word interruption.
    """

    type: str
    keyword: str = ""
    pcm16: bytes = b""
    detail: str = ""
    generation: int = 0
    purpose: str = ""

    @property
    def epoch(self) -> int:
        """Alias used by cancellation-oriented callers."""

        return self.generation


class LocalAudioPipeline:
    """Local KWS/VAD state machine for one robot microphone.

    Keyword spotting stays active outside the user-speaking ``listening``
    phase.  FSMN VAD is fed only while listening, so the two recognizers never
    compete for the same question audio.
    """

    _BUSY_STATES = frozenset(
        {"processing", "transcribing", "answering", "playing", "interrupt_ack"}
    )

    def __init__(
        self,
        *,
        wake_engine: Any | None,
        vad_engine: Any,
        vad_config: Any,
    ) -> None:
        self._wake_engine = wake_engine
        self._vad_engine = vad_engine
        self._vad_config = vad_config
        self._wake_session: Any | None = None
        self._vad_session: Any | None = None
        # Use a wall-clock incarnation seed instead of restarting at zero.
        # The answer and bridge nodes may survive an audio-node restart and
        # correctly reject genuinely old generations; a fresh process must
        # therefore start above values emitted by its previous incarnation.
        self._generation = _new_generation_seed()
        self.state = "initializing"
        self.reset()

    @property
    def wakeup_enabled(self) -> bool:
        return self._wake_engine is not None

    @property
    def generation(self) -> int:
        """Current interaction generation (also exposed as ``epoch``)."""

        return self._generation

    @property
    def epoch(self) -> int:
        # ``epoch`` is a useful name for callers that use cancellation tokens.
        return self._generation

    def feed(self, pcm16: bytes) -> tuple[AudioPipelineEvent, ...]:
        """Consume one PCM16 block and return state-machine events.

        KWS is evaluated before VAD outside ``listening``.  In a busy state a
        hit emits exactly one ``interrupt`` event and no VAD event for that block.
        """

        if not pcm16 or self.state == "closed":
            return ()

        # Keep KWS hot while transcribing, answering, and playing, but never
        # run it alongside question VAD.
        keyword = "" if self.state == "listening" else self._accept_wake(pcm16)
        if keyword:
            self._rearm_wake()
            if self.state == "interrupt_ack":
                return ()
            if self.state in self._BUSY_STATES:
                self._generation += 1
                self._close_session("_vad_session")
                self.state = "interrupt_ack"
                return (
                    AudioPipelineEvent(
                        type="interrupt",
                        keyword=keyword,
                        detail="wake word during active turn",
                        generation=self._generation,
                        purpose="interrupt",
                    ),
                )
            if self.state == "armed":
                self._start_vad()
                self.state = "listening"
                return (
                    AudioPipelineEvent(
                        type="wake",
                        keyword=keyword,
                        generation=self._generation,
                        purpose="wake",
                    ),
                )
            return ()

        # VAD is intentionally restricted to listening.  In particular, do
        # not feed it during ASR or TTS where microphone/speaker bleed is
        # expected.
        if self.state != "listening":
            return ()
        if self._vad_session is None:
            self._start_vad()
        assert self._vad_session is not None

        output: list[AudioPipelineEvent] = []
        for vad_event in self._vad_session.feed_pcm16(pcm16, is_final=False):
            event_type = _event_value(vad_event, "type")
            if event_type == "speech_start":
                output.append(
                    AudioPipelineEvent(
                        type="speech_start", generation=self._generation
                    )
                )
            elif event_type in {"speech_end", "forced_end"}:
                utterance = _event_bytes(vad_event, "audio_pcm")
                self._close_session("_vad_session")
                if utterance:
                    self.state = "transcribing"
                    output.append(
                        AudioPipelineEvent(
                            type="utterance",
                            pcm16=utterance,
                            detail=event_type,
                            generation=self._generation,
                            purpose="query",
                        )
                    )
                else:
                    self.reset()
                    output.append(
                        AudioPipelineEvent(
                            type="error",
                            detail="empty utterance",
                            generation=self._generation,
                        )
                    )
            elif event_type == "no_speech":
                detail = _event_value(vad_event, "reason")
                self.reset()
                output.append(
                    AudioPipelineEvent(
                        type="timeout",
                        detail=detail,
                        generation=self._generation,
                    )
                )
        return tuple(output)

    def reset(self) -> None:
        """Start a fresh interaction and invalidate all prior work."""

        self._generation += 1
        self._close_session("_wake_session")
        self._close_session("_vad_session")
        if self._wake_engine is not None:
            self._wake_session = self._create_wake_session()
            self.state = "armed"
        else:
            self._start_vad()
            self.state = "listening"

    def resume_listening(self) -> None:
        """Resume VAD after an ignored turn or a completed interruption ack."""

        if self.state == "closed":
            return
        self._generation += 1
        self._close_session("_vad_session")
        self._start_vad()
        self.state = "listening"
        self._rearm_wake()

    def begin_transcribing(self, generation: int | None = None) -> bool:
        return self._set_busy_state("transcribing", generation)

    def begin_answering(self, generation: int | None = None) -> bool:
        return self._set_busy_state("answering", generation)

    def playback_started(
        self, *, purpose: str = "answer", generation: int | None = None
    ) -> bool:
        """Acknowledge that an output adapter accepted audio for playback."""

        if generation is not None and int(generation) != self._generation:
            return False
        if self.state not in self._BUSY_STATES:
            return False
        self._close_session("_vad_session")
        if purpose in {"interrupt", "interrupt_ack", "wakeup_ack", "ack"}:
            self.state = "interrupt_ack"
        else:
            self.state = "playing"
        return True

    # Short aliases keep the state machine convenient for injected adapters
    # and preserve the terminology used by older callers.
    begin_playing = playback_started
    ack_playback = playback_started

    def playback_finished(
        self,
        *,
        purpose: str = "answer",
        generation: int | None = None,
        failed: bool = False,
    ) -> bool:
        """Handle a completion/failure acknowledgement from the adapter."""

        if generation is not None and int(generation) != self._generation:
            return False
        if purpose in {"interrupt", "interrupt_ack", "wakeup_ack", "ack"}:
            self.resume_listening()
        else:
            self.reset()
        return True

    def interrupt(self, keyword: str = "", detail: str = "") -> AudioPipelineEvent:
        """Cancel the current turn and enter the playback-ack wait state."""

        self._generation += 1
        self._close_session("_vad_session")
        self._rearm_wake()
        self.state = "interrupt_ack"
        return AudioPipelineEvent(
            type="interrupt",
            keyword=keyword,
            detail=detail,
            generation=self._generation,
            purpose="interrupt",
        )

    def close(self) -> None:
        self._close_session("_wake_session")
        self._close_session("_vad_session")
        self.state = "closed"

    def _set_busy_state(self, state: str, generation: int | None) -> bool:
        if self.state == "closed":
            return False
        if generation is not None and int(generation) != self._generation:
            return False
        self._close_session("_vad_session")
        self.state = state
        return True

    def _start_vad(self) -> None:
        self._vad_session = self._vad_engine.create_session(self._vad_config)

    def _create_wake_session(self) -> Any:
        create_session = getattr(self._wake_engine, "create_session")
        return create_session()

    def _accept_wake(self, pcm16: bytes) -> str:
        if self._wake_engine is None or self._wake_session is None:
            return ""
        accept = getattr(self._wake_session, "accept_pcm16", None)
        if not callable(accept):
            accept = getattr(self._wake_session, "feed_pcm16", None)
        if not callable(accept):
            return ""
        return _event_value(accept(pcm16), "keyword")

    def _rearm_wake(self) -> None:
        if self._wake_engine is None:
            return
        session = self._wake_session
        if session is None:
            self._wake_session = self._create_wake_session()
            return
        reset = getattr(session, "reset", None)
        if callable(reset):
            try:
                reset()
                return
            except Exception:
                # A backend may not support resetting a detected stream.  A
                # fresh session is equivalent and keeps KWS always-on.
                pass
        self._close_session("_wake_session")
        self._wake_session = self._create_wake_session()

    def _close_session(self, name: str) -> None:
        session = getattr(self, name)
        if session is None:
            return
        close = getattr(session, "close", None)
        if callable(close):
            close()
        setattr(self, name, None)


def _event_value(event: Any, name: str) -> str:
    if event is None:
        return ""
    if isinstance(event, bool):
        return "" if not event else ("wake" if name == "keyword" else "true")
    if isinstance(event, str):
        return event.strip() if name == "keyword" else ""
    value = event.get(name) if isinstance(event, dict) else getattr(event, name, None)
    return str(value or "").strip()


def _event_bytes(event: Any, name: str) -> bytes:
    if event is None:
        return b""
    value = event.get(name) if isinstance(event, dict) else getattr(event, name, None)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return b""


def _new_generation_seed() -> int:
    """Return a non-zero uint64 seed that advances across process restarts."""

    return max(1, min((1 << 64) - 2, time.time_ns()))


__all__ = ["AudioPipelineEvent", "LocalAudioPipeline"]
