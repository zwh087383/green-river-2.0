from __future__ import annotations

import math
import os
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Literal


VadEventType = Literal["speech_start", "speech_end", "no_speech", "forced_end"]


class VadDependencyError(RuntimeError):
    """Raised when the optional offline VAD runtime cannot be loaded."""


@dataclass(frozen=True, slots=True)
class VadConfig:
    end_silence_ms: int = 400
    frame_ms: int = 100
    sample_rate: int = 16_000
    initial_silence_ms: int = 10_000
    max_utterance_ms: int = 30_000
    pre_pad_ms: int = 300
    post_pad_ms: int = 200

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000:
            raise ValueError("FSMN VAD requires a 16000 Hz sample rate")
        if self.frame_ms != 100:
            raise ValueError("FSMN VAD requires 100 ms frames")
        if not 100 <= self.end_silence_ms <= 3_000:
            raise ValueError("end_silence_ms must be between 100 and 3000")
        if self.end_silence_ms % self.frame_ms:
            raise ValueError("end_silence_ms must be a multiple of frame_ms")
        if self.initial_silence_ms <= 0:
            raise ValueError("initial_silence_ms must be positive")
        if self.max_utterance_ms <= 0:
            raise ValueError("max_utterance_ms must be positive")
        if self.pre_pad_ms < 0 or self.post_pad_ms < 0:
            raise ValueError("pre_pad_ms and post_pad_ms cannot be negative")

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1_000

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2


@dataclass(frozen=True, slots=True)
class VadEvent:
    type: VadEventType
    start_ms: int | None = None
    end_ms: int | None = None
    reason: str | None = None
    audio_pcm: bytes | None = None

    def to_dict(self) -> dict[str, str | int]:
        """Return the websocket-safe event payload without the binary audio."""

        payload: dict[str, str | int] = {"type": self.type}
        if self.start_ms is not None:
            payload["start_ms"] = self.start_ms
        if self.end_ms is not None:
            payload["end_ms"] = self.end_ms
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


class FsmnVadEngine:
    """A process-shareable FSMN VAD model with per-session streaming state."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        end_silence_ms: int = 400,
        intra_op_num_threads: int | None = None,
        backend: Any | None = None,
    ) -> None:
        # Validate the value before it is passed to the third-party runtime.
        VadConfig(end_silence_ms=end_silence_ms)
        self.end_silence_ms = end_silence_ms
        self.intra_op_num_threads = max(
            1,
            min(4, int(intra_op_num_threads or os.cpu_count() or 4)),
        )
        self._inference_lock = Lock()

        if backend is not None:
            self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
            self._backend = backend
            return

        if model_dir is None:
            try:
                from rag_system.voice.vad_models import default_vad_model_dir
            except ImportError as exc:
                raise VadDependencyError("VAD model management is unavailable") from exc
            model_dir = default_vad_model_dir()

        self.model_dir = Path(model_dir).expanduser().resolve()
        try:
            from funasr_onnx import Fsmn_vad_online
        except ImportError as exc:
            raise VadDependencyError(
                'funasr-onnx is not installed; install with the "vad" or "voice" extra'
            ) from exc

        try:
            self._backend = Fsmn_vad_online(
                self.model_dir,
                quantize=True,
                device_id=-1,
                intra_op_num_threads=self.intra_op_num_threads,
                max_end_sil=end_silence_ms,
            )
        except Exception as exc:
            raise VadDependencyError(
                f"failed to load the local FSMN VAD model at {self.model_dir}: {exc}"
            ) from exc

    def create_session(self, config: VadConfig | None = None) -> StreamingVadSession:
        return StreamingVadSession(self, config or VadConfig(end_silence_ms=self.end_silence_ms))

    def _infer(
        self,
        audio: Any,
        *,
        param_dict: dict[str, Any],
        end_silence_ms: int,
    ) -> Any:
        with self._inference_lock:
            # funasr-onnx stores max_end_sil on the otherwise stateless model object.
            # Temporarily setting it under the same lock lets sessions use immutable
            # configuration snapshots without loading duplicate ONNX sessions.
            previous_end_silence = getattr(self._backend, "max_end_sil", None)
            can_configure = hasattr(self._backend, "max_end_sil")
            if can_configure:
                self._backend.max_end_sil = end_silence_ms
            try:
                return self._backend(audio_in=audio, param_dict=param_dict)
            finally:
                if can_configure:
                    self._backend.max_end_sil = previous_end_silence


class StreamingVadSession:
    """Reassembles PCM chunks and emits one terminal event for one utterance."""

    def __init__(self, engine: FsmnVadEngine, config: VadConfig) -> None:
        self.engine = engine
        self.config = config
        self._pending_pcm = bytearray()
        self._all_pcm = bytearray()
        self._param_dict: dict[str, Any] = {"in_cache": []}
        self._processed_real_samples = 0
        self._speech_start_ms: int | None = None
        self._terminal = False
        self._finalized = False

    @property
    def finished(self) -> bool:
        return self._terminal

    def feed_pcm16(self, data: bytes, *, is_final: bool = False) -> list[VadEvent]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("PCM input must be bytes-like")
        pcm = bytes(data)
        if len(pcm) % 2:
            raise ValueError("PCM16 input must contain an even number of bytes")
        if self._finalized or self._terminal:
            return []

        self._pending_pcm.extend(pcm)
        self._all_pcm.extend(pcm)
        events: list[VadEvent] = []

        while len(self._pending_pcm) >= self.config.frame_bytes and not self._terminal:
            frame = bytes(self._pending_pcm[: self.config.frame_bytes])
            del self._pending_pcm[: self.config.frame_bytes]
            events.extend(
                self._process_frame(
                    frame,
                    real_samples=self.config.frame_samples,
                    is_final=False,
                )
            )

        if is_final and not self._terminal:
            real_samples = len(self._pending_pcm) // 2
            if self._pending_pcm:
                padding = self.config.frame_bytes - len(self._pending_pcm)
                final_frame = bytes(self._pending_pcm) + (b"\x00" * padding)
                self._pending_pcm.clear()
            else:
                # Never invoke Fsmn_vad_online with empty final input. Its
                # feats.size == 0 branch references an unassigned vad_scorer.
                final_frame = b"\x00" * self.config.frame_bytes
            events.extend(
                self._process_frame(
                    final_frame,
                    real_samples=real_samples,
                    is_final=True,
                )
            )
            self._finalized = True

            if not self._terminal:
                if self._speech_start_ms is None:
                    events.append(
                        VadEvent(
                            type="no_speech",
                            end_ms=self._current_real_ms,
                            reason="final",
                        )
                    )
                else:
                    events.append(self._finish_speech(self._current_real_ms, reason="final"))
                self._terminal = True

        return events

    @property
    def _current_real_ms(self) -> int:
        return int(round(self._processed_real_samples * 1_000 / self.config.sample_rate))

    def _process_frame(
        self,
        frame_pcm: bytes,
        *,
        real_samples: int,
        is_final: bool,
    ) -> list[VadEvent]:
        try:
            import numpy as np
        except ImportError as exc:
            raise VadDependencyError("numpy is not installed") from exc

        audio = np.frombuffer(frame_pcm, dtype="<i2").astype(np.float32) / 32_768.0
        self._param_dict["is_final"] = is_final
        raw_segments = self.engine._infer(
            audio,
            param_dict=self._param_dict,
            end_silence_ms=self.config.end_silence_ms,
        )
        self._processed_real_samples += real_samples

        events = self._handle_segments(
            raw_segments,
            end_reason="final" if is_final else "silence",
        )
        if self._terminal:
            return events

        now_ms = self._current_real_ms
        if self._speech_start_ms is None and now_ms >= self.config.initial_silence_ms:
            self._terminal = True
            events.append(
                VadEvent(type="no_speech", end_ms=now_ms, reason="initial_timeout")
            )
        elif (
            self._speech_start_ms is not None
            and now_ms - self._speech_start_ms >= self.config.max_utterance_ms
        ):
            forced_end_ms = self._speech_start_ms + self.config.max_utterance_ms
            self._terminal = True
            events.append(self._finish_speech(forced_end_ms, reason="max_utterance", forced=True))
        return events

    def _handle_segments(self, raw_segments: Any, *, end_reason: str) -> list[VadEvent]:
        events: list[VadEvent] = []
        for raw_start_ms, raw_end_ms in _iter_segment_pairs(raw_segments):
            if raw_start_ms >= 0 and self._speech_start_ms is None:
                start_ms = max(0, min(raw_start_ms, self._current_real_ms))
                self._speech_start_ms = start_ms
                events.append(VadEvent(type="speech_start", start_ms=start_ms))

            if raw_end_ms >= 0 and self._speech_start_ms is not None:
                end_ms = max(
                    self._speech_start_ms,
                    min(raw_end_ms, self._current_real_ms),
                )
                events.append(self._finish_speech(end_ms, reason=end_reason))
                self._terminal = True
                break
        return events

    def _finish_speech(
        self,
        end_ms: int,
        *,
        reason: str,
        forced: bool = False,
    ) -> VadEvent:
        assert self._speech_start_ms is not None
        event_type: VadEventType = "forced_end" if forced else "speech_end"
        return VadEvent(
            type=event_type,
            start_ms=self._speech_start_ms,
            end_ms=end_ms,
            reason=reason,
            audio_pcm=self._slice_utterance(self._speech_start_ms, end_ms),
        )

    def _slice_utterance(self, start_ms: int, end_ms: int) -> bytes:
        padded_start_ms = max(0, start_ms - self.config.pre_pad_ms)
        padded_end_ms = end_ms + self.config.post_pad_ms
        start_sample = padded_start_ms * self.config.sample_rate // 1_000
        end_sample = math.ceil(padded_end_ms * self.config.sample_rate / 1_000)
        start_byte = start_sample * 2
        end_byte = min(len(self._all_pcm), end_sample * 2)
        return bytes(self._all_pcm[start_byte:end_byte])


def _iter_segment_pairs(value: Any) -> Iterator[tuple[int, int]]:
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            yield int(round(value[0])), int(round(value[1]))
            return
        for item in value:
            yield from _iter_segment_pairs(item)


def pcm16_to_wav_bytes(pcm: bytes, *, sample_rate: int = 16_000) -> bytes:
    if not isinstance(pcm, (bytes, bytearray, memoryview)):
        raise TypeError("PCM input must be bytes-like")
    audio = bytes(pcm)
    if len(audio) % 2:
        raise ValueError("PCM16 input must contain an even number of bytes")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio)
    return output.getvalue()


__all__ = [
    "FsmnVadEngine",
    "StreamingVadSession",
    "VadConfig",
    "VadDependencyError",
    "VadEvent",
    "pcm16_to_wav_bytes",
]
