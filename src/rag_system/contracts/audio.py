from __future__ import annotations

import sys
import wave
from array import array
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator


DEFAULT_SAMPLE_RATE = 16_000


class AudioInputError(ValueError):
    """Raised when a module-test audio input cannot be normalized."""


@dataclass(frozen=True, slots=True)
class AudioSpec:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    sample_width: int = 2
    encoding: str = "pcm_s16le"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 1:
            raise ValueError("module audio must be mono")
        if self.sample_width != 2 or self.encoding != "pcm_s16le":
            raise ValueError("module audio must use little-endian signed PCM16")

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    pcm16: bytes
    spec: AudioSpec = AudioSpec()

    def __post_init__(self) -> None:
        if not isinstance(self.pcm16, bytes):
            raise TypeError("pcm16 must be bytes")
        if len(self.pcm16) % self.spec.sample_width:
            raise AudioInputError("PCM16 byte length must be even")

    @property
    def duration_ms(self) -> int:
        if not self.pcm16:
            return 0
        return round(len(self.pcm16) * 1_000 / self.spec.bytes_per_second)

    def chunks(self, chunk_ms: int = 100) -> Iterator[bytes]:
        return iter_pcm_chunks(self, chunk_ms=chunk_ms)

    def to_wav_bytes(self) -> bytes:
        return pcm16_to_wav_bytes(self)


def load_wav_pcm16(
    source: str | Path | bytes | bytearray | memoryview,
    *,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioBuffer:
    """Load WAV input and normalize it to mono, little-endian PCM16.

    The implementation intentionally has no third-party dependency so every
    standalone module test accepts the same input on a clean installation.
    """

    if isinstance(source, (str, Path)):
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise AudioInputError(f"audio file does not exist: {source_path}")
        payload = source_path.read_bytes()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    else:
        raise TypeError("source must be a WAV path or bytes-like object")

    if not payload:
        raise AudioInputError("audio input is empty")

    try:
        with wave.open(BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            compression = handle.getcomptype()
            frames = handle.readframes(handle.getnframes())
    except (EOFError, wave.Error) as exc:
        raise AudioInputError("input is not a valid WAV file") from exc

    if compression != "NONE":
        raise AudioInputError("compressed WAV input is not supported")
    if sample_width != 2:
        raise AudioInputError("WAV input must use 16-bit PCM samples")
    if channels <= 0:
        raise AudioInputError("WAV input has no channels")
    if sample_rate <= 0:
        raise AudioInputError("WAV input has an invalid sample rate")

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    mono = _downmix(samples, channels)
    normalized = _resample_linear(mono, sample_rate, target_sample_rate)
    if sys.byteorder != "little":
        normalized.byteswap()
    return AudioBuffer(
        pcm16=normalized.tobytes(),
        spec=AudioSpec(sample_rate=target_sample_rate),
    )


def iter_pcm_chunks(audio: AudioBuffer | bytes, *, chunk_ms: int = 100) -> Iterator[bytes]:
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    if isinstance(audio, AudioBuffer):
        payload = audio.pcm16
        spec = audio.spec
    elif isinstance(audio, bytes):
        payload = audio
        spec = AudioSpec()
    else:
        raise TypeError("audio must be AudioBuffer or PCM16 bytes")
    if len(payload) % 2:
        raise AudioInputError("PCM16 byte length must be even")

    chunk_bytes = spec.bytes_per_second * chunk_ms // 1_000
    chunk_bytes -= chunk_bytes % spec.sample_width
    if chunk_bytes <= 0:
        raise ValueError("chunk_ms is too small for the audio format")
    for offset in range(0, len(payload), chunk_bytes):
        chunk = payload[offset : offset + chunk_bytes]
        if chunk:
            yield chunk


def pcm16_to_wav_bytes(audio: AudioBuffer | bytes, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    if isinstance(audio, AudioBuffer):
        payload = audio.pcm16
        spec = audio.spec
    elif isinstance(audio, bytes):
        payload = audio
        spec = AudioSpec(sample_rate=sample_rate)
    else:
        raise TypeError("audio must be AudioBuffer or PCM16 bytes")
    if len(payload) % 2:
        raise AudioInputError("PCM16 byte length must be even")

    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(spec.channels)
        handle.setsampwidth(spec.sample_width)
        handle.setframerate(spec.sample_rate)
        handle.writeframes(payload)
    return output.getvalue()


def write_wav_file(path: str | Path, audio: AudioBuffer | bytes) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pcm16_to_wav_bytes(audio))
    return target


def _downmix(samples: array[int], channels: int) -> array[int]:
    if channels == 1:
        return array("h", samples)
    frame_count = len(samples) // channels
    mono = array("h")
    for frame_index in range(frame_count):
        start = frame_index * channels
        value = round(sum(samples[start : start + channels]) / channels)
        mono.append(max(-32_768, min(32_767, value)))
    return mono


def _resample_linear(samples: array[int], source_rate: int, target_rate: int) -> array[int]:
    if source_rate == target_rate or not samples:
        return array("h", samples)
    output_count = max(1, round(len(samples) * target_rate / source_rate))
    output = array("h")
    scale = source_rate / target_rate
    last_index = len(samples) - 1
    for output_index in range(output_count):
        source_position = min(output_index * scale, last_index)
        left_index = int(source_position)
        right_index = min(left_index + 1, last_index)
        fraction = source_position - left_index
        value = round(
            samples[left_index] * (1.0 - fraction)
            + samples[right_index] * fraction
        )
        output.append(max(-32_768, min(32_767, value)))
    return output


__all__ = [
    "AudioBuffer",
    "AudioInputError",
    "AudioSpec",
    "DEFAULT_SAMPLE_RATE",
    "iter_pcm_chunks",
    "load_wav_pcm16",
    "pcm16_to_wav_bytes",
    "write_wav_file",
]
