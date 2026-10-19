from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


SAMPLE_RATE = 16_000


def decode_mp3_to_wav(source: Path, target: Path) -> dict[str, int | str]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to prepare the bundled WAV samples") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    with av.open(str(source)) as container, wave.open(str(target), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        stream = container.streams.audio[0]
        for decoded in container.decode(stream):
            for frame in resampler.resample(decoded):
                samples = frame.to_ndarray().reshape(-1)[: frame.samples]
                output.writeframes(samples.astype("<i2", copy=False).tobytes())
                sample_count += frame.samples
        for frame in resampler.resample(None):
            samples = frame.to_ndarray().reshape(-1)[: frame.samples]
            output.writeframes(samples.astype("<i2", copy=False).tobytes())
            sample_count += frame.samples
    return {
        "path": _manifest_path(target),
        "sample_rate": SAMPLE_RATE,
        "duration_ms": round(sample_count * 1_000 / SAMPLE_RATE),
    }


def combine_wavs(first: Path, second: Path, target: Path, *, silence_ms: int) -> dict[str, int | str]:
    first_pcm = _read_pcm(first)
    second_pcm = _read_pcm(second)
    silence = b"\x00\x00" * (SAMPLE_RATE * silence_ms // 1_000)
    payload = first_pcm + silence + second_pcm
    _write_pcm(target, payload)
    return {
        "path": _manifest_path(target),
        "sample_rate": SAMPLE_RATE,
        "duration_ms": round(len(payload) * 1_000 / (SAMPLE_RATE * 2)),
    }


def write_silence(target: Path, *, duration_ms: int) -> dict[str, int | str]:
    payload = b"\x00\x00" * (SAMPLE_RATE * duration_ms // 1_000)
    _write_pcm(target, payload)
    return {
        "path": _manifest_path(target),
        "sample_rate": SAMPLE_RATE,
        "duration_ms": duration_ms,
    }


def _read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != SAMPLE_RATE
        ):
            raise ValueError(f"sample must be mono 16 kHz PCM16: {path}")
        return source.readframes(source.getnframes())


def _write_pcm(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(payload)


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic WAV acceptance samples")
    parser.add_argument("--directory", type=Path, default=Path("samples/audio"))
    args = parser.parse_args()
    root = args.directory.resolve()
    wake_mp3 = root / "wakeup_xiaolv.mp3"
    question_mp3 = root / "question_green_river.mp3"
    if not wake_mp3.is_file() or not question_mp3.is_file():
        raise FileNotFoundError("generate wakeup_xiaolv.mp3 and question_green_river.mp3 first")

    wake_wav = root / "wakeup_xiaolv.wav"
    question_wav = root / "question_green_river.wav"
    manifest = {
        "audio_contract": "mono/16000Hz/PCM16/WAV",
        "origin": "synthetic TTS fixtures",
        "purpose": "module smoke tests only; use agreed human recordings for accuracy acceptance",
        "samples": {
            "wakeup": {
                **decode_mp3_to_wav(wake_mp3, wake_wav),
                "expected_keyword": "小绿小绿",
            },
            "question": {
                **decode_mp3_to_wav(question_mp3, question_wav),
                "expected_text": "介绍一下绿色江河",
            },
            "wakeup_then_question": {
                **combine_wavs(
                    wake_wav,
                    question_wav,
                    root / "wakeup_then_question.wav",
                    silence_ms=1_000,
                ),
                "expected_keyword": "小绿小绿",
                "expected_text": "介绍一下绿色江河",
            },
            "silence": write_silence(root / "silence_2s.wav", duration_ms=2_000),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
