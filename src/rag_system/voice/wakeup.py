from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from rag_system.voice.wakeup_models import (
    WAKEUP_DECODER_FILE,
    WAKEUP_ENCODER_FILE,
    WAKEUP_JOINER_FILE,
    WAKEUP_TOKENS_FILE,
    default_wakeup_model_dir,
    verify_wakeup_model,
)


DEFAULT_WAKE_WORD = "小绿小绿"
DEFAULT_KEYWORD_SPEC = "x iǎo l ǜ x iǎo l ǜ @小绿小绿"
DEFAULT_KEYWORDS_FILENAME = "runtime_keywords.txt"


class WakeWordDependencyError(RuntimeError):
    """Raised when the optional sherpa-onnx runtime cannot be loaded."""


class WakeWordModelError(RuntimeError):
    """Raised when the local KWS model is absent, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class WakeWordEvent:
    keyword: str

    def to_dict(self) -> dict[str, str]:
        return {"type": "wakeup", "keyword": self.keyword}


class WakeWordEngine:
    """A process-shareable sherpa-onnx KWS model with per-client streams.

    The model and inference lock live on this object. Call ``create_session``
    once for every microphone/websocket stream so decoder state is never shared
    between clients.
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        sample_rate: int = 16_000,
        num_threads: int = 1,
        max_active_paths: int = 4,
        keywords_score: float = 1.0,
        keywords_threshold: float = 0.25,
        keywords_file: Path | str | None = None,
        keyword_specs: Sequence[str] | None = None,
        keyword_words: str | Sequence[str] | None = None,
        backend: Any | None = None,
    ) -> None:
        if sample_rate != 16_000:
            raise ValueError("the selected wake-up model requires a 16000 Hz sample rate")
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if max_active_paths <= 0:
            raise ValueError("max_active_paths must be positive")
        if keywords_score <= 0:
            raise ValueError("keywords_score must be positive")
        if not 0 <= keywords_threshold <= 1:
            raise ValueError("keywords_threshold must be between 0 and 1")
        keyword_sources = sum(
            source is not None
            for source in (keywords_file, keyword_specs, keyword_words)
        )
        if keyword_sources > 1:
            raise ValueError(
                "provide only one of keywords_file, keyword_specs, or keyword_words"
            )

        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.max_active_paths = max_active_paths
        self.keywords_score = float(keywords_score)
        self.keywords_threshold = float(keywords_threshold)
        self._inference_lock = Lock()
        self._closed = False
        self._retired = False
        self._session_count = 0

        if backend is not None:
            self.model_dir = (
                Path(model_dir).expanduser().resolve() if model_dir is not None else None
            )
            self.keywords_file = (
                Path(keywords_file).expanduser().resolve()
                if keywords_file is not None
                else None
            )
            self._backend = backend
            return

        root = Path(model_dir or default_wakeup_model_dir()).expanduser().resolve()
        verification = verify_wakeup_model(root)
        if not verification.ready:
            details = "; ".join(verification.errors)
            raise WakeWordModelError(
                f"local wake-up model is not ready at {root}: {details}; "
                "download it with download_wakeup_model()"
            )

        if keyword_words is not None:
            words = (keyword_words,) if isinstance(keyword_words, str) else tuple(keyword_words)
            if not words:
                raise ValueError("at least one wake word is required")
            keyword_specs = tuple(
                keyword_spec_for_text(word, root / WAKEUP_TOKENS_FILE)
                for word in words
            )

        resolved_keywords_file = _resolve_keywords_file(
            root,
            keywords_file=keywords_file,
            keyword_specs=keyword_specs,
        )
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise WakeWordDependencyError(
                'sherpa-onnx is not installed; install with the "wakeup" or "voice" extra'
            ) from exc

        try:
            self._backend = sherpa_onnx.KeywordSpotter(
                tokens=str(root / WAKEUP_TOKENS_FILE),
                encoder=str(root / WAKEUP_ENCODER_FILE),
                decoder=str(root / WAKEUP_DECODER_FILE),
                joiner=str(root / WAKEUP_JOINER_FILE),
                keywords_file=str(resolved_keywords_file),
                num_threads=num_threads,
                sample_rate=sample_rate,
                max_active_paths=max_active_paths,
                keywords_score=keywords_score,
                keywords_threshold=keywords_threshold,
                provider="cpu",
            )
        except Exception as exc:
            raise WakeWordModelError(
                f"failed to load sherpa-onnx wake-up model at {root}: {exc}"
            ) from exc

        self.model_dir = root
        self.keywords_file = resolved_keywords_file

    def create_session(self) -> WakeWordSession:
        with self._inference_lock:
            if self._closed:
                raise RuntimeError("wake-word engine is closed")
            if self._retired:
                raise RuntimeError("wake-word engine is retired")
            stream = self._backend.create_stream()
            self._session_count += 1
        return WakeWordSession(self, stream)

    def retire(self) -> None:
        """Reject new sessions and close after existing sessions drain."""
        with self._inference_lock:
            if self._closed:
                return
            self._retired = True
            if self._session_count == 0:
                self._close_backend()

    def close(self) -> None:
        with self._inference_lock:
            if self._closed:
                return
            self._retired = True
            self._close_backend()

    def _close_backend(self) -> None:
        self._closed = True
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def _release_session(self) -> None:
        with self._inference_lock:
            if self._session_count > 0:
                self._session_count -= 1
            if self._retired and not self._closed and self._session_count == 0:
                self._close_backend()

    def _decode(self, stream: Any, samples: Any) -> str:
        if self._closed:
            raise RuntimeError("wake-word engine is closed")
        with self._inference_lock:
            stream.accept_waveform(self.sample_rate, samples)
            while self._backend.is_ready(stream):
                self._backend.decode_stream(stream)
                keyword = _keyword_from_result(self._backend.get_result(stream))
                if keyword:
                    # sherpa-onnx explicitly requires an immediate reset after a hit.
                    self._backend.reset_stream(stream)
                    return keyword
        return ""

    def _reset_stream(self, stream: Any) -> None:
        if self._closed:
            raise RuntimeError("wake-word engine is closed")
        with self._inference_lock:
            self._backend.reset_stream(stream)


class WakeWordSession:
    """Streaming PCM16 state for one microphone source."""

    def __init__(self, engine: WakeWordEngine, stream: Any) -> None:
        self.engine = engine
        self._stream = stream
        self._detected = False
        self._backend_was_reset = False
        self._closed = False

    @property
    def detected(self) -> bool:
        return self._detected

    @property
    def closed(self) -> bool:
        return self._closed

    def accept_pcm16(self, data: bytes) -> WakeWordEvent | None:
        if self._closed:
            raise RuntimeError("wake-word session is closed")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("PCM input must be bytes-like")
        pcm = bytes(data)
        if len(pcm) % 2:
            raise ValueError("PCM16 input must contain an even number of bytes")
        if not pcm or self._detected:
            return None

        try:
            import numpy as np
        except ImportError as exc:
            raise WakeWordDependencyError(
                'numpy is not installed; install with the "wakeup" or "voice" extra'
            ) from exc

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        samples *= 1.0 / 32_768.0
        keyword = self.engine._decode(self._stream, samples)
        if not keyword:
            return None

        self._detected = True
        self._backend_was_reset = True
        return WakeWordEvent(keyword=keyword)

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("wake-word session is closed")
        if not self._backend_was_reset:
            self.engine._reset_stream(self._stream)
        self._detected = False
        self._backend_was_reset = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream = None
        self.engine._release_session()


def _resolve_keywords_file(
    model_dir: Path,
    *,
    keywords_file: Path | str | None,
    keyword_specs: Sequence[str] | None,
) -> Path:
    if keywords_file is not None:
        resolved = Path(keywords_file).expanduser().resolve()
        if not resolved.is_file():
            raise WakeWordModelError(f"wake-word keywords file does not exist: {resolved}")
        return resolved

    if keyword_specs is None:
        specs = (DEFAULT_KEYWORD_SPEC,)
    elif isinstance(keyword_specs, str):
        specs = (keyword_specs,)
    else:
        specs = tuple(keyword_specs)
    _validate_keyword_specs(specs, model_dir / WAKEUP_TOKENS_FILE)
    resolved = model_dir / DEFAULT_KEYWORDS_FILENAME
    content = "".join(f"{spec.strip()}\n" for spec in specs)
    if not resolved.is_file() or resolved.read_text(encoding="utf-8") != content:
        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise WakeWordModelError(
                f"cannot write wake-word keywords file at {resolved}: {exc}; "
                "provide a writable keywords_file explicitly"
            ) from exc
    return resolved


def keyword_spec_for_text(
    word: str,
    tokens_file: Path | str | None = None,
) -> str:
    """Convert a Chinese wake word to sherpa-onnx ``ppinyin`` syntax.

    For example, ``小绿小绿`` becomes
    ``x iǎo l ǜ x iǎo l ǜ @小绿小绿``. The generated symbols are checked
    against the selected model's ``tokens.txt`` before being returned.
    """

    cleaned_word = "".join(str(word or "").split())
    if not cleaned_word:
        raise ValueError("wake word cannot be empty")
    if any(not "\u3400" <= char <= "\u9fff" for char in cleaned_word):
        raise ValueError("the selected wake-up model supports Chinese wake words only")

    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise WakeWordDependencyError(
            'pypinyin is not installed; install with the "wakeup" or "voice" extra'
        ) from exc

    initials = lazy_pinyin(cleaned_word, style=Style.INITIALS, strict=True)
    finals = lazy_pinyin(cleaned_word, style=Style.FINALS_TONE, strict=True)
    full_syllables = lazy_pinyin(cleaned_word, style=Style.TONE, strict=True)
    if not (len(initials) == len(finals) == len(full_syllables) == len(cleaned_word)):
        raise ValueError(f"failed to tokenize Chinese wake word: {cleaned_word}")

    keyword_tokens: list[str] = []
    for initial, final, full_syllable in zip(initials, finals, full_syllables):
        if initial:
            keyword_tokens.append(initial)
        if final:
            keyword_tokens.append(final)
        elif not initial and full_syllable:
            # Interjections such as “嗯” have no regular initial/final in
            # pypinyin but are represented as a complete syllable in this model.
            keyword_tokens.append(full_syllable)

    if not keyword_tokens:
        raise ValueError(f"failed to tokenize Chinese wake word: {cleaned_word}")

    spec = f"{' '.join(keyword_tokens)} @{cleaned_word}"
    resolved_tokens = Path(
        tokens_file or (default_wakeup_model_dir() / WAKEUP_TOKENS_FILE)
    ).expanduser().resolve()
    if not resolved_tokens.is_file():
        raise WakeWordModelError(f"wake-word tokens file does not exist: {resolved_tokens}")
    _validate_keyword_specs((spec,), resolved_tokens)
    return spec


def _validate_keyword_specs(specs: Sequence[str], tokens_file: Path) -> None:
    if not specs:
        raise ValueError("at least one keyword specification is required")
    available_tokens = {
        line.rsplit(maxsplit=1)[0]
        for line in tokens_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for spec in specs:
        cleaned = str(spec or "").strip()
        if not cleaned or "@" not in cleaned:
            raise ValueError(
                "each keyword specification must contain tokenized pinyin and an @label"
            )
        missing = [
            token
            for token in cleaned.split()
            if not token.startswith(("@", ":", "#")) and token not in available_tokens
        ]
        if missing:
            raise ValueError(
                f"keyword specification contains tokens absent from tokens.txt: {missing}"
            )


def _keyword_from_result(result: Any) -> str:
    if isinstance(result, str):
        cleaned = result.strip()
        if not cleaned:
            return ""
        # Some older bindings exposed the structured result as JSON text.
        if cleaned.startswith("{"):
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                return cleaned
            return str(payload.get("keyword") or "").strip()
        return cleaned
    return str(getattr(result, "keyword", "") or "").strip()


__all__ = [
    "DEFAULT_KEYWORD_SPEC",
    "DEFAULT_KEYWORDS_FILENAME",
    "DEFAULT_WAKE_WORD",
    "WakeWordDependencyError",
    "WakeWordEngine",
    "WakeWordEvent",
    "WakeWordModelError",
    "WakeWordSession",
    "keyword_spec_for_text",
]
