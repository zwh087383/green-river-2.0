from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Literal


DEFAULT_SETTINGS_PATH = Path("data/settings.json")
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MULTIMODAL_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
DEFAULT_DASHSCOPE_TEXT_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "text-embedding/text-embedding"
)
DEFAULT_DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-2"
DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_EMBEDDING_URL = "http://127.0.0.1:8765/v1/embeddings"
DEFAULT_AUDIO_REPLY_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_IMAGE_EMBEDDING_MODEL = "qwen3-vl-embedding"
DEFAULT_IMAGE_EMBEDDING_DIMENSIONS = 2048
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_ANSWER_MODEL = "qwen-audio-3.1-realtime-plus"
DEFAULT_ROUTER_MODEL = "qwen3.7-flash"
DEFAULT_DEEPSEEK_TURN_PROMPT = """判断当前用户消息属于下列哪一种，严格输出对应的 turn_signal：
- [[IGNORE_TURN]]：只有无实义的拟声词、停顿词或环境误识别，例如“嗯”“啊”“呃”“哦”，且没有问题、请求、确认或其他语义。
- [[END_SESSION]]：明确表示本轮对话结束，例如“没有别的问题了”“先这样，再见”“不用了，谢谢”。仅当没有新问题、请求或转折补充时使用。
- [[ANSWER]]：其余全部情况，包括任何可回答内容、含有问题的短句、以及“嗯我想问……”或“没别的问题，但是……”等带实义的表达。
turn_reason 用一句简短中文说明依据。轮次控制标记只用于程序控制，不能改写用户语义。"""
DEEPSEEK_TURN_PROMPT_MARKERS = (
    "[[IGNORE_TURN]]",
    "[[END_SESSION]]",
    "[[ANSWER]]",
)
MAX_DEEPSEEK_TURN_PROMPT_LENGTH = 8000
DEFAULT_RERANKER_MODEL = "qwen3-rerank"
DEFAULT_CHUNK_SIZE = 2400
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_CONVERSATION_MAX_ROUNDS = 40
DEFAULT_STT_MODEL = "medium"
DEFAULT_STT_TEMPERATURE = 0.0
DEFAULT_STT_BEAM_SIZE = 5
DEFAULT_STT_BEST_OF = 5
DEFAULT_STT_PATIENCE = 1.0
DEFAULT_STT_CORRECTION_ENABLED = True
DEFAULT_ASR_PROVIDER = "dashscope_realtime"
DEFAULT_DASHSCOPE_ASR_MODEL = "qwen3-asr-flash-realtime"
DEFAULT_VAD_ENABLED = True
DEFAULT_VAD_END_SILENCE_MS = 400
DEFAULT_WAKEUP_ENABLED = True
DEFAULT_WAKEUP_WORD = "小绿小绿"
DEFAULT_WAKEUP_SCORE = 1.0
DEFAULT_WAKEUP_THRESHOLD = 0.25
DEFAULT_WAKEUP_COOLDOWN_MS = 800
DEFAULT_WAKEUP_MODEL = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
DEFAULT_WAKEUP_PROVIDER = "cpu"
DEFAULT_WAKEUP_NUM_THREADS = 1
DEFAULT_TTS_NORMALIZE_ENABLED = True
DEFAULT_TTS_SSML_ENABLED = False
DEFAULT_TTS_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_TTS_RATE = "+0%"
DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

ApiKeySource = Literal["file", "env", "none"]
AsrProvider = Literal["dashscope_realtime", "local_whisper"]
WakeupProvider = Literal["cpu"]
AsrCorrectionRule = tuple[str, str]

_SETTINGS_LOCKS_GUARD = Lock()
_SETTINGS_LOCKS: dict[str, RLock] = {}


def _settings_lock_for(path: Path) -> RLock:
    key = str(path.expanduser().resolve())
    with _SETTINGS_LOCKS_GUARD:
        lock = _SETTINGS_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _SETTINGS_LOCKS[key] = lock
        return lock


@dataclass(frozen=True)
class AppSettings:
    dashscope_base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    dashscope_multimodal_url: str = DEFAULT_DASHSCOPE_MULTIMODAL_URL
    dashscope_text_embedding_url: str = DEFAULT_DASHSCOPE_TEXT_EMBEDDING_URL
    dashscope_rerank_url: str = DEFAULT_DASHSCOPE_RERANK_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    embedding_url: str = DEFAULT_EMBEDDING_URL
    audio_reply_url: str = DEFAULT_AUDIO_REPLY_URL
    reply_use_original_audio: bool = False
    audio_reply_voice: str = "longanqian_v3.1"
    image_embedding_model: str = DEFAULT_IMAGE_EMBEDDING_MODEL
    image_embedding_dimensions: int = DEFAULT_IMAGE_EMBEDDING_DIMENSIONS
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    router_model: str = DEFAULT_ROUTER_MODEL
    answer_model: str = DEFAULT_ANSWER_MODEL
    deepseek_turn_prompt: str = DEFAULT_DEEPSEEK_TURN_PROMPT
    sparse_embedding_enabled: bool = False
    reranker_enabled: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_local_files_only: bool = True
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    conversation_max_rounds: int = DEFAULT_CONVERSATION_MAX_ROUNDS
    stt_model: str = DEFAULT_STT_MODEL
    stt_temperature: float = DEFAULT_STT_TEMPERATURE
    stt_beam_size: int = DEFAULT_STT_BEAM_SIZE
    stt_best_of: int = DEFAULT_STT_BEST_OF
    stt_patience: float = DEFAULT_STT_PATIENCE
    asr_provider: AsrProvider = DEFAULT_ASR_PROVIDER
    dashscope_asr_model: str = DEFAULT_DASHSCOPE_ASR_MODEL
    asr_correction_enabled: bool = DEFAULT_STT_CORRECTION_ENABLED
    asr_correction_rules: tuple[AsrCorrectionRule, ...] = field(default_factory=tuple)
    vad_enabled: bool = DEFAULT_VAD_ENABLED
    vad_end_silence_ms: int = DEFAULT_VAD_END_SILENCE_MS
    wakeup_enabled: bool = DEFAULT_WAKEUP_ENABLED
    wakeup_word: str = DEFAULT_WAKEUP_WORD
    wakeup_score: float = DEFAULT_WAKEUP_SCORE
    wakeup_threshold: float = DEFAULT_WAKEUP_THRESHOLD
    wakeup_cooldown_ms: int = DEFAULT_WAKEUP_COOLDOWN_MS
    wakeup_model: str = DEFAULT_WAKEUP_MODEL
    wakeup_provider: WakeupProvider = DEFAULT_WAKEUP_PROVIDER
    wakeup_num_threads: int = DEFAULT_WAKEUP_NUM_THREADS
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_rate: str = DEFAULT_TTS_RATE
    dashscope_api_key: str | None = field(default=None, repr=False)
    deepseek_api_key: str | None = field(default=None, repr=False)
    api_key_source: ApiKeySource = "none"
    deepseek_api_key_source: ApiKeySource = "none"

    @property
    def has_api_key(self) -> bool:
        return bool(self.dashscope_api_key)

    @property
    def has_deepseek_api_key(self) -> bool:
        return bool(self.deepseek_api_key)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "dashscope_base_url": self.dashscope_base_url,
            "dashscope_multimodal_url": self.dashscope_multimodal_url,
            "dashscope_text_embedding_url": self.dashscope_text_embedding_url,
            "dashscope_rerank_url": self.dashscope_rerank_url,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_url": self.embedding_url,
            "audio_reply_url": self.audio_reply_url,
            "audio_reply_voice": self.audio_reply_voice,
            "reply_use_original_audio": self.reply_use_original_audio,
            "image_embedding_model": self.image_embedding_model,
            "image_embedding_dimensions": self.image_embedding_dimensions,
            "deepseek_base_url": self.deepseek_base_url,
            "router_model": self.router_model,
            "answer_model": self.answer_model,
            "deepseek_turn_prompt": self.deepseek_turn_prompt,
            "sparse_embedding_enabled": self.sparse_embedding_enabled,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": self.reranker_model,
            "reranker_local_files_only": self.reranker_local_files_only,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "conversation_max_rounds": self.conversation_max_rounds,
            "stt_model": self.stt_model,
            "stt_temperature": self.stt_temperature,
            "stt_beam_size": self.stt_beam_size,
            "stt_best_of": self.stt_best_of,
            "stt_patience": self.stt_patience,
            "asr_provider": self.asr_provider,
            "dashscope_asr_model": self.dashscope_asr_model,
            "asr_correction_enabled": self.asr_correction_enabled,
            "asr_correction_rules": [
                {"wrong": wrong, "correct": correct}
                for wrong, correct in self.asr_correction_rules
            ],
            "vad_enabled": self.vad_enabled,
            "vad_end_silence_ms": self.vad_end_silence_ms,
            "wakeup_enabled": self.wakeup_enabled,
            "wakeup_word": self.wakeup_word,
            "wakeup_score": self.wakeup_score,
            "wakeup_threshold": self.wakeup_threshold,
            "wakeup_cooldown_ms": self.wakeup_cooldown_ms,
            "wakeup_model": self.wakeup_model,
            "wakeup_provider": self.wakeup_provider,
            "wakeup_num_threads": self.wakeup_num_threads,
            "tts_voice": self.tts_voice,
            "tts_rate": self.tts_rate,
            "has_api_key": self.has_api_key,
            "masked_api_key": mask_api_key(self.dashscope_api_key),
            "api_key_source": self.api_key_source,
            "has_deepseek_api_key": self.has_deepseek_api_key,
            "masked_deepseek_api_key": mask_api_key(self.deepseek_api_key),
            "deepseek_api_key_source": self.deepseek_api_key_source,
        }


class SettingsStore:
    def __init__(self, path: Path | str = DEFAULT_SETTINGS_PATH) -> None:
        self.path = Path(path)
        self._transaction_lock = _settings_lock_for(self.path)

    def load(self) -> AppSettings:
        saved = self._normalise_payload(self._read_payload())
        saved_key = saved.get("dashscope_api_key")
        env_key = _clean_api_key(os.environ.get(DASHSCOPE_API_KEY_ENV))
        saved_deepseek_key = saved.get("deepseek_api_key")
        env_deepseek_key = _clean_api_key(os.environ.get(DEEPSEEK_API_KEY_ENV))

        if saved_key:
            api_key = str(saved_key)
            api_key_source: ApiKeySource = "file"
        elif env_key:
            api_key = env_key
            api_key_source = "env"
        else:
            api_key = None
            api_key_source = "none"

        if saved_deepseek_key:
            deepseek_api_key = str(saved_deepseek_key)
            deepseek_api_key_source: ApiKeySource = "file"
        elif env_deepseek_key:
            deepseek_api_key = env_deepseek_key
            deepseek_api_key_source = "env"
        else:
            deepseek_api_key = None
            deepseek_api_key_source = "none"

        return AppSettings(
            dashscope_base_url=str(saved["dashscope_base_url"]),
            dashscope_multimodal_url=str(saved["dashscope_multimodal_url"]),
            dashscope_text_embedding_url=str(saved["dashscope_text_embedding_url"]),
            dashscope_rerank_url=str(saved["dashscope_rerank_url"]),
            embedding_model=str(saved["embedding_model"]),
            embedding_dimensions=int(saved["embedding_dimensions"]),
            embedding_url=str(saved["embedding_url"]),
            audio_reply_url=str(saved["audio_reply_url"]),
            audio_reply_voice=str(saved["audio_reply_voice"]),
            reply_use_original_audio=bool(saved["reply_use_original_audio"]),
            image_embedding_model=str(saved["image_embedding_model"]),
            image_embedding_dimensions=int(saved["image_embedding_dimensions"]),
            deepseek_base_url=str(saved["deepseek_base_url"]),
            router_model=str(saved["router_model"]),
            answer_model=str(saved["answer_model"]),
            deepseek_turn_prompt=str(saved["deepseek_turn_prompt"]),
            sparse_embedding_enabled=bool(saved["sparse_embedding_enabled"]),
            reranker_enabled=bool(saved["reranker_enabled"]),
            reranker_model=str(saved["reranker_model"]),
            reranker_local_files_only=bool(saved["reranker_local_files_only"]),
            chunk_size=int(saved["chunk_size"]),
            chunk_overlap=int(saved["chunk_overlap"]),
            conversation_max_rounds=int(saved["conversation_max_rounds"]),
            stt_model=str(saved["stt_model"]),
            stt_temperature=float(saved.get("stt_temperature", DEFAULT_STT_TEMPERATURE)),
            stt_beam_size=int(saved.get("stt_beam_size", DEFAULT_STT_BEAM_SIZE)),
            stt_best_of=int(saved.get("stt_best_of", DEFAULT_STT_BEST_OF)),
            stt_patience=float(saved.get("stt_patience", DEFAULT_STT_PATIENCE)),
            asr_provider=str(saved["asr_provider"]),
            dashscope_asr_model=str(saved["dashscope_asr_model"]),
            asr_correction_enabled=bool(saved["asr_correction_enabled"]),
            asr_correction_rules=tuple(saved["asr_correction_rules"]),
            vad_enabled=bool(saved["vad_enabled"]),
            vad_end_silence_ms=int(saved["vad_end_silence_ms"]),
            wakeup_enabled=bool(saved["wakeup_enabled"]),
            wakeup_word=str(saved["wakeup_word"]),
            wakeup_score=float(saved["wakeup_score"]),
            wakeup_threshold=float(saved["wakeup_threshold"]),
            wakeup_cooldown_ms=int(saved["wakeup_cooldown_ms"]),
            wakeup_model=str(saved["wakeup_model"]),
            wakeup_provider=str(saved["wakeup_provider"]),
            wakeup_num_threads=int(saved["wakeup_num_threads"]),
            tts_voice=str(saved["tts_voice"]),
            tts_rate=str(saved["tts_rate"]),
            dashscope_api_key=api_key,
            deepseek_api_key=deepseek_api_key,
            api_key_source=api_key_source,
            deepseek_api_key_source=deepseek_api_key_source,
        )

    def public_status(self) -> dict[str, Any]:
        return self.load().to_public_dict()

    def save_api_key(self, api_key: str) -> AppSettings:
        cleaned = validate_api_key(api_key)
        return self._update_payload({"dashscope_api_key": cleaned})

    def delete_api_key(self) -> AppSettings:
        return self._update_payload(removals=("dashscope_api_key",))

    def save_deepseek_api_key(self, api_key: str) -> AppSettings:
        cleaned = validate_api_key(api_key)
        return self._update_payload({"deepseek_api_key": cleaned})

    def delete_deepseek_api_key(self) -> AppSettings:
        return self._update_payload(removals=("deepseek_api_key",))

    def save_deepseek_turn_prompt(self, prompt: str) -> AppSettings:
        return self._update_payload(
            {"deepseek_turn_prompt": validate_deepseek_turn_prompt(prompt)}
        )

    def reset_deepseek_turn_prompt(self) -> AppSettings:
        return self._update_payload(removals=("deepseek_turn_prompt",))

    def save_chunking(self, chunk_size: int, chunk_overlap: int) -> AppSettings:
        validated_size, validated_overlap = validate_chunking(chunk_size, chunk_overlap)
        return self._update_payload(
            {
                "chunk_size": validated_size,
                "chunk_overlap": validated_overlap,
            }
        )

    def save_reply_input(self, enabled: bool) -> AppSettings:
        return self._update_payload({"reply_use_original_audio": enabled})

    def save_conversation_history(self, max_rounds: int) -> AppSettings:
        validated_rounds = validate_conversation_max_rounds(max_rounds)
        return self._update_payload({"conversation_max_rounds": validated_rounds})

    def save_vad(self, enabled: bool, end_silence_ms: int) -> AppSettings:
        validated_enabled, validated_end_silence_ms = validate_vad_settings(
            enabled,
            end_silence_ms,
        )
        return self._update_payload(
            {
                "vad_enabled": validated_enabled,
                "vad_end_silence_ms": validated_end_silence_ms,
            }
        )

    def save_asr(self, provider: str, model: str | None = None) -> AppSettings:
        validated_provider, validated_model = validate_asr_settings(provider, model)
        return self._update_payload(
            {
                "asr_provider": validated_provider,
                "dashscope_asr_model": validated_model,
            }
        )

    def save_asr_corrections(
        self,
        enabled: bool,
        rules: Any,
    ) -> AppSettings:
        validated_enabled, validated_rules = validate_asr_correction_settings(
            enabled,
            rules,
        )
        return self._update_payload(
            {
                "asr_correction_enabled": validated_enabled,
                "asr_correction_rules": [
                    {"wrong": wrong, "correct": correct}
                    for wrong, correct in validated_rules
                ],
            }
        )

    def save_wakeup(
        self,
        enabled: bool,
        word: str,
        score: float,
        threshold: float,
        cooldown_ms: int,
        model: str = DEFAULT_WAKEUP_MODEL,
        provider: str = DEFAULT_WAKEUP_PROVIDER,
        num_threads: int = DEFAULT_WAKEUP_NUM_THREADS,
    ) -> AppSettings:
        validated = validate_wakeup_settings(
            enabled,
            word,
            score,
            threshold,
            cooldown_ms,
            model,
            provider,
            num_threads,
        )
        return self._update_payload(
            dict(
                zip(
                    (
                        "wakeup_enabled",
                        "wakeup_word",
                        "wakeup_score",
                        "wakeup_threshold",
                        "wakeup_cooldown_ms",
                        "wakeup_model",
                        "wakeup_provider",
                        "wakeup_num_threads",
                    ),
                    validated,
                )
            )
        )

    def _update_payload(
        self,
        updates: dict[str, Any] | None = None,
        *,
        removals: tuple[str, ...] = (),
    ) -> AppSettings:
        with self._transaction_lock:
            payload = self._normalise_payload(self._read_payload())
            payload.update(updates or {})
            for key in removals:
                payload.pop(key, None)
            self._write_payload(payload)
            return self.load()

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        return data

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            temp_path.replace(self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _normalise_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalised: dict[str, Any] = {
            "dashscope_base_url": _clean_text(
                payload.get("dashscope_base_url"),
                DEFAULT_DASHSCOPE_BASE_URL,
            ),
            # Legacy settings must never select the old vector space or reply provider.
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "embedding_dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
            "embedding_url": _clean_text(payload.get("embedding_url"), DEFAULT_EMBEDDING_URL),
            "audio_reply_url": _clean_text(payload.get("audio_reply_url"), DEFAULT_AUDIO_REPLY_URL),
            "reply_use_original_audio": payload.get("reply_use_original_audio") is True,
            "audio_reply_voice": _clean_text(payload.get("audio_reply_voice"), "longanqian_v3.1"),
            "dashscope_multimodal_url": _clean_text(
                payload.get("dashscope_multimodal_url"),
                DEFAULT_DASHSCOPE_MULTIMODAL_URL,
            ),
            "dashscope_text_embedding_url": _clean_text(
                payload.get("dashscope_text_embedding_url"),
                DEFAULT_DASHSCOPE_TEXT_EMBEDDING_URL,
            ),
            "dashscope_rerank_url": _clean_text(
                payload.get("dashscope_rerank_url"),
                DEFAULT_DASHSCOPE_RERANK_URL,
            ),
            "image_embedding_model": _clean_text(
                payload.get("image_embedding_model"),
                DEFAULT_IMAGE_EMBEDDING_MODEL,
            ),
            "image_embedding_dimensions": _clean_dimension(
                payload.get("image_embedding_dimensions"),
                default=DEFAULT_IMAGE_EMBEDDING_DIMENSIONS,
            ),
            "deepseek_base_url": _clean_text(
                payload.get("deepseek_base_url"),
                DEFAULT_DEEPSEEK_BASE_URL,
            ),
            "router_model": _clean_text(payload.get("router_model"), DEFAULT_ROUTER_MODEL),
            "answer_model": DEFAULT_ANSWER_MODEL,
            "deepseek_turn_prompt": payload.get(
                "deepseek_turn_prompt",
                DEFAULT_DEEPSEEK_TURN_PROMPT,
            ),
            "sparse_embedding_enabled": False,
            "reranker_enabled": False,
            "reranker_model": _clean_text(payload.get("reranker_model"), DEFAULT_RERANKER_MODEL),
            "reranker_local_files_only": _clean_bool(payload.get("reranker_local_files_only"), True),
            "chunk_size": _clean_int(payload.get("chunk_size"), DEFAULT_CHUNK_SIZE),
            "chunk_overlap": _clean_int(payload.get("chunk_overlap"), DEFAULT_CHUNK_OVERLAP),
            "conversation_max_rounds": _clean_int(
                payload.get("conversation_max_rounds"),
                DEFAULT_CONVERSATION_MAX_ROUNDS,
            ),
            "stt_model": _clean_text(payload.get("stt_model"), DEFAULT_STT_MODEL),
            "stt_temperature": float(payload.get("stt_temperature", DEFAULT_STT_TEMPERATURE)),
            "stt_beam_size": int(payload.get("stt_beam_size", DEFAULT_STT_BEAM_SIZE)),
            "stt_best_of": int(payload.get("stt_best_of", DEFAULT_STT_BEST_OF)),
            "stt_patience": float(payload.get("stt_patience", DEFAULT_STT_PATIENCE)),
            "asr_provider": _clean_text(payload.get("asr_provider"), DEFAULT_ASR_PROVIDER),
            "dashscope_asr_model": _clean_text(
                payload.get("dashscope_asr_model"),
                DEFAULT_DASHSCOPE_ASR_MODEL,
            ),
            "asr_correction_enabled": _clean_bool(
                payload.get("asr_correction_enabled"),
                DEFAULT_STT_CORRECTION_ENABLED,
            ),
            "asr_correction_rules": payload.get("asr_correction_rules", ()),
            "vad_enabled": _clean_bool(payload.get("vad_enabled"), DEFAULT_VAD_ENABLED),
            "vad_end_silence_ms": _clean_int(
                payload.get("vad_end_silence_ms"),
                DEFAULT_VAD_END_SILENCE_MS,
            ),
            "wakeup_enabled": _clean_bool(
                payload.get("wakeup_enabled"),
                DEFAULT_WAKEUP_ENABLED,
            ),
            "wakeup_word": _clean_text(
                payload.get("wakeup_word"),
                DEFAULT_WAKEUP_WORD,
            ),
            "wakeup_score": _clean_float(
                payload.get("wakeup_score"),
                DEFAULT_WAKEUP_SCORE,
            ),
            "wakeup_threshold": _clean_float(
                payload.get("wakeup_threshold"),
                DEFAULT_WAKEUP_THRESHOLD,
            ),
            "wakeup_cooldown_ms": _clean_int(
                payload.get("wakeup_cooldown_ms"),
                DEFAULT_WAKEUP_COOLDOWN_MS,
            ),
            "wakeup_model": _clean_text(
                payload.get("wakeup_model"),
                DEFAULT_WAKEUP_MODEL,
            ),
            "wakeup_provider": _clean_text(
                payload.get("wakeup_provider"),
                DEFAULT_WAKEUP_PROVIDER,
            ),
            "wakeup_num_threads": _clean_int(
                payload.get("wakeup_num_threads"),
                DEFAULT_WAKEUP_NUM_THREADS,
            ),
            "tts_voice": _clean_text(payload.get("tts_voice"), DEFAULT_TTS_VOICE),
            "tts_rate": _clean_text(payload.get("tts_rate"), DEFAULT_TTS_RATE),
        }
        try:
            normalised["chunk_size"], normalised["chunk_overlap"] = validate_chunking(
                int(normalised["chunk_size"]),
                int(normalised["chunk_overlap"]),
            )
        except ValueError:
            normalised["chunk_size"] = DEFAULT_CHUNK_SIZE
            normalised["chunk_overlap"] = DEFAULT_CHUNK_OVERLAP
        try:
            normalised["conversation_max_rounds"] = validate_conversation_max_rounds(
                int(normalised["conversation_max_rounds"])
            )
        except ValueError:
            normalised["conversation_max_rounds"] = DEFAULT_CONVERSATION_MAX_ROUNDS
        try:
            normalised["deepseek_turn_prompt"] = validate_deepseek_turn_prompt(
                normalised["deepseek_turn_prompt"]
            )
        except ValueError:
            normalised["deepseek_turn_prompt"] = DEFAULT_DEEPSEEK_TURN_PROMPT
        try:
            normalised["asr_provider"], normalised["dashscope_asr_model"] = (
                validate_asr_settings(
                    str(normalised["asr_provider"]),
                    str(normalised["dashscope_asr_model"]),
                )
            )
        except ValueError:
            normalised["asr_provider"] = DEFAULT_ASR_PROVIDER
            normalised["dashscope_asr_model"] = DEFAULT_DASHSCOPE_ASR_MODEL
        try:
            (
                normalised["asr_correction_enabled"],
                normalised["asr_correction_rules"],
            ) = validate_asr_correction_settings(
                normalised["asr_correction_enabled"],
                normalised["asr_correction_rules"],
            )
        except ValueError:
            normalised["asr_correction_enabled"] = DEFAULT_STT_CORRECTION_ENABLED
            normalised["asr_correction_rules"] = ()
        try:
            normalised["vad_enabled"], normalised["vad_end_silence_ms"] = (
                validate_vad_settings(
                    normalised["vad_enabled"],
                    int(normalised["vad_end_silence_ms"]),
                )
            )
        except ValueError:
            normalised["vad_enabled"] = DEFAULT_VAD_ENABLED
            normalised["vad_end_silence_ms"] = DEFAULT_VAD_END_SILENCE_MS
        try:
            (
                normalised["wakeup_enabled"],
                normalised["wakeup_word"],
                normalised["wakeup_score"],
                normalised["wakeup_threshold"],
                normalised["wakeup_cooldown_ms"],
                normalised["wakeup_model"],
                normalised["wakeup_provider"],
                normalised["wakeup_num_threads"],
            ) = validate_wakeup_settings(
                normalised["wakeup_enabled"],
                str(normalised["wakeup_word"]),
                float(normalised["wakeup_score"]),
                float(normalised["wakeup_threshold"]),
                int(normalised["wakeup_cooldown_ms"]),
                str(normalised["wakeup_model"]),
                str(normalised["wakeup_provider"]),
                int(normalised["wakeup_num_threads"]),
            )
        except ValueError:
            normalised["wakeup_enabled"] = DEFAULT_WAKEUP_ENABLED
            normalised["wakeup_word"] = DEFAULT_WAKEUP_WORD
            normalised["wakeup_score"] = DEFAULT_WAKEUP_SCORE
            normalised["wakeup_threshold"] = DEFAULT_WAKEUP_THRESHOLD
            normalised["wakeup_cooldown_ms"] = DEFAULT_WAKEUP_COOLDOWN_MS
            normalised["wakeup_model"] = DEFAULT_WAKEUP_MODEL
            normalised["wakeup_provider"] = DEFAULT_WAKEUP_PROVIDER
            normalised["wakeup_num_threads"] = DEFAULT_WAKEUP_NUM_THREADS
        saved_key = _clean_api_key(payload.get("dashscope_api_key"))
        if saved_key:
            normalised["dashscope_api_key"] = saved_key
        saved_deepseek_key = _clean_api_key(payload.get("deepseek_api_key"))
        if saved_deepseek_key:
            normalised["deepseek_api_key"] = saved_deepseek_key
        return normalised


def validate_api_key(api_key: str) -> str:
    cleaned = _clean_api_key(api_key)
    if not cleaned:
        raise ValueError("API Key 不能为空")
    if len(cleaned) < 8:
        raise ValueError("API Key 太短")
    if any(char.isspace() for char in cleaned):
        raise ValueError("API Key 不能包含空白字符")
    return cleaned


def validate_deepseek_turn_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("DeepSeek 提示词必须是文本")
    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("DeepSeek 提示词不能为空")
    if len(cleaned) > MAX_DEEPSEEK_TURN_PROMPT_LENGTH:
        raise ValueError(
            f"DeepSeek 提示词不能超过 {MAX_DEEPSEEK_TURN_PROMPT_LENGTH} 个字符"
        )
    missing = [marker for marker in DEEPSEEK_TURN_PROMPT_MARKERS if marker not in cleaned]
    if missing:
        raise ValueError(f"DeepSeek 提示词缺少必要标记：{'、'.join(missing)}")
    return cleaned


def validate_chunking(chunk_size: int, chunk_overlap: int) -> tuple[int, int]:
    try:
        size = int(chunk_size)
        overlap = int(chunk_overlap)
    except (TypeError, ValueError) as exc:
        raise ValueError("Chunk 配置必须是整数") from exc
    if size < 500 or size > 8000:
        raise ValueError("chunk_size 需要在 500 到 8000 之间")
    if overlap < 0 or overlap > 2000:
        raise ValueError("chunk_overlap 需要在 0 到 2000 之间")
    if overlap >= size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    return size, overlap


def validate_conversation_max_rounds(max_rounds: int) -> int:
    try:
        rounds = int(max_rounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("历史轮数必须是整数") from exc
    if rounds < 1 or rounds > 100:
        raise ValueError("历史轮数需要在 1 到 100 之间")
    return rounds


def validate_asr_settings(provider: str, model: str | None = None) -> tuple[AsrProvider, str]:
    cleaned_provider = str(provider or "").strip().lower()
    if cleaned_provider not in {"dashscope_realtime", "local_whisper"}:
        raise ValueError("asr_provider must be dashscope_realtime or local_whisper")
    cleaned_model = str(model or DEFAULT_DASHSCOPE_ASR_MODEL).strip()
    if not cleaned_model:
        raise ValueError("dashscope_asr_model cannot be empty")
    return cleaned_provider, cleaned_model  # type: ignore[return-value]


def validate_asr_correction_settings(
    enabled: bool,
    rules: Any,
) -> tuple[bool, tuple[AsrCorrectionRule, ...]]:
    if not isinstance(enabled, bool):
        raise ValueError("asr_correction_enabled must be a boolean")

    if isinstance(rules, dict):
        candidates: list[Any] = list(rules.items())
    elif isinstance(rules, (list, tuple)):
        candidates = list(rules)
    else:
        raise ValueError("asr_correction_rules must be a list")
    if len(candidates) > 200:
        raise ValueError("asr_correction_rules cannot contain more than 200 rules")

    validated: list[AsrCorrectionRule] = []
    seen_wrong: set[str] = set()
    total_characters = 0
    for index, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, dict):
            wrong_value = candidate.get("wrong")
            correct_value = candidate.get("correct")
        elif (
            isinstance(candidate, (list, tuple))
            and len(candidate) == 2
        ):
            wrong_value, correct_value = candidate
        else:
            raise ValueError(f"asr_correction_rules item {index} must contain wrong and correct")

        if not isinstance(wrong_value, str) or not isinstance(correct_value, str):
            raise ValueError(f"asr_correction_rules item {index} must use strings")
        wrong = wrong_value.strip()
        correct = correct_value.strip()
        if not wrong or not correct:
            raise ValueError(f"asr_correction_rules item {index} cannot be empty")
        if len(wrong) > 64 or len(correct) > 64:
            raise ValueError(
                f"asr_correction_rules item {index} must use at most 64 characters per value"
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in wrong + correct):
            raise ValueError(f"asr_correction_rules item {index} cannot contain control characters")
        if wrong == correct:
            raise ValueError(f"asr_correction_rules item {index} must change the text")
        if wrong in seen_wrong:
            raise ValueError(f"asr_correction_rules contains duplicate wrong value: {wrong}")
        total_characters += len(wrong) + len(correct)
        if total_characters > 8192:
            raise ValueError("asr_correction_rules total length cannot exceed 8192 characters")
        seen_wrong.add(wrong)
        validated.append((wrong, correct))
    return enabled, tuple(validated)


def validate_vad_settings(enabled: bool, end_silence_ms: int) -> tuple[bool, int]:
    if not isinstance(enabled, bool):
        raise ValueError("vad_enabled must be a boolean")
    try:
        silence_ms = int(end_silence_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("vad_end_silence_ms must be an integer") from exc
    if silence_ms < 200 or silence_ms > 3000:
        raise ValueError("vad_end_silence_ms must be between 200 and 3000")
    if silence_ms % 100 != 0:
        raise ValueError("vad_end_silence_ms must use 100 ms steps")
    return enabled, silence_ms


def validate_wakeup_settings(
    enabled: bool,
    word: str,
    score: float,
    threshold: float,
    cooldown_ms: int,
    model: str = DEFAULT_WAKEUP_MODEL,
    provider: str = DEFAULT_WAKEUP_PROVIDER,
    num_threads: int = DEFAULT_WAKEUP_NUM_THREADS,
) -> tuple[bool, str, float, float, int, str, WakeupProvider, int]:
    if not isinstance(enabled, bool):
        raise ValueError("wakeup_enabled must be a boolean")

    cleaned_word = "".join(str(word or "").split())
    if len(cleaned_word) < 2 or len(cleaned_word) > 32:
        raise ValueError("wakeup_word must contain between 2 and 32 characters")
    if any(not "\u3400" <= character <= "\u9fff" for character in cleaned_word):
        raise ValueError("wakeup_word must contain Chinese characters only")

    try:
        cleaned_score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("wakeup_score must be a number") from exc
    if not math.isfinite(cleaned_score) or cleaned_score < 0.1 or cleaned_score > 10.0:
        raise ValueError("wakeup_score must be between 0.1 and 10.0")

    try:
        cleaned_threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("wakeup_threshold must be a number") from exc
    if (
        not math.isfinite(cleaned_threshold)
        or cleaned_threshold < 0.01
        or cleaned_threshold > 1.0
    ):
        raise ValueError("wakeup_threshold must be between 0.01 and 1.0")

    try:
        cleaned_cooldown_ms = int(cooldown_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("wakeup_cooldown_ms must be an integer") from exc
    if cleaned_cooldown_ms < 0 or cleaned_cooldown_ms > 60_000:
        raise ValueError("wakeup_cooldown_ms must be between 0 and 60000")

    cleaned_model = str(model or "").strip()
    if cleaned_model != DEFAULT_WAKEUP_MODEL:
        raise ValueError(f"wakeup_model must be {DEFAULT_WAKEUP_MODEL}")

    cleaned_provider = str(provider or "").strip().lower()
    if cleaned_provider != DEFAULT_WAKEUP_PROVIDER:
        raise ValueError("wakeup_provider must be cpu")

    try:
        cleaned_num_threads = int(num_threads)
    except (TypeError, ValueError) as exc:
        raise ValueError("wakeup_num_threads must be an integer") from exc
    if cleaned_num_threads != DEFAULT_WAKEUP_NUM_THREADS:
        raise ValueError("wakeup_num_threads must be 1")

    return (
        enabled,
        cleaned_word,
        cleaned_score,
        cleaned_threshold,
        cleaned_cooldown_ms,
        cleaned_model,
        cleaned_provider,  # type: ignore[return-value]
        cleaned_num_threads,
    )


def mask_api_key(api_key: str | None) -> str | None:
    cleaned = _clean_api_key(api_key)
    if not cleaned:
        return None
    if len(cleaned) <= 8:
        return "****"
    prefix_length = 3 if cleaned.startswith("sk-") else 4
    return f"{cleaned[:prefix_length]}****{cleaned[-4:]}"


def _clean_api_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_text(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip()
    return cleaned or default


def _clean_dimension(value: Any, default: int = DEFAULT_EMBEDDING_DIMENSIONS) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return default
    if dimension <= 0:
        return default
    return dimension


def _clean_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default
