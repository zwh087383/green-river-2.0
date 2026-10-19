from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress
from collections import OrderedDict

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import base64
import mimetypes
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_system.generation import AnswerGenerationService, ConversationHistory
from rag_system.generation.live_retrieval import RetrievalPrefetch, register, discard
from rag_system.generation.history import ROUTER_HISTORY_MAX_ROUNDS
from rag_system.generation.prompt_catalog import build_llm_prompt_catalog
from rag_system.display_state import DisplayStateStore
from rag_system.indexing import ImageIndexingService, IndexingService, MissingApiKeyError
from rag_system.indexing.document_ids import document_id_for_file, document_id_for_document
from rag_system.parsing import ParsingService
from rag_system.parsing.cache import file_sha256, parsed_document_from_dict
from rag_system.settings import (
    SettingsStore,
    mask_api_key,
    validate_api_key,
    validate_chunking,
    validate_conversation_max_rounds,
)
from rag_system.vad import FsmnVadEngine, VadConfig, pcm16_to_wav_bytes
from rag_system.voice import (
    VoiceDependencyError,
    VoiceInputError,
    VoiceService,
)
from rag_system.web.display_page import DISPLAY_HTML

try:
    from rag_system.wakeup import WakeWordEngine, keyword_spec_for_text
except ImportError:  # Optional dependency/model failures must not break manual voice input.
    WakeWordEngine = None  # type: ignore[assignment,misc]
    keyword_spec_for_text = None  # type: ignore[assignment]


DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "parser_cache"
INDEX_DIR = DATA_DIR / "index"
DISPLAY_STATE_PATH = Path(os.environ.get("RAG_DATA_DIR", str(DATA_DIR))) / "runtime" / "display_state.json"
DISPLAY_EMOTIONS_DIR = Path(__file__).with_name("assets") / "emotions"
WAKEUP_ACKNOWLEDGEMENT_TEXT = "嗨，我是小绿，有什么环保问题随时问我哦！"
WAKEUP_ACKNOWLEDGEMENT_PATH = Path(__file__).with_name("assets") / "wakeup_acknowledgement.mp3"
WAKEUP_ACKNOWLEDGEMENT_URL = "/api/voice/wakeup-acknowledgement"
INTERRUPT_ACKNOWLEDGEMENT_TEXT = "我在"
INTERRUPT_ACKNOWLEDGEMENT_PATH = (
    Path(__file__).with_name("assets") / "wakeup_acknowledgement_wo-zai.mp3"
)
INTERRUPT_ACKNOWLEDGEMENT_URL = "/api/voice/interrupt-acknowledgement"

LOGGER = logging.getLogger(__name__)


def _strict_no_fallback_enabled() -> bool:
    return os.environ.get("RAG_SYSTEM_STRICT_NO_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@asynccontextmanager
async def _application_lifespan(_app: FastAPI):
    """Warm reusable engines and optionally require every preferred provider."""
    service = voice_service_factory()
    strict = _strict_no_fallback_enabled()
    if strict:
        try:
            transport = await service.open_realtime_asr(force=True)
            if transport is None:
                raise RuntimeError("DashScope realtime ASR returned no transport")
            await service.release_realtime_asr(transport)
        except Exception as exc:
            raise RuntimeError(
                f"Strict startup failed: DashScope realtime ASR is unavailable: {exc}"
            ) from exc
    try:
        if bool(getattr(settings_store.load(), "wakeup_enabled", True)):
            wakeword_engine_factory()
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Strict startup failed: offline wake-word engine is unavailable: {exc}"
            ) from exc
        LOGGER.warning(
            "Offline wake-word warm-up failed; manual voice input remains available: %s",
            exc,
        )
    if strict:
        try:
            vad_engine_factory()
        except Exception as exc:
            raise RuntimeError(
                f"Strict startup failed: offline VAD is unavailable: {exc}"
            ) from exc
    # Run the complete local retrieval path before accepting the first request:
    # embedding HTTP, Chroma, tokenisation and BM25 all initialise lazily.
    started = time.monotonic()
    try:
        await asyncio.to_thread(_warm_retrieval)
        _app.state.retrieval_warmup = {"status": "ready", "elapsed_ms": round((time.monotonic() - started) * 1000, 1)}
        LOGGER.info("Retrieval startup warm-up completed in %.1f ms", _app.state.retrieval_warmup["elapsed_ms"])
    except Exception as exc:
        _app.state.retrieval_warmup = {"status": "failed", "error": type(exc).__name__}
        if strict:
            raise RuntimeError("Strict startup failed: retrieval warm-up is unavailable") from exc
        # Keep the settings/upload UI available on a new installation, but
        # never mark a failed warm-up as ready or suppress later query errors.
        LOGGER.warning("Retrieval warm-up failed (%s); queries will retry explicitly", type(exc).__name__)
    try:
        yield
    finally:
        try:
            await service.close_asr_connections()
        except Exception as exc:
            LOGGER.warning("ASR connection shutdown failed: %s", exc)
        try:
            await service.close_tts_connection()
        except Exception as exc:
            LOGGER.warning("TTS connection shutdown failed: %s", exc)
        close_wakeword_engine()


app = FastAPI(title="RAG Document Parser", lifespan=_application_lifespan)
app.mount(
    "/api/display/emotions",
    StaticFiles(directory=str(DISPLAY_EMOTIONS_DIR)),
    name="display_emotions",
)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()
_index_jobs: dict[str, dict[str, Any]] = {}
_index_jobs_lock = Lock()
_image_index_jobs: dict[str, dict[str, Any]] = {}
_image_index_jobs_lock = Lock()
settings_store = SettingsStore(DATA_DIR / "settings.json")
display_state_store = DisplayStateStore(DISPLAY_STATE_PATH)
conversation_history = ConversationHistory()
_conversation_histories: OrderedDict[str, ConversationHistory] = OrderedDict()
_conversation_histories_lock = Lock()
_voice_service_instance: VoiceService | None = None
_voice_service_settings_store: SettingsStore | None = None
_voice_service_lock = Lock()
_vad_engine_instance: FsmnVadEngine | None = None
_vad_engine_error: str | None = None
_vad_engine_lock = Lock()
_stt_inference_lock = Lock()
_wakeword_engine_instance: Any | None = None
_wakeword_engine_settings_store: SettingsStore | None = None
_wakeword_engine_config: tuple[Any, ...] | None = None
_wakeword_engine_lock = Lock()

WAKEUP_ASR_CONNECT_BUFFER_BYTES = 16_000 * 2 * 15


def indexing_service_factory() -> IndexingService:
    return IndexingService(settings_store=settings_store, index_dir=INDEX_DIR)


def _warm_retrieval() -> None:
    indexing_service_factory().search("检索预热", top_k=1)


def image_indexing_service_factory() -> ImageIndexingService:
    return ImageIndexingService(settings_store=settings_store, index_dir=INDEX_DIR)


def answer_generation_service_factory() -> AnswerGenerationService:
    return AnswerGenerationService(
        settings_store=settings_store,
        index_dir=INDEX_DIR,
        context_top_k=4,
        conversation_history=conversation_history,
    )


def voice_service_factory() -> VoiceService:
    global _voice_service_instance, _voice_service_settings_store
    with _voice_service_lock:
        if _voice_service_instance is None or _voice_service_settings_store is not settings_store:
            _voice_service_instance = VoiceService(
                settings_store=settings_store,
                strict_no_fallback=_strict_no_fallback_enabled(),
            )
            _voice_service_settings_store = settings_store
        return _voice_service_instance


def vad_engine_factory() -> FsmnVadEngine:
    global _vad_engine_instance, _vad_engine_error
    with _vad_engine_lock:
        if _vad_engine_instance is None:
            try:
                _vad_engine_instance = FsmnVadEngine(end_silence_ms=400)
                _vad_engine_error = None
            except Exception as exc:
                _vad_engine_error = str(exc)
                raise
        return _vad_engine_instance


def wakeword_engine_factory() -> Any:
    """Create one reusable offline KWS engine; sessions remain connection-local."""
    global _wakeword_engine_config, _wakeword_engine_instance, _wakeword_engine_settings_store
    with _wakeword_engine_lock:
        settings = settings_store.load()
        config = (
            str(getattr(settings, "wakeup_model", "")),
            str(getattr(settings, "wakeup_word", "小绿小绿")),
            float(getattr(settings, "wakeup_score", 1.0)),
            float(getattr(settings, "wakeup_threshold", 0.25)),
            str(getattr(settings, "wakeup_provider", "cpu")),
            int(getattr(settings, "wakeup_num_threads", 1)),
        )
        if (
            _wakeword_engine_instance is None
            or _wakeword_engine_settings_store is not settings_store
            or _wakeword_engine_config != config
        ):
            if WakeWordEngine is None:
                raise VoiceDependencyError("离线唤醒组件尚未安装")
            replacement = WakeWordEngine(
                num_threads=config[5],
                keywords_score=config[2],
                keywords_threshold=config[3],
                keyword_words=config[1],
            )
            previous = _wakeword_engine_instance
            _wakeword_engine_instance = replacement
            _wakeword_engine_settings_store = settings_store
            _wakeword_engine_config = config
            retire_previous = getattr(previous, "retire", None)
            if callable(retire_previous):
                retire_previous()
            else:
                close_previous = getattr(previous, "close", None)
                if callable(close_previous):
                    close_previous()
        return _wakeword_engine_instance


def close_wakeword_engine() -> None:
    """Release the process-wide KWS engine and allow a later clean reload."""
    global _wakeword_engine_config, _wakeword_engine_instance, _wakeword_engine_settings_store
    with _wakeword_engine_lock:
        engine = _wakeword_engine_instance
        _wakeword_engine_instance = None
        _wakeword_engine_settings_store = None
        _wakeword_engine_config = None
    close = getattr(engine, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


class ApiKeyPayload(BaseModel):
    api_key: str


class ApiKeyTestPayload(BaseModel):
    api_key: str | None = None


class ChunkingPayload(BaseModel):
    chunk_size: int
    chunk_overlap: int


class ConversationHistoryPayload(BaseModel):
    max_rounds: int


class DeepSeekPromptPayload(BaseModel):
    prompt: str


class VadSettingsPayload(BaseModel):
    enabled: bool
    end_silence_ms: int = 400


class ASRSettingsPayload(BaseModel):
    provider: str
    model: str | None = None


class WakeupSettingsPayload(BaseModel):
    word: str
    enabled: bool | None = None


class ASRCorrectionRulePayload(BaseModel):
    wrong: str
    correct: str


class ASRCorrectionsPayload(BaseModel):
    enabled: bool
    rules: list[ASRCorrectionRulePayload]


class SearchPayload(BaseModel):
    query: str
    top_k: int = 5


class AnswerPayload(BaseModel):
    query: str
    top_k: int = 4
    audio: bool = True
    retrieval_token: str | None = Field(default=None, max_length=64)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


def _answer_service_for(payload: AnswerPayload) -> Any:
    service = answer_generation_service_factory()
    if payload.conversation_id:
        with _conversation_histories_lock:
            history = _conversation_histories.pop(payload.conversation_id, None)
            if history is None:
                history = ConversationHistory()
            _conversation_histories[payload.conversation_id] = history
            while len(_conversation_histories) > 128:
                _conversation_histories.popitem(last=False)
        service.conversation_history = history
    return service


class TTSPayload(BaseModel):
    text: str


class DocumentEnabledPayload(BaseModel):
    enabled: bool


@app.get("/api/voice/pcm-player.js")
def pcm_player_script() -> FileResponse:
    return FileResponse(Path(__file__).with_name("assets") / "qwen_pcm.js", media_type="text/javascript")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/display", response_class=HTMLResponse)
def display() -> str:
    """Full-screen kiosk shell for the robot's dedicated display."""
    return DISPLAY_HTML


@app.get("/api/display/state")
def get_display_state() -> JSONResponse:
    return JSONResponse(display_state_store.snapshot(), headers={"Cache-Control": "no-store"})


@app.get("/api/display/client.js")
def get_display_client() -> FileResponse:
    return FileResponse(Path(__file__).with_name("assets") / "display_sync.js",
                        media_type="application/javascript", headers={"Cache-Control": "no-store"})


class DisplaySessionPayload(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class DisplaySessionOpenPayload(DisplaySessionPayload):
    client_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(default=0, ge=0)


class DisplaySessionStatePayload(DisplaySessionPayload):
    sequence: int = Field(ge=0)
    state: str
    claim: bool = False


@app.post("/api/display/session")
def open_display_session(payload: DisplaySessionOpenPayload) -> JSONResponse:
    return JSONResponse(display_state_store.open_web_session(
        payload.client_id, payload.session_id, payload.generation))


@app.post("/api/display/session/state")
def sync_display_session(payload: DisplaySessionStatePayload) -> JSONResponse:
    try:
        return JSONResponse(display_state_store.sync_web_session(
            payload.session_id, payload.sequence, payload.state, claim=payload.claim))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/display/session/release")
def release_display_session(payload: DisplaySessionPayload) -> JSONResponse:
    return JSONResponse(display_state_store.release_web_session(payload.session_id))


class DisplayModePayload(BaseModel):
    mode: str
    reason: str = ""


class DisplayEmotionPayload(BaseModel):
    state: str
    text: str = ""
    reason: str = ""


@app.post("/api/display/mode")
def set_display_mode(payload: DisplayModePayload) -> JSONResponse:
    try:
        return JSONResponse(display_state_store.set_mode(payload.mode, reason=payload.reason))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/display/emotion")
def set_display_emotion(payload: DisplayEmotionPayload) -> JSONResponse:
    try:
        return JSONResponse(display_state_store.set_emotion(
            payload.state, text=payload.text, reason=payload.reason
        ))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "rag-system"})


@app.get(WAKEUP_ACKNOWLEDGEMENT_URL)
def get_wakeup_acknowledgement() -> FileResponse:
    if not WAKEUP_ACKNOWLEDGEMENT_PATH.is_file():
        raise HTTPException(status_code=503, detail="唤醒提示音文件不存在")
    return FileResponse(WAKEUP_ACKNOWLEDGEMENT_PATH, media_type="audio/mpeg")


@app.get(INTERRUPT_ACKNOWLEDGEMENT_URL)
def get_interrupt_acknowledgement() -> FileResponse:
    """Serve the fixed short prompt played after a wake-word interruption."""

    if not INTERRUPT_ACKNOWLEDGEMENT_PATH.is_file():
        raise HTTPException(status_code=503, detail="打断提示音文件不存在")
    return FileResponse(INTERRUPT_ACKNOWLEDGEMENT_PATH, media_type="audio/mpeg")


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    return JSONResponse(settings_store.public_status())


@app.get("/api/settings/llm-prompts")
def get_llm_prompts() -> JSONResponse:
    """Expose the current, credential-free LLM prompt catalog for diagnostics."""

    return JSONResponse(build_llm_prompt_catalog(settings_store.load()))


@app.post("/api/settings/api-key")
def save_api_key(payload: ApiKeyPayload) -> JSONResponse:
    try:
        settings = settings_store.save_api_key(payload.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(settings.to_public_dict())


@app.delete("/api/settings/api-key")
def delete_api_key() -> JSONResponse:
    return JSONResponse(settings_store.delete_api_key().to_public_dict())


@app.post("/api/settings/deepseek-api-key")
def save_deepseek_api_key(payload: ApiKeyPayload) -> JSONResponse:
    try:
        settings = settings_store.save_deepseek_api_key(payload.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(settings.to_public_dict())


@app.delete("/api/settings/deepseek-api-key")
def delete_deepseek_api_key() -> JSONResponse:
    return JSONResponse(settings_store.delete_deepseek_api_key().to_public_dict())


@app.post("/api/settings/deepseek-prompt")
def save_deepseek_prompt(payload: DeepSeekPromptPayload) -> JSONResponse:
    try:
        settings = settings_store.save_deepseek_turn_prompt(payload.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(settings.to_public_dict())


@app.delete("/api/settings/deepseek-prompt")
def reset_deepseek_prompt() -> JSONResponse:
    return JSONResponse(settings_store.reset_deepseek_turn_prompt().to_public_dict())


@app.post("/api/settings/chunking")
def save_chunking(payload: ChunkingPayload) -> JSONResponse:
    try:
        settings = settings_store.save_chunking(payload.chunk_size, payload.chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(settings.to_public_dict())


@app.post("/api/settings/conversation-history")
def save_conversation_history(payload: ConversationHistoryPayload) -> JSONResponse:
    try:
        max_rounds = validate_conversation_max_rounds(payload.max_rounds)
        settings = settings_store.save_conversation_history(max_rounds)
        conversation_history.set_max_rounds(max(max_rounds, ROUTER_HISTORY_MAX_ROUNDS))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(settings.to_public_dict())


@app.post("/api/settings/vad")
def save_vad_settings(payload: VadSettingsPayload) -> JSONResponse:
    if payload.end_silence_ms < 200 or payload.end_silence_ms > 3000:
        raise HTTPException(status_code=400, detail="VAD 断句间隔必须在 200 到 3000 毫秒之间")
    if payload.end_silence_ms % 100:
        raise HTTPException(status_code=400, detail="VAD 断句间隔必须为 100 毫秒的整数倍")
    try:
        settings = settings_store.save_vad(payload.enabled, payload.end_silence_ms)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(settings.to_public_dict())


@app.post("/api/settings/asr")
def save_asr_settings(payload: ASRSettingsPayload) -> JSONResponse:
    try:
        settings = settings_store.save_asr(payload.provider, payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(settings.to_public_dict())


@app.post("/api/settings/wakeup")
def save_wakeup_settings(payload: WakeupSettingsPayload) -> JSONResponse:
    if keyword_spec_for_text is None:
        raise HTTPException(status_code=503, detail="离线唤醒组件尚未安装")
    current = settings_store.load()
    try:
        keyword_spec_for_text(payload.word)
        settings = settings_store.save_wakeup(
            current.wakeup_enabled if payload.enabled is None else payload.enabled,
            payload.word,
            current.wakeup_score,
            current.wakeup_threshold,
            current.wakeup_cooldown_ms,
            current.wakeup_model,
            current.wakeup_provider,
            current.wakeup_num_threads,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"唤醒词不可用：{exc}") from exc
    return JSONResponse(settings.to_public_dict())


@app.post("/api/settings/asr-corrections")
def save_asr_corrections(payload: ASRCorrectionsPayload) -> JSONResponse:
    try:
        settings = settings_store.save_asr_corrections(
            payload.enabled,
            [rule.model_dump() for rule in payload.rules],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    service = voice_service_factory()
    refresh = getattr(service, "refresh_asr_corrections", None)
    if callable(refresh):
        refresh()
    return JSONResponse(settings.to_public_dict())


@app.post("/api/settings/test-api-key")
def test_api_key(payload: ApiKeyTestPayload | None = None) -> JSONResponse:
    candidate = payload.api_key if payload and payload.api_key is not None else None
    if candidate is None:
        candidate = settings_store.load().dashscope_api_key
    if candidate is None:
        raise HTTPException(status_code=400, detail="API Key 尚未配置")
    try:
        cleaned = validate_api_key(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {
            "ok": True,
            "live_test": False,
            "message": "API Key 格式有效；真实 embedding 连通性会在向量化模块接入后启用。",
            "checked_api_key": mask_api_key(cleaned),
            "settings": settings_store.public_status(),
        }
    )


@app.post("/api/settings/test-deepseek-api-key")
def test_deepseek_api_key(payload: ApiKeyTestPayload | None = None) -> JSONResponse:
    candidate = payload.api_key if payload and payload.api_key is not None else None
    if candidate is None:
        candidate = settings_store.load().deepseek_api_key
    if candidate is None:
        raise HTTPException(status_code=400, detail="DeepSeek API Key 尚未配置")
    try:
        cleaned = validate_api_key(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {
            "ok": True,
            "live_test": False,
            "message": "DeepSeek API Key 格式有效；真实连通性会在答案生成调用时验证。",
            "checked_api_key": mask_api_key(cleaned),
            "settings": settings_store.public_status(),
        }
    )


@app.post("/api/parse")
async def parse_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    chunk_size: int | None = Form(None),
    chunk_overlap: int | None = Form(None),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    settings = settings_store.load()
    try:
        selected_chunk_size, selected_chunk_overlap = validate_chunking(
            chunk_size if chunk_size is not None else settings.chunk_size,
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for upload in files:
        name = _safe_filename(upload.filename or "upload.bin")
        target = job_dir / name
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved_paths.append(target)

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": len(saved_paths),
            "results": [],
            "errors": [],
            "chunking": {
                "chunk_size": selected_chunk_size,
                "chunk_overlap": selected_chunk_overlap,
            },
        }

    background_tasks.add_task(_run_parse_job, job_id, saved_paths, selected_chunk_size, selected_chunk_overlap)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(job)


@app.get("/api/assets/{job_id}/{asset_id}")
def get_asset(job_id: str, asset_id: str) -> FileResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        for result in job.get("results", []):
            document = result.get("document") or {}
            for asset in document.get("assets", []):
                if asset.get("asset_id") == asset_id and asset.get("path"):
                    path = Path(asset["path"])
                    if path.exists():
                        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Asset not found")


@app.post("/api/index/jobs/{job_id}")
def index_parse_job(job_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Parse job not found")
        if job.get("status") not in {"completed", "completed_with_errors"}:
            raise HTTPException(status_code=409, detail="Parse job is not completed")
        source_results = [
            result
            for result in job.get("results", [])
            if result.get("status") == "success" and result.get("document")
        ]

    if not source_results:
        raise HTTPException(status_code=400, detail="No successfully parsed documents to index")

    index_job_id = uuid.uuid4().hex
    with _index_jobs_lock:
        _index_jobs[index_job_id] = {
            "index_job_id": index_job_id,
            "parse_job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": len(source_results),
            "results": [],
            "errors": [],
        }
    background_tasks.add_task(_run_index_job, index_job_id, source_results)
    return JSONResponse({"index_job_id": index_job_id, "status": "queued"})


@app.get("/api/index/jobs/{index_job_id}")
def get_index_job(index_job_id: str) -> JSONResponse:
    with _index_jobs_lock:
        job = _index_jobs.get(index_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Index job not found")
        return JSONResponse(job)


@app.get("/api/index/stats")
def get_index_stats() -> JSONResponse:
    service = indexing_service_factory()
    stats = asdict(service.stats())
    stats["documents"] = [asdict(document) for document in service.list_documents()]
    return JSONResponse(stats)


@app.get("/api/documents")
def list_documents() -> JSONResponse:
    service = indexing_service_factory()
    documents = [asdict(document) for document in service.list_documents()]
    return JSONResponse({"documents": documents})


@app.get("/api/documents/{document_id}")
def get_document_detail(document_id: str) -> JSONResponse:
    service = indexing_service_factory()
    detail = service.get_document_detail(document_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Document not found")
    images = image_indexing_service_factory().list_document_images(document_id)
    for image in images:
        image_path = Path(str(image.get("metadata", {}).get("image_path") or ""))
        image["image_data_url"] = _image_data_url(image_path)
    detail["images"] = images
    return JSONResponse(detail)


@app.patch("/api/documents/{document_id}")
def update_document_enabled(document_id: str, payload: DocumentEnabledPayload) -> JSONResponse:
    service = indexing_service_factory()
    document = service.set_document_enabled(document_id, payload.enabled)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return JSONResponse({"document": asdict(document)})


@app.post("/api/documents/{document_id}/reparse")
def reparse_document(document_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    record = indexing_service_factory().manifest.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    source = Path(record.source)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    settings = settings_store.load()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": 1,
            "results": [],
            "errors": [],
            "chunking": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            "source": "document_reparse",
            "document_id": document_id,
        }
    background_tasks.add_task(_run_parse_job, job_id, [source], settings.chunk_size, settings.chunk_overlap)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.post("/api/documents/{document_id}/reindex")
def reindex_document(document_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    record = indexing_service_factory().manifest.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    source = Path(record.source)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    index_job_id = uuid.uuid4().hex
    with _index_jobs_lock:
        _index_jobs[index_job_id] = {
            "index_job_id": index_job_id,
            "parse_job_id": None,
            "document_id": document_id,
            "status": "queued",
            "progress": 0,
            "total": 1,
            "results": [],
            "errors": [],
            "source": "document_reindex",
        }
    background_tasks.add_task(_run_reindex_document_job, index_job_id, document_id, source)
    return JSONResponse({"index_job_id": index_job_id, "status": "queued"})


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> JSONResponse:
    service = indexing_service_factory()
    record = service.manifest.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    image_deleted = image_indexing_service_factory().delete_document_images(document_id)
    text_deleted = service.delete_document(document_id, remove_manifest=True)
    artifacts = _delete_document_artifacts(record)
    return JSONResponse(
        {
            "document_id": document_id,
            "deleted": {
                **text_deleted,
                **image_deleted,
                **artifacts,
            },
        }
    )


@app.post("/api/search")
def search_index(payload: SearchPayload) -> JSONResponse:
    top_k = max(1, min(payload.top_k, 20))
    service = indexing_service_factory()
    try:
        results = service.search(payload.query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {
            "query": payload.query,
            "query_tokens": service.tokenize_query(payload.query),
            "results": [asdict(result) for result in results],
        }
    )


@app.post("/api/answer")
def answer_query(payload: AnswerPayload) -> JSONResponse:
    settings = settings_store.load()
    if not settings.has_api_key:
        raise HTTPException(status_code=400, detail="请先配置百炼 API Key")
    top_k = max(1, min(payload.top_k, 4))
    service = _answer_service_for(payload)
    try:
        result = service.answer(payload.query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = result.to_dict()
    if result.retrieval.get("reason") == "ignored_turn":
        response["ignored"] = True
    return JSONResponse(response)


_ANSWER_STREAM_END = object()


def _next_answer_stream_event(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _ANSWER_STREAM_END


def _web_answer_events(
    service: Any,
    query: str,
    *,
    top_k: int,
    should_cancel: Any,
    audio: bool = True,
    retrieval_token: str | None = None,
) -> Any:
    stream_answer_events = service.stream_answer_events
    try:
        parameters = inspect.signature(stream_answer_events).parameters
    except (TypeError, ValueError):
        return stream_answer_events(query, top_k=top_k)
    supports_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {"top_k": top_k}
    if "should_cancel" in parameters or supports_kwargs:
        kwargs["should_cancel"] = should_cancel
    if "include_llm_inputs" in parameters or supports_kwargs:
        kwargs["include_llm_inputs"] = True
    if "audio" in parameters or supports_kwargs:
        kwargs["audio"] = audio
    if "retrieval_token" in parameters or supports_kwargs:
        kwargs["retrieval_token"] = retrieval_token
    return stream_answer_events(query, **kwargs)


async def _watch_answer_disconnect(request: Request, cancelled: asyncio.Event) -> None:
    try:
        while not cancelled.is_set():
            if await request.is_disconnected():
                cancelled.set()
                return
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        raise
    except Exception:
        # StreamingResponse cancellation still runs the generator's finally
        # block.  A failed disconnect probe must not turn into an answer error.
        LOGGER.debug("Answer disconnect monitor stopped", exc_info=True)


@app.post("/api/answer/stream")
async def answer_query_stream(payload: AnswerPayload, request: Request) -> StreamingResponse:
    settings = settings_store.load()
    if not settings.has_api_key:
        raise HTTPException(status_code=400, detail="请先配置百炼 API Key")
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Answer query cannot be empty")
    top_k = max(1, min(payload.top_k, 4))
    service = _answer_service_for(payload)

    async def event_stream():
        cancelled = asyncio.Event()
        monitor = asyncio.create_task(_watch_answer_disconnect(request, cancelled))
        iterator: Any | None = None
        try:
            iterator = iter(
                _web_answer_events(
                    service,
                    payload.query,
                    top_k=top_k,
                    should_cancel=cancelled.is_set,
                    audio=payload.audio,
                    retrieval_token=payload.retrieval_token,
                )
            )
            while not cancelled.is_set():
                event = await asyncio.to_thread(_next_answer_stream_event, iterator)
                if event is _ANSWER_STREAM_END or cancelled.is_set():
                    break
                yield _json_line(event)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except Exception as exc:
            if not cancelled.is_set():
                yield _json_line({"type": "error", "detail": str(exc)})
        finally:
            cancelled.set()
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            close_stream = getattr(iterator, "close", None)
            if callable(close_stream):
                with suppress(Exception):
                    close_stream()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/voice/status")
def get_voice_status() -> JSONResponse:
    settings = settings_store.load()
    payload = voice_service_factory().status().to_dict()
    payload["reply_audio_available"] = settings.has_api_key
    payload["reply_model"] = settings.answer_model
    payload["reply_voice"] = settings.audio_reply_voice
    vad_enabled = bool(getattr(settings, "vad_enabled", True))
    end_silence_ms = int(getattr(settings, "vad_end_silence_ms", 400))
    cloud_vad_active = payload.get("asr_provider_active") == "dashscope_realtime"
    vad_available = cloud_vad_active
    vad_error: str | None = None
    if not cloud_vad_active:
        try:
            vad_engine_factory()
            vad_available = True
        except Exception as exc:
            vad_error = str(exc)
    errors = list(payload.get("errors") or [])
    if vad_error and vad_error not in errors:
        errors.append(vad_error)
    wakeup_enabled = bool(getattr(settings, "wakeup_enabled", True))
    wakeup_available = False
    wakeup_error: str | None = None
    if wakeup_enabled:
        try:
            wakeword_engine_factory()
            wakeup_available = True
        except Exception as exc:
            wakeup_error = str(exc)
    if wakeup_error and wakeup_error not in errors:
        errors.append(wakeup_error)
    payload.update(
        {
            "vad_available": vad_available,
            "vad_enabled": vad_enabled,
            "streaming_stt_available": bool(
                vad_enabled
                and (
                    cloud_vad_active
                    or (payload.get("stt_available") and vad_available)
                )
            ),
            "vad_provider": (
                "dashscope_server_vad" if cloud_vad_active else "local_fsmn_vad"
            ),
            "vad_model": (
                payload.get("asr_model")
                if cloud_vad_active
                else "iic/speech_fsmn_vad_zh-cn-16k-common-onnx"
            ),
            "vad_end_silence_ms": end_silence_ms,
            "vad_error": vad_error,
            "wakeup_available": wakeup_available,
            "wakeup_enabled": wakeup_enabled,
            "wakeup_word": str(getattr(settings, "wakeup_word", "小绿小绿")),
            "wakeup_model": str(getattr(settings, "wakeup_model", "")),
            "wakeup_provider": str(getattr(settings, "wakeup_provider", "cpu")),
            "wakeup_num_threads": int(getattr(settings, "wakeup_num_threads", 1)),
            "wakeup_score": float(getattr(settings, "wakeup_score", 1.0)),
            "wakeup_threshold": float(getattr(settings, "wakeup_threshold", 0.25)),
            "wakeup_cooldown_ms": int(getattr(settings, "wakeup_cooldown_ms", 800)),
            "wakeup_error": wakeup_error,
            "errors": errors,
        }
    )
    return JSONResponse(payload)


@app.post("/api/voice/stt")
async def voice_speech_to_text(audio: UploadFile = File(...)) -> JSONResponse:
    try:
        audio_bytes = await audio.read()
        text = await asyncio.to_thread(
            _run_stt_inference,
            audio_bytes,
            audio.filename or "recording.wav",
        )
    except VoiceInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VoiceDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse({"text": text})


def _run_stt_inference(audio_bytes: bytes, filename: str) -> str:
    return _run_stt_inference_with_service(
        voice_service_factory(),
        audio_bytes,
        filename,
    )


def _run_stt_inference_with_service(
    service: VoiceService,
    audio_bytes: bytes,
    filename: str,
) -> str:
    with _stt_inference_lock:
        return service.speech_to_text(audio_bytes, filename=filename)


def _validate_vad_start(message: dict[str, Any], default_end_silence_ms: int) -> VadConfig:
    if message.get("type") != "start":
        raise ValueError("第一条消息必须是 start 配置")
    if message.get("sample_rate", 16000) != 16000:
        raise ValueError("流式 VAD 只支持 16000Hz 音频")
    if message.get("channels", 1) != 1:
        raise ValueError("流式 VAD 只支持单声道音频")
    if message.get("encoding", "pcm_s16le") != "pcm_s16le":
        raise ValueError("流式 VAD 只支持 pcm_s16le 编码")
    end_silence_ms = int(message.get("end_silence_ms", default_end_silence_ms))
    if not 200 <= end_silence_ms <= 3000 or end_silence_ms % 100:
        raise ValueError("VAD 断句间隔必须为 200 到 3000 毫秒之间的 100 毫秒整数倍")
    return VadConfig(
        sample_rate=16000,
        frame_ms=100,
        end_silence_ms=end_silence_ms,
        initial_silence_ms=10000,
        max_utterance_ms=30000,
        pre_pad_ms=300,
        post_pad_ms=200,
    )


async def _send_vad_events(
    websocket: WebSocket,
    events: list[Any],
    *,
    service: VoiceService | None = None,
    provider: str | None = None,
    wakeup_word: str | None = None,
) -> tuple[bool, bool]:
    """Send VAD events and recognize completed speech; return (completed, speech_seen)."""
    speech_seen = False
    for event in events:
        event_type = str(getattr(event, "type", ""))
        if event_type == "speech_start":
            speech_seen = True
        await websocket.send_json(event.to_dict())
        if event_type == "no_speech":
            return True, speech_seen
        if event_type not in {"speech_end", "forced_end"}:
            continue
        audio_pcm = bytes(getattr(event, "audio_pcm", b"") or b"")
        if not audio_pcm:
            await websocket.send_json({"type": "no_speech", "reason": "empty_utterance"})
            return True, True
        try:
            wav_bytes = pcm16_to_wav_bytes(audio_pcm, sample_rate=16000)
            inference = _run_stt_inference
            inference_args: tuple[Any, ...] = (wav_bytes, "vad-utterance.wav")
            if service is not None:
                inference = _run_stt_inference_with_service
                inference_args = (service, wav_bytes, "vad-utterance.wav")
            text = await asyncio.to_thread(inference, *inference_args)
        except (VoiceInputError, VoiceDependencyError) as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            return True, True
        except Exception as exc:
            await websocket.send_json({"type": "error", "detail": f"语音识别失败：{exc}"})
            return True, True
        text = text.strip()
        if wakeup_word:
            text = _remove_wakeup_prefix(text, wakeup_word)
        if not text:
            await websocket.send_json({"type": "no_speech", "reason": "empty_transcript"})
            return True, True
        final_payload: dict[str, Any] = {
            "type": "final",
            "text": text,
            "reason": getattr(event, "reason", None) or event_type,
        }
        if provider:
            final_payload["provider"] = provider
        for key in ("start_ms", "end_ms"):
            value = getattr(event, key, None)
            if value is not None:
                final_payload[key] = value
        await websocket.send_json(final_payload)
        return True, True
    return False, speech_seen


def _remove_wakeup_prefix(text: str, wakeup_word: str) -> str:
    """Remove only a leading wake phrase, preserving same words inside the question."""
    value = str(text or "").strip()
    keyword = str(wakeup_word or "").strip()
    separators = " \t\r\n,，。.!！?？、:：;；-—"
    if not value or not keyword:
        return value
    if value.startswith(keyword):
        return value[len(keyword) :].lstrip(separators)
    midpoint = len(keyword) // 2
    if len(keyword) % 2 == 0 and keyword[:midpoint] == keyword[midpoint:]:
        short_word = keyword[:midpoint]
        if value.startswith(short_word):
            remainder = value[len(short_word) :].lstrip(separators)
            if remainder.startswith(short_word):
                return remainder[len(short_word) :].lstrip(separators)
    return value


async def _forward_realtime_asr_events(
    websocket: WebSocket,
    transport: Any,
    service: VoiceService,
    wakeup_word: str | None = None,
) -> None:
    prefetch = RetrievalPrefetch(indexing_service_factory())
    token = register(prefetch)
    state = {"final": False}
    try:
        await _forward_realtime_asr_with_retrieval(websocket, transport, service, wakeup_word,
                                                  prefetch, token, state)
    finally:
        if not state["final"]:
            discard(token)


async def _forward_realtime_asr_with_retrieval(
    websocket: WebSocket, transport: Any, service: VoiceService, wakeup_word: str | None,
    prefetch: RetrievalPrefetch, token: str, state: dict[str, bool],
) -> None:
    """Forward Qwen partial/final events while browser audio is still arriving."""
    speech_started = False
    previous_preview = ""
    postprocess = getattr(service, "postprocess_asr_text", lambda value: str(value or "").strip())
    async for event in transport.events():
        event_type = str(getattr(event, "type", ""))
        raw_text = postprocess(getattr(event, "text", ""))
        text = raw_text
        if wakeup_word:
            text = _remove_wakeup_prefix(text, wakeup_word)
        if event_type == "partial":
            if not text:
                continue
            if not speech_started:
                speech_started = True
                await websocket.send_json({"type": "speech_start", "reason": "server_vad"})
            confirmed = postprocess(getattr(event, "confirmed_text", ""))
            if wakeup_word:
                confirmed = _remove_wakeup_prefix(confirmed, wakeup_word)
            # Short ASR turns may contain only stash until completion. Prefetch
            # a growing preview after a stable prefix; it remains speculative
            # and cannot be consumed unless final text and index revision match.
            stable_prefix = os.path.commonprefix([previous_preview, text])
            if len(confirmed) >= 2 or len(stable_prefix) >= 4:
                prefetch.offer(text)
            previous_preview = text
            await websocket.send_json(
                {
                    "type": "partial",
                    "text": text,
                    "provider": "dashscope_realtime",
                    "confirmed_text": confirmed,
                    "stash": getattr(event, "stash", ""),
                    "item_id": getattr(event, "item_id", None),
                }
            )
            continue
        if event_type != "completed":
            continue
        if not text:
            # A leaked wake-only turn must never end the question session. This
            # is defensive—the wake path normally excludes the KWS trigger
            # frame—but keeps listening if a provider still returns the keyword.
            if wakeup_word and raw_text:
                speech_started = False
                continue
            await websocket.send_json({"type": "no_speech", "reason": "empty_transcript"})
            return
        if not speech_started:
            await websocket.send_json({"type": "speech_start", "reason": "server_vad"})
        await websocket.send_json({"type": "speech_end", "reason": "server_vad"})
        await websocket.send_json(
            {
                "type": "final",
                "text": text,
                "reason": "server_vad",
                "provider": "dashscope_realtime",
                "retrieval_token": token,
                "item_id": getattr(event, "item_id", None),
            }
        )
        state["final"] = True
        return
    await websocket.send_json({"type": "no_speech", "reason": "session_finished"})


async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


async def _run_local_asr_stream(
    websocket: WebSocket,
    service: VoiceService,
    config: VadConfig,
    *,
    initial_audio: tuple[bytes, ...] = (),
    incoming_task: asyncio.Task[Any] | None = None,
    fallback_reason: str | None = None,
    wakeup_word: str | None = None,
    finish_after_initial: bool = False,
) -> None:
    """Continue one utterance with local FSMN-VAD and Whisper."""
    if fallback_reason:
        await websocket.send_json(
            {
                "type": "asr_fallback",
                "provider": "local_whisper",
                "reason": fallback_reason,
            }
        )
    try:
        session = vad_engine_factory().create_session(config)
    except Exception as exc:
        raise VoiceDependencyError(f"本地 VAD 不可用：{exc}") from exc

    speech_seen = False

    async def feed(audio_chunk: bytes, *, is_final: bool = False) -> bool:
        nonlocal speech_seen
        try:
            events = session.feed_pcm16(audio_chunk, is_final=is_final)
        except (ValueError, VoiceInputError):
            raise
        except Exception as exc:
            raise RuntimeError(f"本地 VAD 处理失败：{exc}") from exc
        completed, event_speech_seen = await _send_vad_events(
            websocket,
            events,
            service=service,
            provider="local_whisper",
            wakeup_word=wakeup_word,
        )
        speech_seen = speech_seen or event_speech_seen
        return completed

    try:
        for audio_chunk in initial_audio:
            if audio_chunk and await feed(audio_chunk):
                return
        if finish_after_initial:
            if not await feed(b"", is_final=True):
                await websocket.send_json(
                    {
                        "type": "no_speech",
                        "reason": "manual_stop" if speech_seen else "initial_silence",
                    }
                )
            return

        while True:
            if incoming_task is None:
                incoming_task = asyncio.create_task(websocket.receive())
            message = await incoming_task
            incoming_task = None
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            audio_chunk = message.get("bytes")
            is_final = False
            if audio_chunk is None:
                try:
                    command = json.loads(message.get("text") or "{}")
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "detail": "控制消息必须为 JSON"}
                    )
                    continue
                command_type = command.get("type")
                if command_type == "stop":
                    audio_chunk = b""
                    is_final = True
                elif command_type in {"pause", "resume"}:
                    continue
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": "本地 ASR 仅支持 pause、resume 或 stop 控制消息",
                        }
                    )
                    continue
            elif len(audio_chunk) % 2:
                await websocket.send_json(
                    {"type": "error", "detail": "PCM 数据长度必须为偶数"}
                )
                continue

            if await feed(bytes(audio_chunk), is_final=is_final):
                return
            if is_final:
                await websocket.send_json(
                    {
                        "type": "no_speech",
                        "reason": "manual_stop" if speech_seen else "initial_silence",
                    }
                )
                return
    finally:
        await _cancel_task(incoming_task)


async def _run_realtime_asr_stream(
    websocket: WebSocket,
    transport: Any,
    service: VoiceService,
    wakeup_word: str | None = None,
    *,
    initial_audio: tuple[bytes, ...] = (),
    incoming_task: asyncio.Task[Any] | None = None,
    fallback_config: VadConfig | None = None,
) -> None:
    """Pump cloud ASR, retaining PCM so local Whisper can take over."""
    captured_audio = [chunk for chunk in initial_audio if chunk]
    receiver_task: asyncio.Task[Any] | None = None
    finish_after_initial = False
    try:
        for audio_chunk in captured_audio:
            await transport.append_audio(audio_chunk)
        receiver_task = asyncio.create_task(
            _forward_realtime_asr_events(websocket, transport, service, wakeup_word)
        )
        while True:
            if incoming_task is None:
                incoming_task = asyncio.create_task(websocket.receive())
            done, _pending = await asyncio.wait(
                {incoming_task, receiver_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if receiver_task in done:
                await receiver_task
                return

            message = incoming_task.result()
            incoming_task = None
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            audio_chunk = message.get("bytes")
            if audio_chunk is not None:
                if len(audio_chunk) % 2:
                    await websocket.send_json({"type": "error", "detail": "PCM 数据长度必须为偶数"})
                    continue
                if audio_chunk:
                    chunk = bytes(audio_chunk)
                    captured_audio.append(chunk)
                    await transport.append_audio(chunk)
                continue

            raw_text = message.get("text")
            try:
                command = json.loads(raw_text or "{}")
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "控制消息必须为 JSON"})
                continue
            if command.get("type") != "stop":
                await websocket.send_json({"type": "error", "detail": "仅支持 stop 控制消息"})
                continue
            finish_after_initial = True
            await transport.finish()
            try:
                await asyncio.wait_for(receiver_task, timeout=15.0)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("等待 DashScope ASR 最终结果超时") from exc
            return
    except WebSocketDisconnect:
        raise
    except Exception as exc:
        if fallback_config is None:
            raise
        note_failure = getattr(service, "note_realtime_asr_failure", None)
        if callable(note_failure):
            note_failure(exc)
        LOGGER.warning(
            "DashScope realtime ASR interrupted; continuing locally: %s",
            exc,
        )
        await _run_local_asr_stream(
            websocket,
            service,
            fallback_config,
            initial_audio=tuple(captured_audio),
            incoming_task=incoming_task,
            fallback_reason="cloud_disconnected",
            wakeup_word=wakeup_word,
            finish_after_initial=finish_after_initial,
        )
        incoming_task = None
    finally:
        await _cancel_task(incoming_task)
        await _cancel_task(receiver_task)


@app.websocket("/api/voice/stt/stream")
async def voice_speech_to_text_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    speech_seen = False
    try:
        settings = settings_store.load()
        if not bool(getattr(settings, "vad_enabled", True)):
            await websocket.send_json({"type": "error", "detail": "VAD 已在配置中关闭"})
            await websocket.close(code=1008)
            return
        try:
            start_message = await websocket.receive_json()
            config = _validate_vad_start(
                start_message,
                int(getattr(settings, "vad_end_silence_ms", 400)),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1008)
            return

        await websocket.send_json(
            {
                "type": "ready",
                "frame_ms": config.frame_ms,
                "sample_rate": config.sample_rate,
                "end_silence_ms": config.end_silence_ms,
            }
        )

        service = voice_service_factory()
        open_realtime_asr = getattr(service, "open_realtime_asr", None)
        transport = None
        if callable(open_realtime_asr):
            transport = await open_realtime_asr(end_silence_ms=config.end_silence_ms)
        if transport is not None:
            try:
                await _run_realtime_asr_stream(
                    websocket,
                    transport,
                    service,
                    fallback_config=config,
                )
                with suppress(Exception):
                    await websocket.close(code=1000)
            except WebSocketDisconnect:
                return
            except Exception as exc:
                with suppress(Exception):
                    await websocket.send_json(
                        {"type": "error", "detail": f"实时语音识别失败：{exc}"}
                    )
                    await websocket.close(code=1011)
            finally:
                release_realtime_asr = getattr(service, "release_realtime_asr", None)
                if callable(release_realtime_asr):
                    with suppress(Exception):
                        await release_realtime_asr(transport)
                else:
                    with suppress(Exception):
                        await transport.close()
            return

        try:
            session = vad_engine_factory().create_session(config)
        except Exception as exc:
            await websocket.send_json({"type": "error", "detail": f"VAD 不可用：{exc}"})
            await websocket.close(code=1011)
            return

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            audio_chunk = message.get("bytes")
            is_final = False
            if audio_chunk is None:
                raw_text = message.get("text")
                try:
                    command = json.loads(raw_text or "{}")
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "控制消息必须为 JSON"})
                    continue
                if command.get("type") != "stop":
                    await websocket.send_json({"type": "error", "detail": "仅支持 stop 控制消息"})
                    continue
                audio_chunk = b""
                is_final = True
            elif len(audio_chunk) % 2:
                await websocket.send_json({"type": "error", "detail": "PCM 数据长度必须为偶数"})
                continue

            try:
                events = session.feed_pcm16(audio_chunk, is_final=is_final)
            except (ValueError, VoiceInputError) as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
                await websocket.close(code=1008)
                return
            except Exception as exc:
                await websocket.send_json({"type": "error", "detail": f"VAD 处理失败：{exc}"})
                await websocket.close(code=1011)
                return

            completed, event_speech_seen = await _send_vad_events(websocket, events)
            speech_seen = speech_seen or event_speech_seen
            if completed:
                await websocket.close(code=1000)
                return
            if is_final:
                await websocket.send_json(
                    {"type": "no_speech", "reason": "manual_stop" if speech_seen else "initial_silence"}
                )
                await websocket.close(code=1000)
                return
    except WebSocketDisconnect:
        return


async def _connect_realtime_asr_while_buffering(
    websocket: WebSocket,
    open_realtime_asr: Any,
    *,
    end_silence_ms: int,
) -> tuple[Any | None, tuple[bytes, ...], asyncio.Task[Any] | None]:
    """Open cloud ASR while retaining only audio captured after wake detection.

    The chunk that triggered KWS is deliberately excluded. Sending any wake-word
    tail to server VAD can create a completed turn after 400 ms of silence and
    discard the user's actual question when they pause after the wake phrase.
    """
    connect_task = asyncio.create_task(
        open_realtime_asr(end_silence_ms=end_silence_ms)
    )
    incoming_task: asyncio.Task[Any] | None = asyncio.create_task(websocket.receive())
    buffered_audio: list[bytes] = []
    buffered_bytes = 0
    try:
        while True:
            done, _pending = await asyncio.wait(
                {connect_task, incoming_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if incoming_task in done:
                message = incoming_task.result()
                incoming_task = None
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()
                audio_chunk = message.get("bytes")
                if audio_chunk is not None:
                    if len(audio_chunk) % 2:
                        await websocket.send_json(
                            {"type": "error", "detail": "PCM 数据长度必须为偶数"}
                        )
                    elif audio_chunk:
                        chunk = bytes(audio_chunk)
                        buffered_bytes += len(chunk)
                        if buffered_bytes > WAKEUP_ASR_CONNECT_BUFFER_BYTES:
                            raise RuntimeError("Qwen ASR 建连过慢，问题音频缓冲已满")
                        buffered_audio.append(chunk)
                else:
                    try:
                        command = json.loads(message.get("text") or "{}")
                    except json.JSONDecodeError:
                        await websocket.send_json(
                            {"type": "error", "detail": "控制消息必须为 JSON"}
                        )
                    else:
                        command_type = command.get("type")
                        if command_type == "stop":
                            if connect_task.done():
                                candidate = await connect_task
                                close = getattr(candidate, "close", None)
                                if callable(close):
                                    with suppress(Exception):
                                        await close()
                            else:
                                await _cancel_task(connect_task)
                            return None, (), None
                        if command_type not in {"pause", "resume"}:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "detail": "ASR 建连时仅支持 pause、resume 或 stop",
                                }
                            )
                incoming_task = asyncio.create_task(websocket.receive())

            if connect_task in done:
                try:
                    transport = await connect_task
                except Exception as exc:
                    LOGGER.warning(
                        "DashScope realtime ASR connection task failed; using local fallback: %s",
                        exc,
                    )
                    transport = None
                return transport, tuple(buffered_audio), incoming_task
    except BaseException:
        await _cancel_task(incoming_task)
        await _cancel_task(connect_task)
        raise


def _create_wakeup_session(engine: Any, wakeup_word: str) -> Any:
    create_session = getattr(engine, "create_session", None)
    if not callable(create_session):
        raise VoiceDependencyError("离线唤醒引擎没有 create_session 接口")
    try:
        return create_session()
    except TypeError:
        # Keep factory injection compatible with simple/open-vocabulary test doubles.
        return create_session([wakeup_word])


def _accept_wakeup_pcm(session: Any, audio_chunk: bytes) -> Any | None:
    accept = getattr(session, "accept_pcm16", None)
    if callable(accept):
        return accept(audio_chunk)
    feed = getattr(session, "feed_pcm16", None)
    if callable(feed):
        return feed(audio_chunk)
    raise VoiceDependencyError("离线唤醒会话没有 accept_pcm16 接口")


def _wakeup_event_payload(
    event: Any,
    configured_word: str,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    keyword = str(getattr(event, "keyword", "") or configured_word)
    is_interrupt = reason == "interrupt"
    payload: dict[str, Any] = {
        "type": "wake_detected",
        "keyword": keyword,
        "prompt_text": (
            INTERRUPT_ACKNOWLEDGEMENT_TEXT
            if is_interrupt
            else WAKEUP_ACKNOWLEDGEMENT_TEXT
        ),
        "prompt_audio_url": (
            INTERRUPT_ACKNOWLEDGEMENT_URL
            if is_interrupt
            else WAKEUP_ACKNOWLEDGEMENT_URL
        ),
    }
    if reason:
        payload["reason"] = reason
    for key in ("score", "timestamp_ms"):
        value = getattr(event, key, None)
        if value is not None:
            payload[key] = value
    return payload


async def _wait_for_wakeup_prompt_finished(websocket: WebSocket) -> bool:
    """Discard microphone audio until the browser finishes the wake response."""
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return False
        if message.get("bytes") is not None:
            continue
        try:
            command = json.loads(message.get("text") or "{}")
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "detail": "控制消息必须为 JSON"})
            continue
        command_type = command.get("type")
        if command_type == "prompt_finished":
            return True
        if command_type == "stop":
            return False
        if command_type == "pause":
            continue
        await websocket.send_json(
            {"type": "error", "detail": "播放唤醒提示音时仅支持 pause、prompt_finished 或 stop"}
        )


def _rearm_wakeup_session(engine: Any, session: Any, wakeup_word: str) -> Any:
    """Reset a detected KWS stream, replacing it when the backend cannot reset."""
    reset = getattr(session, "reset", None)
    if callable(reset):
        try:
            reset()
            return session
        except Exception:
            LOGGER.debug("Wake-word session reset failed; creating a fresh session", exc_info=True)
    close = getattr(session, "close", None)
    if callable(close):
        with suppress(Exception):
            close()
    return _create_wakeup_session(engine, wakeup_word)


async def _wait_for_voice_next_action(
    websocket: WebSocket,
    *,
    wake_session: Any | None = None,
    wake_engine: Any | None = None,
    wakeup_word: str = "",
    wake_session_holder: list[Any] | None = None,
) -> str | None:
    """Monitor KWS while answer/TTS is active and wait for the next ASR action.

    PCM received in this phase is deliberately never forwarded to ASR.  It is
    only inspected by the independent wake-word session so a spoken wake word
    can cancel the previous turn immediately.
    """
    if wake_session_holder is None:
        wake_session_holder = [wake_session]
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return None
        audio_chunk = message.get("bytes")
        if audio_chunk is not None:
            if len(audio_chunk) % 2:
                await websocket.send_json({"type": "error", "detail": "PCM 数据长度必须为偶数"})
                continue
            if not audio_chunk or wake_engine is None or wake_session_holder[0] is None:
                continue
            try:
                wake_event = _accept_wakeup_pcm(wake_session_holder[0], bytes(audio_chunk))
            except Exception as exc:
                # A transient backend failure must not tear down the active
                # answer session; continue dropping this frame and keep the
                # control channel available for `continue`/`resume`.
                LOGGER.warning("Wake-word monitor rejected PCM: %s", exc)
                continue
            if wake_event is None or wake_event is False:
                continue
            await websocket.send_json(
                _wakeup_event_payload(
                    wake_event,
                    wakeup_word,
                    reason="interrupt",
                )
            )
            # Do not leave a latched detector active while the acknowledgement
            # is playing or during the next listening turn.
            wake_session_holder[0] = _rearm_wakeup_session(
                wake_engine,
                wake_session_holder[0],
                wakeup_word,
            )
            if not await _wait_for_wakeup_prompt_finished(websocket):
                return None
            return "interrupt"
        try:
            command = json.loads(message.get("text") or "{}")
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "detail": "控制消息必须为 JSON"})
            continue
        command_type = command.get("type")
        if command_type in {"continue", "resume"}:
            return str(command_type)
        if command_type == "stop":
            return None
        if command_type != "pause":
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": "等待下一轮时仅支持 pause、continue、resume 或 stop",
                }
            )


@app.websocket("/api/voice/session/stream")
async def voice_wakeup_session_stream(websocket: WebSocket) -> None:
    """Persistent local-wakeup -> cloud-ASR session.

    Audio is never opened or forwarded to DashScope while the connection is armed.
    """
    await websocket.accept()
    transport: Any | None = None
    wake_session: Any | None = None
    try:
        settings = settings_store.load()
        if not bool(getattr(settings, "wakeup_enabled", True)):
            await websocket.send_json({"type": "error", "detail": "离线唤醒已在配置中关闭"})
            await websocket.close(code=1008)
            return
        if not bool(getattr(settings, "vad_enabled", True)):
            await websocket.send_json({"type": "error", "detail": "VAD 已在配置中关闭"})
            await websocket.close(code=1008)
            return
        try:
            start_message = await websocket.receive_json()
            config = _validate_vad_start(
                start_message,
                int(getattr(settings, "vad_end_silence_ms", 400)),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1008)
            return

        wakeup_word = str(getattr(settings, "wakeup_word", "小绿小绿")).strip()
        cooldown_ms = int(getattr(settings, "wakeup_cooldown_ms", 800))
        try:
            engine = wakeword_engine_factory()
            wake_session = _create_wakeup_session(engine, wakeup_word)
        except Exception as exc:
            await websocket.send_json({"type": "error", "detail": f"离线唤醒不可用：{exc}"})
            await websocket.close(code=1011)
            return

        await websocket.send_json(
            {
                "type": "armed",
                "keyword": wakeup_word,
                "sample_rate": config.sample_rate,
                "end_silence_ms": config.end_silence_ms,
                "cooldown_ms": cooldown_ms,
            }
        )

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            audio_chunk = message.get("bytes")
            if audio_chunk is None:
                try:
                    command = json.loads(message.get("text") or "{}")
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "控制消息必须为 JSON"})
                    continue
                command_type = command.get("type")
                if command_type == "stop":
                    await websocket.close(code=1000)
                    return
                if command_type == "resume":
                    # A typed/manual answer can finish while the persistent
                    # wake socket is still in its outer armed phase.  Re-arm
                    # the detector explicitly and acknowledge the transition;
                    # otherwise the browser would keep its AudioWorklet
                    # suspended waiting for an `armed` event that never comes.
                    wake_session = _rearm_wakeup_session(
                        engine,
                        wake_session,
                        wakeup_word,
                    )
                    await websocket.send_json(
                        {
                            "type": "armed",
                            "keyword": wakeup_word,
                            "sample_rate": config.sample_rate,
                            "end_silence_ms": config.end_silence_ms,
                            "cooldown_ms": cooldown_ms,
                        }
                    )
                    continue
                if command_type == "pause":
                    continue
                await websocket.send_json(
                    {"type": "error", "detail": "待唤醒时仅支持 pause、resume 或 stop"}
                )
                continue
            if len(audio_chunk) % 2:
                await websocket.send_json({"type": "error", "detail": "PCM 数据长度必须为偶数"})
                continue
            if not audio_chunk:
                continue

            chunk = bytes(audio_chunk)
            try:
                wake_event = _accept_wakeup_pcm(wake_session, chunk)
            except Exception as exc:
                await websocket.send_json({"type": "error", "detail": f"离线唤醒处理失败：{exc}"})
                await websocket.close(code=1011)
                return
            if wake_event is None or wake_event is False:
                continue

            await websocket.send_json(_wakeup_event_payload(wake_event, wakeup_word))
            if not await _wait_for_wakeup_prompt_finished(websocket):
                with suppress(Exception):
                    await websocket.close(code=1000)
                return
            # The detector that produced the initial wake event is latched on
            # most backends.  Rearm it before opening ASR so it can monitor the
            # later answer/TTS phase without competing with question VAD.
            wake_session = _rearm_wakeup_session(engine, wake_session, wakeup_word)
            service = voice_service_factory()
            open_realtime_asr = getattr(service, "open_realtime_asr", None)
            if not callable(open_realtime_asr):
                await websocket.send_json({"type": "error", "detail": "Qwen Realtime ASR 不可用"})
                await websocket.close(code=1011)
                return
            first_question_after_wakeup = True
            while True:
                pending_input: asyncio.Task[Any] | None = None
                try:
                    buffered_audio: tuple[bytes, ...]
                    transport, buffered_audio, pending_input = (
                        await _connect_realtime_asr_while_buffering(
                            websocket,
                            open_realtime_asr,
                            end_silence_ms=config.end_silence_ms,
                        )
                    )
                    if transport is None and pending_input is None:
                        with suppress(Exception):
                            await websocket.close(code=1000)
                        return
                    await websocket.send_json(
                        {
                            "type": "listening",
                            "keyword": wakeup_word,
                            "end_silence_ms": config.end_silence_ms,
                        }
                    )
                    if transport is None:
                        await _run_local_asr_stream(
                            websocket,
                            service,
                            config,
                            initial_audio=buffered_audio,
                            incoming_task=pending_input,
                            fallback_reason="cloud_connect_failed",
                            wakeup_word=(
                                wakeup_word if first_question_after_wakeup else None
                            ),
                        )
                    else:
                        await _run_realtime_asr_stream(
                            websocket,
                            transport,
                            service,
                            wakeup_word=wakeup_word if first_question_after_wakeup else None,
                            initial_audio=buffered_audio,
                            incoming_task=pending_input,
                            fallback_config=config,
                        )
                    pending_input = None
                except WebSocketDisconnect:
                    return
                except Exception as exc:
                    with suppress(Exception):
                        await websocket.send_json(
                            {"type": "error", "detail": f"实时语音识别失败：{exc}"}
                        )
                        await websocket.close(code=1011)
                    return
                finally:
                    await _cancel_task(pending_input)
                    if transport is not None:
                        release_realtime_asr = getattr(service, "release_realtime_asr", None)
                        if callable(release_realtime_asr):
                            with suppress(Exception):
                                await release_realtime_asr(transport)
                        else:
                            with suppress(Exception):
                                await transport.close()
                        transport = None

                session_holder = [wake_session]
                next_action = await _wait_for_voice_next_action(
                    websocket,
                    wake_session=wake_session,
                    wake_engine=engine,
                    wakeup_word=wakeup_word,
                    wake_session_holder=session_holder,
                )
                wake_session = session_holder[0]
                if next_action is None:
                    with suppress(Exception):
                        await websocket.close(code=1000)
                    return
                if next_action == "continue":
                    # Discard any detector state accumulated while the
                    # previous answer was playing before the next ASR turn.
                    wake_session = _rearm_wakeup_session(
                        engine,
                        wake_session,
                        wakeup_word,
                    )
                    first_question_after_wakeup = False
                    continue
                if next_action == "interrupt":
                    # The interrupt path already rearms immediately after the
                    # hit, before the acknowledgement is played.
                    first_question_after_wakeup = False
                    continue
                break

            # Returning to armed mode must also start with a clean detector;
            # otherwise speaker audio from the completed turn can remain in a
            # streaming keyword session.
            wake_session = _rearm_wakeup_session(engine, wake_session, wakeup_word)
            await websocket.send_json(
                {
                    "type": "armed",
                    "keyword": wakeup_word,
                    "sample_rate": config.sample_rate,
                    "end_silence_ms": config.end_silence_ms,
                    "cooldown_ms": cooldown_ms,
                }
            )
    except WebSocketDisconnect:
        return
    finally:
        if transport is not None:
            with suppress(Exception):
                await transport.close()
        close_session = getattr(wake_session, "close", None)
        if callable(close_session):
            with suppress(Exception):
                close_session()


@app.post("/api/voice/tts")
def voice_text_to_speech(payload: TTSPayload) -> JSONResponse:
    try:
        audio_bytes = voice_service_factory().text_to_speech(payload.text)
    except VoiceInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VoiceDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse({"audio": base64.b64encode(audio_bytes).decode("ascii"), "format": "mp3"})


@app.post("/api/voice/tts/stream")
async def voice_text_to_speech_stream(payload: TTSPayload) -> StreamingResponse:
    """Stream MP3 data after eagerly surfacing setup/input errors as HTTP errors."""
    try:
        audio_iterator = voice_service_factory().stream_text_to_speech(payload.text).__aiter__()
        first_chunk = await anext(audio_iterator)
        while not first_chunk:
            first_chunk = await anext(audio_iterator)
    except VoiceInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VoiceDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StopAsyncIteration as exc:
        raise HTTPException(status_code=503, detail="TTS provider returned no audio") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS provider failed: {exc}") from exc

    if not isinstance(first_chunk, bytes):
        close_iterator = getattr(audio_iterator, "aclose", None)
        if close_iterator is not None:
            await close_iterator()
        raise HTTPException(status_code=503, detail="TTS provider returned an invalid audio chunk")

    async def event_stream():
        try:
            yield first_chunk
            async for chunk in audio_iterator:
                if chunk:
                    yield chunk
        finally:
            close_iterator = getattr(audio_iterator, "aclose", None)
            if close_iterator is not None:
                await close_iterator()

    return StreamingResponse(
        event_stream(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/image-index/jobs/{job_id}")
def index_parse_job_images(job_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Parse job not found")
        if job.get("status") not in {"completed", "completed_with_errors"}:
            raise HTTPException(status_code=409, detail="Parse job is not completed")
        source_results = [
            result
            for result in job.get("results", [])
            if result.get("status") == "success" and result.get("document")
        ]

    if not source_results:
        raise HTTPException(status_code=400, detail="No successfully parsed documents to image-index")
    if not settings_store.load().has_api_key:
        raise HTTPException(status_code=400, detail="请先配置百炼 API Key")

    image_index_job_id = uuid.uuid4().hex
    with _image_index_jobs_lock:
        _image_index_jobs[image_index_job_id] = {
            "image_index_job_id": image_index_job_id,
            "parse_job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": len(source_results),
            "results": [],
            "errors": [],
        }
    background_tasks.add_task(_run_image_index_job, image_index_job_id, source_results)
    return JSONResponse({"image_index_job_id": image_index_job_id, "status": "queued"})


@app.get("/api/image-index/jobs/{image_index_job_id}")
def get_image_index_job(image_index_job_id: str) -> JSONResponse:
    with _image_index_jobs_lock:
        job = _image_index_jobs.get(image_index_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Image index job not found")
        return JSONResponse(job)


@app.get("/api/image-index/stats")
def get_image_index_stats() -> JSONResponse:
    return JSONResponse(asdict(image_indexing_service_factory().stats()))


@app.post("/api/image-search")
async def search_images(file: UploadFile = File(...), top_k: int = 5) -> JSONResponse:
    if not settings_store.load().has_api_key:
        raise HTTPException(status_code=400, detail="请先配置百炼 API Key")
    query_dir = DATA_DIR / "image_queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(file.filename or "query.png")
    query_path = query_dir / f"{uuid.uuid4().hex}-{name}"
    with query_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        results = image_indexing_service_factory().search_by_image(query_path, top_k=max(1, min(top_k, 20)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = []
    for result in results:
        item = asdict(result)
        item["image_data_url"] = _image_data_url(Path(result.image_path))
        payload.append(item)
    return JSONResponse({"results": payload})


def _run_parse_job(job_id: str, paths: list[Path], chunk_size: int, chunk_overlap: int) -> None:
    service = ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=CACHE_DIR,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_languages="eng+chi_sim",
    )
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        try:
            document = service.parse_document(path)
            indexing_service_factory().manifest.record_parsed_document(document)
            results.append(
                {
                    "source": str(path),
                    "status": "success",
                    "document": asdict(document),
                }
            )
        except Exception as exc:
            errors.append({"source": str(path), "error": str(exc)})
            _record_parse_failure(path, str(exc))

        with _jobs_lock:
            _jobs[job_id]["progress"] = index
            _jobs[job_id]["results"] = results
            _jobs[job_id]["errors"] = errors

    with _jobs_lock:
        _jobs[job_id]["status"] = "completed" if not errors else "completed_with_errors"


def _run_index_job(index_job_id: str, source_results: list[dict[str, Any]]) -> None:
    service = indexing_service_factory()
    with _index_jobs_lock:
        _index_jobs[index_job_id]["status"] = "running"

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, result in enumerate(source_results, start=1):
        document_payload = result.get("document") or {}
        try:
            document = parsed_document_from_dict(document_payload)
            indexed = service.index_document(document)
            results.append(asdict(indexed))
        except MissingApiKeyError as exc:
            errors.append({"source": str(result.get("source", "")), "error": str(exc)})
            _record_index_failure(document_payload, str(exc))
        except Exception as exc:
            errors.append({"source": str(result.get("source", "")), "error": str(exc)})
            _record_index_failure(document_payload, str(exc))

        with _index_jobs_lock:
            _index_jobs[index_job_id]["progress"] = index
            _index_jobs[index_job_id]["results"] = results
            _index_jobs[index_job_id]["errors"] = errors

    with _index_jobs_lock:
        _index_jobs[index_job_id]["status"] = "completed" if not errors else "completed_with_errors"


def _run_reindex_document_job(index_job_id: str, document_id: str, source_path: Path) -> None:
    settings = settings_store.load()
    parse_service = ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=CACHE_DIR,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        ocr_languages="eng+chi_sim",
    )
    text_service = indexing_service_factory()
    image_service = image_indexing_service_factory()
    with _index_jobs_lock:
        _index_jobs[index_job_id]["status"] = "running"

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        document = parse_service.parse_document(source_path)
        new_document_id = document_id_for_document(document)
        text_service.delete_document(document_id, remove_manifest=False)
        image_service.delete_document_images(document_id)
        indexed = text_service.index_document(document)
        payload: dict[str, Any] = {"text_index": asdict(indexed)}
        try:
            image_indexed = image_service.index_document_images(document)
            payload["image_index"] = asdict(image_indexed)
        except ValueError as exc:
            payload["image_index"] = {"status": "skipped", "message": str(exc)}
        results.append(payload)
        if new_document_id != document_id:
            text_service.manifest.delete_document(document_id)
    except MissingApiKeyError as exc:
        errors.append({"source": str(source_path), "error": str(exc)})
        text_service.manifest.update_index_error(document_id, str(exc))
    except Exception as exc:
        errors.append({"source": str(source_path), "error": str(exc)})
        text_service.manifest.update_index_error(document_id, str(exc))

    with _index_jobs_lock:
        _index_jobs[index_job_id]["progress"] = 1
        _index_jobs[index_job_id]["results"] = results
        _index_jobs[index_job_id]["errors"] = errors
        _index_jobs[index_job_id]["status"] = "completed" if not errors else "completed_with_errors"


def _run_image_index_job(image_index_job_id: str, source_results: list[dict[str, Any]]) -> None:
    service = image_indexing_service_factory()
    with _image_index_jobs_lock:
        _image_index_jobs[image_index_job_id]["status"] = "running"

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, result in enumerate(source_results, start=1):
        document_payload = result.get("document") or {}
        try:
            document = parsed_document_from_dict(document_payload)
            indexed = service.index_document_images(document)
            results.append(asdict(indexed))
        except MissingApiKeyError as exc:
            errors.append({"source": str(result.get("source", "")), "error": str(exc)})
            _record_image_index_failure(document_payload, str(exc))
        except Exception as exc:
            errors.append({"source": str(result.get("source", "")), "error": str(exc)})
            _record_image_index_failure(document_payload, str(exc))

        with _image_index_jobs_lock:
            _image_index_jobs[image_index_job_id]["progress"] = index
            _image_index_jobs[image_index_job_id]["results"] = results
            _image_index_jobs[image_index_job_id]["errors"] = errors

    with _image_index_jobs_lock:
        _image_index_jobs[image_index_job_id]["status"] = "completed" if not errors else "completed_with_errors"


def _record_parse_failure(path: Path, error: str) -> None:
    try:
        digest = file_sha256(path)
    except Exception:
        digest = ""
    if not digest:
        digest = uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex
    document_id = document_id_for_file(path, digest)
    indexing_service_factory().manifest.record_parse_failure(
        document_id=document_id,
        source=str(path),
        source_name=path.name,
        file_sha256=digest,
        error=error,
    )


def _record_index_failure(document_payload: dict[str, Any], error: str) -> None:
    try:
        document = parsed_document_from_dict(document_payload)
        document_id = document_id_for_document(document)
    except Exception:
        return
    indexing_service_factory().manifest.update_index_error(document_id, error)


def _record_image_index_failure(document_payload: dict[str, Any], error: str) -> None:
    try:
        document = parsed_document_from_dict(document_payload)
        document_id = document_id_for_document(document)
    except Exception:
        return
    indexing_service_factory().manifest.update_image_index(
        document_id,
        image_count=0,
        image_vector_count=0,
        status="failed",
        error=error,
    )


def _delete_document_artifacts(record: Any) -> dict[str, Any]:
    deleted_source = False
    deleted_cache = False
    source = Path(str(record.source))
    try:
        upload_root = UPLOAD_DIR.resolve()
        resolved_source = source.resolve()
        if source.exists() and _is_relative_to(resolved_source, upload_root):
            source.unlink()
            deleted_source = True
            try:
                if resolved_source.parent != upload_root and not any(resolved_source.parent.iterdir()):
                    resolved_source.parent.rmdir()
            except OSError:
                pass
    except OSError:
        deleted_source = False

    cache_key = getattr(record, "cache_key", None)
    if cache_key:
        cache_path = CACHE_DIR / f"{cache_key}.json"
        try:
            if cache_path.exists():
                cache_path.unlink()
                deleted_cache = True
        except OSError:
            deleted_cache = False
    return {"source_file": deleted_source, "parser_cache": deleted_cache}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-()[] " else "_" for ch in name).strip()
    return cleaned or "upload.bin"


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _image_data_url(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>小绿 · 绿色江河志愿讲解员</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #131a2b;
      --panel-2: #182235;
      --text: #edf2ff;
      --muted: #aeb9cf;
      --line: #2b3953;
      --green: #4dd6a5;
      --cyan: #56c7f2;
      --pink: #ff7a9f;
      --amber: #f0bd4f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .shell { max-width: 1420px; margin: 0 auto; padding: 24px; }
    .app-shell { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 16px;
      padding: 14px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(8, 13, 26, .92);
      backdrop-filter: blur(16px);
    }
    .brand {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--text);
      font-weight: 800;
      letter-spacing: 0;
    }
    .brand-mark {
      width: 26px;
      height: 26px;
      border-radius: 999px;
      border: 1px solid var(--line);
      display: grid;
      place-items: center;
      color: var(--green);
      background: #10182a;
      font-size: 13px;
    }
    .app-nav {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #10182a;
    }
    .nav-btn {
      border-radius: 999px;
      background: transparent;
      color: var(--muted);
      padding: 8px 14px;
      font-weight: 700;
    }
    .nav-btn.active {
      background: var(--text);
      color: #08101e;
    }
    .topbar .status { justify-self: end; }
    .app-main { min-height: 0; }
    .app-page { display: none; padding: 24px; }
    .app-page.active { display: block; }
    .settings-page {
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(320px, 460px) minmax(0, 1fr);
      gap: 16px;
    }
    .app-page.chat-page { display: none; }
    .app-page.chat-page.active {
      height: calc(100vh - 62px);
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 28px 20px;
      overflow: hidden;
    }
    .chat-workspace {
      width: min(1340px, 100%);
      height: 100%;
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 920px);
      justify-content: center;
      gap: 18px;
    }
    .chat-shell {
      width: 100%;
      height: 100%;
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto auto auto;
      gap: 14px;
    }
    .llm-trace-panel {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f1728;
    }
    .llm-trace-header {
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
    }
    .llm-trace-header h2 {
      margin: 0;
      font-size: 14px;
      letter-spacing: 0;
    }
    .local-badge {
      flex: 0 0 auto;
      border: 1px solid var(--green);
      border-radius: 999px;
      padding: 2px 7px;
      color: var(--green);
      font: 10px/1.4 Consolas, "Cascadia Mono", monospace;
    }
    .llm-trace-status {
      min-height: 38px;
      padding: 9px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }
    .llm-trace-feed {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      overscroll-behavior: contain;
      scrollbar-gutter: stable;
    }
    .llm-trace-empty {
      margin: 0;
      padding: 18px 14px;
      color: var(--muted);
      font-size: 12px;
    }
    .llm-trace-entry {
      border-bottom: 1px solid var(--line);
    }
    .llm-trace-entry > summary {
      cursor: pointer;
      padding: 11px 14px;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }
    .llm-trace-entry > summary span {
      display: block;
      margin: 3px 0 0 18px;
      color: var(--muted);
      font: 11px/1.4 Consolas, "Cascadia Mono", monospace;
      font-weight: 400;
    }
    .llm-message-list {
      border-top: 1px solid var(--line);
    }
    .llm-message {
      min-width: 0;
      padding: 10px 12px 12px;
      border-bottom: 1px solid rgba(43, 57, 83, .72);
    }
    .llm-message:last-child { border-bottom: 0; }
    .llm-message-role {
      display: inline-flex;
      margin-bottom: 7px;
      font: 10px/1.4 Consolas, "Cascadia Mono", monospace;
      text-transform: uppercase;
    }
    .llm-message.system .llm-message-role { color: var(--amber); }
    .llm-message.user .llm-message-role { color: var(--cyan); }
    .llm-message.assistant .llm-message-role { color: var(--green); }
    .llm-message pre {
      margin: 0;
      padding: 0 0 0 9px;
      border-left: 2px solid var(--line);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      color: #d6deed;
      font: 11px/1.55 Consolas, "Cascadia Mono", monospace;
    }
    .chat-feed {
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      scrollbar-gutter: stable;
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 10px 4px 16px;
    }
    .chat-empty {
      margin: auto;
      text-align: center;
      color: var(--muted);
      padding: 56px 16px;
    }
    .chat-empty h1 {
      font-size: clamp(32px, 5vw, 56px);
      line-height: 1.06;
      margin-bottom: 12px;
    }
    .chat-message {
      width: min(760px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      line-height: 1.7;
      overflow-wrap: anywhere;
    }
    .chat-message.user {
      justify-self: end;
      background: #d7def0;
      color: #07111e;
    }
    .chat-message.assistant {
      justify-self: start;
      background: var(--panel);
      color: var(--text);
    }
    .chat-message .citation-grid { margin-top: 12px; }
    .stream-cursor {
      display: inline-block;
      width: 8px;
      height: 1em;
      margin-left: 2px;
      background: var(--green);
      vertical-align: -2px;
      animation: blink 1s steps(2, start) infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }
    .chat-composer {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto auto;
      align-items: end;
      gap: 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 18px 80px rgba(0, 0, 0, .22);
    }
    .chat-composer textarea {
      min-height: 54px;
      max-height: 180px;
      resize: vertical;
      border: 0;
      outline: none;
      background: transparent;
      color: var(--text);
      padding: 12px;
      font: inherit;
      line-height: 1.5;
    }
    .chat-composer textarea.stt-partial {
      color: var(--cyan);
      font-style: italic;
    }
    .chat-composer button {
      width: 44px;
      height: 44px;
      border-radius: 999px;
      padding: 0;
    }
    .chat-composer button.recording {
      background: var(--pink);
      color: #fff;
      border-color: var(--pink);
    }
    .chat-composer button.enabled {
      background: var(--cyan);
      color: #07111e;
      border-color: var(--cyan);
    }
    .chat-message-tools {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-bottom: 8px;
    }
    .chat-message-tools button {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      padding: 0;
      font-size: 14px;
    }
    .chat-status {
      color: var(--muted);
      text-align: center;
      font-size: 12px;
    }
    .wakeup-bar {
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      gap: 10px 16px;
      color: var(--muted);
      font-size: 12px;
    }
    .wakeup-bar .toggle-row {
      width: auto;
      margin: 0;
      gap: 8px;
    }
    .wakeup-status.armed { color: var(--green); }
    .wakeup-status.listening { color: var(--cyan); }
    .wakeup-status.error { color: var(--pink); }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: 0; }
    .status { color: var(--muted); font-size: 14px; }
    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      min-height: calc(100vh - 100px);
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      font-size: 15px;
      border-bottom: 1px solid var(--line);
    }
    .dropzone {
      margin: 16px;
      min-height: 180px;
      border: 1px dashed #6b7ea4;
      border-radius: 8px;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
      background: #10182a;
      cursor: pointer;
      transition: border-color .15s, background .15s;
    }
    .dropzone.active { border-color: var(--cyan); background: #122238; }
    .dropzone strong { color: var(--text); display: block; margin-bottom: 8px; }
    input[type=file] { display: none; }
    .file-list { padding: 0 16px 16px; display: grid; gap: 8px; }
    .file-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 6px;
      font-size: 13px;
    }
    .file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    button {
      border: 0;
      border-radius: 6px;
      color: #07111e;
      background: var(--green);
      font-weight: 700;
      padding: 10px 14px;
      cursor: pointer;
    }
    button:disabled { opacity: .45; cursor: default; }
    .actions { padding: 0 16px 16px; display: flex; gap: 10px; }
    .secondary { background: #d7def0; }
    .danger { background: var(--pink); color: #180711; }
    .settings-form {
      padding: 16px;
      border-top: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }
    .settings-form label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .settings-form input,
    .settings-form textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f1728;
      color: var(--text);
      padding: 10px 12px;
      outline: none;
    }
    .settings-form textarea {
      min-height: 150px;
      resize: vertical;
      font-family: Consolas, "Cascadia Mono", monospace;
      line-height: 1.55;
    }
    .prompt-catalog {
      display: grid;
      gap: 10px;
    }
    .prompt-entry {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      overflow: hidden;
    }
    .prompt-entry summary {
      cursor: pointer;
      padding: 10px 12px;
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
      list-style-position: inside;
    }
    .prompt-entry summary span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
    }
    .prompt-entry pre {
      margin: 0;
      padding: 12px;
      border-top: 1px solid var(--line);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: var(--muted);
      font: 12px/1.55 Consolas, "Cascadia Mono", monospace;
    }
    .settings-form input:focus,
    .settings-form textarea:focus { border-color: var(--cyan); }
    .settings-status {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      min-height: 38px;
    }
    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 0 16px 16px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }
    .toggle-row input {
      width: 20px;
      height: 20px;
      accent-color: var(--green);
      cursor: pointer;
    }
    .settings-actions {
      padding: 0;
      flex-wrap: wrap;
    }
    .model-line { color: var(--muted); font-size: 12px; }
    .search-results {
      display: grid;
      gap: 8px;
      max-height: 220px;
      overflow: auto;
    }
    .search-hit {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      padding: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .search-hit b { color: var(--text); }
    .token-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 12px;
    }
    .token-pill, .source-pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--text);
      background: #0f1728;
      font-size: 12px;
    }
    .source-pill.hybrid { border-color: var(--green); color: var(--green); }
    .source-pill.vector { border-color: var(--cyan); color: var(--cyan); }
    .source-pill.bm25 { border-color: var(--amber); color: var(--amber); }
    .source-pill.sparse { border-color: #b6f06a; color: #b6f06a; }
    .rank-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
      margin: 10px 12px 0;
    }
    .rank-cell {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #10182a;
      color: var(--muted);
      font-size: 12px;
    }
    .rank-cell b { display: block; color: var(--text); font-size: 14px; margin-top: 2px; }
    .bar-row {
      display: grid;
      grid-template-columns: 150px minmax(120px, 1fr) 56px;
      gap: 8px;
      align-items: center;
      margin: 8px 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .bar-track {
      height: 8px;
      border-radius: 999px;
      background: #0f1728;
      border: 1px solid var(--line);
      overflow: hidden;
    }
    .bar-fill { height: 100%; background: var(--green); }
    .bar-fill.vector { background: var(--cyan); }
    .bar-fill.bm25 { background: var(--amber); }
    .bar-fill.sparse { background: #b6f06a; }
    details.match-details {
      margin: 10px 12px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #10182a;
      color: var(--muted);
    }
    details.match-details summary {
      cursor: pointer;
      padding: 8px 10px;
      color: var(--text);
    }
    .match-body { padding: 0 10px 10px; }
    .file-button {
      display: inline-block;
      border-radius: 6px;
      color: #07111e;
      background: #d7def0;
      font-weight: 700;
      padding: 10px 14px;
      cursor: pointer;
      width: fit-content;
    }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }
    .image-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      overflow: hidden;
    }
    .image-card img {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #0f1728;
    }
    .image-card div {
      padding: 8px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .results { display: grid; grid-template-rows: auto 1fr; min-width: 0; }
    .tabs {
      display: flex;
      gap: 6px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #10182a;
      overflow-x: auto;
    }
    .tab {
      padding: 8px 10px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 6px;
      white-space: nowrap;
      font-size: 13px;
    }
    .tab.active { border-color: var(--cyan); color: var(--cyan); }
    .content { padding: 16px; overflow: auto; }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(110px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric { background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px; padding: 12px; }
    .metric b { display: block; font-size: 20px; color: var(--green); }
    .metric span { color: var(--muted); font-size: 12px; }
    .section {
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 10px;
      background: var(--panel-2);
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .answer-box .answer-text {
      padding: 14px;
      line-height: 1.75;
      color: var(--text);
      font-size: 15px;
    }
    .citation-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .citation-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      padding: 12px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .citation-card b,
    .citation-card span,
    .citation-card code,
    .citation-card p { display: block; margin-top: 6px; }
    .citation-card b { color: var(--text); }
    .citation-card code { color: var(--green); font-size: 12px; }
    .citation-card p { line-height: 1.55; }
    .citation-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .citation-meta span {
      display: inline-block;
      margin-top: 0;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      color: var(--muted);
      background: #111d31;
    }
    .citation-badge {
      display: inline-block;
      border: 1px solid var(--cyan);
      border-radius: 6px;
      color: var(--cyan);
      padding: 2px 6px;
      font-weight: 700;
    }
    .inline-citation {
      display: inline-block;
      border: 1px solid var(--cyan);
      border-radius: 5px;
      padding: 0 5px;
      margin: 0 1px;
      color: var(--cyan);
      background: rgba(88, 211, 255, 0.08);
      font-weight: 700;
    }
    .provenance-line {
      margin-top: 10px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      background: #111d31;
      font-size: 12px;
      line-height: 1.55;
    }
    pre {
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 13px;
      line-height: 1.55;
    }
    table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
    td, th { border: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { background: #192943; }
    .diag { border-left: 3px solid var(--amber); padding: 10px 12px; background: var(--panel-2); margin-bottom: 8px; }
    .error { border-left-color: var(--pink); }
    .document-layout {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 16px;
      min-height: calc(100vh - 110px);
    }
    .document-tools {
      padding: 12px;
      display: grid;
      gap: 10px;
      border-bottom: 1px solid var(--line);
      background: #10182a;
    }
    .document-tools input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f1728;
      color: var(--text);
      padding: 10px 12px;
      outline: none;
    }
    .document-list {
      display: grid;
      gap: 8px;
      padding: 12px;
      max-height: calc(100vh - 230px);
      overflow: auto;
    }
    .document-row {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      padding: 10px;
      display: grid;
      gap: 6px;
    }
    .document-row.active { border-color: var(--cyan); }
    .document-meta {
      color: var(--muted);
      font-size: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 22px;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 2px 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .badge.ok { border-color: var(--green); color: var(--green); }
    .badge.warn { border-color: var(--amber); color: var(--amber); }
    .badge.off, .badge.error { border-color: var(--pink); color: var(--pink); }
    .document-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    @media (max-width: 1100px) {
      .app-page.chat-page.active {
        height: auto;
        min-height: calc(100dvh - 62px);
        place-items: start center;
        overflow: visible;
      }
      .chat-workspace {
        width: min(920px, 100%);
        height: auto;
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: minmax(200px, 30dvh) minmax(560px, 65dvh);
      }
    }
    @media (max-width: 900px) {
      .topbar { grid-template-columns: 1fr; justify-items: stretch; }
      .app-nav { justify-self: start; }
      .topbar .status { justify-self: start; }
      .app-page { padding: 14px; }
      .layout { grid-template-columns: 1fr; }
      .document-layout { grid-template-columns: 1fr; }
      .settings-page { grid-template-columns: 1fr; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chat-composer { grid-template-columns: auto minmax(0, 1fr) auto auto; }
      .chat-composer button { width: 44px; border-radius: 999px; }
    }
    @media (max-width: 520px) {
      .app-page.chat-page.active { padding: 12px; }
      .chat-workspace {
        gap: 10px;
        grid-template-rows: 180px minmax(580px, 70dvh);
      }
      .llm-trace-header { padding: 10px 12px; }
      .llm-trace-status { padding: 8px 12px; }
      .chat-message { padding: 12px; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">绿</span><span>小绿 · 绿色江河志愿讲解员</span></div>
      <nav class="app-nav" aria-label="主导航">
        <button class="nav-btn active" data-page="chat">对话</button>
        <button class="nav-btn" data-page="docs">文档解析</button>
        <button class="nav-btn" data-page="manage">文档管理</button>
        <button class="nav-btn" data-page="settings">配置</button>
      </nav>
      <div class="status" id="jobStatus">等待上传</div>
    </header>
    <main class="app-main">
      <section class="app-page chat-page active" id="page-chat">
        <div class="chat-workspace">
          <aside class="llm-trace-panel" aria-label="LLM 实际输入">
            <div class="llm-trace-header">
              <h2>LLM 提示词与上下文</h2>
              <span class="local-badge">LOCAL</span>
            </div>
            <div class="llm-trace-status" id="llmTraceStatus" aria-live="polite">等待本轮请求</div>
            <div class="llm-trace-feed" id="llmTraceFeed">
              <p class="llm-trace-empty">尚无 LLM 输入</p>
            </div>
          </aside>
          <div class="chat-shell">
          <div class="chat-feed" id="chatMessages">
            <div class="chat-empty" id="chatEmpty">
              <h1>你好，我是小绿</h1>
              <p>绿色江河环保组织志愿讲解员，用科技助力环保工作，并基于已入库资料给出可靠回答。</p>
            </div>
          </div>
          <div class="chat-composer">
            <button class="secondary" id="voiceMicBtn" title="语音输入" disabled>◉</button>
            <textarea id="chatInput" placeholder="询问你的知识库，例如：这份文档里图片 OCR 是怎么处理的？"></textarea>
            <button class="secondary" id="autoSpeakBtn" title="语音回复：开" disabled>♪</button>
            <button id="chatSendBtn" title="发送">↑</button>
          </div>
          <div class="wakeup-bar">
            <label class="toggle-row" for="wakeupToggle">
              <span id="wakeupLabel">唤醒“小绿小绿”</span>
              <input id="wakeupToggle" type="checkbox" checked />
            </label>
            <span class="wakeup-status" id="wakeupStatus">正在检查离线唤醒模型</span>
          </div>
          <div class="chat-status" id="chatStatus">Qwen Audio 3.1 / 实时检索 / 流式语音回复</div>
          </div>
        </div>
      </section>
      <section class="app-page" id="page-docs">
        <main class="layout">
      <section class="panel">
        <h2>文件</h2>
        <label class="dropzone" id="dropzone" for="fileInput">
          <span><strong>拖入文件</strong>或点击选择</span>
        </label>
        <input id="fileInput" type="file" multiple />
        <div class="file-list" id="fileList"></div>
        <div class="actions">
          <button id="uploadBtn" disabled>解析</button>
          <button class="secondary" id="clearBtn">清空</button>
        </div>
        <label class="toggle-row" for="autoIndexToggle">
          <span>上传解析后自动全部入库</span>
          <input id="autoIndexToggle" type="checkbox" checked />
        </label>
        <h2>向量入库</h2>
        <div class="settings-form">
          <div class="model-line" id="indexStats">索引统计读取中</div>
          <div class="settings-status" id="indexStatus">解析完成后可以入库</div>
          <div class="actions settings-actions">
            <button id="indexBtn" disabled>入库当前解析结果</button>
            <button class="secondary" id="refreshStatsBtn">刷新统计</button>
          </div>
        </div>
        <h2>图片向量</h2>
        <div class="settings-form">
          <div class="model-line" id="imageIndexStats">图片索引统计读取中</div>
          <div class="settings-status" id="imageIndexStatus">解析完成后可以入库图片</div>
          <div class="actions settings-actions">
            <button id="imageIndexBtn" disabled>入库当前图片</button>
            <button class="secondary" id="refreshImageStatsBtn">刷新图片统计</button>
          </div>
          <label for="imageSearchInput">以图搜图</label>
          <label class="file-button" for="imageSearchInput">选择查询图片</label>
          <input id="imageSearchInput" type="file" accept="image/*" />
          <div class="model-line" id="imageQueryName">未选择图片</div>
          <div class="actions settings-actions">
            <button class="secondary" id="imageSearchBtn" disabled>搜索相似图片</button>
          </div>
          <div class="search-results" id="imageSearchResults"></div>
        </div>
      </section>
      <section class="panel results">
        <div class="tabs" id="tabs"></div>
        <div class="content" id="content">
          <div class="diag">上传后会在这里显示 sections、chunks、tables、assets 和诊断信息。</div>
        </div>
      </section>
        </main>
      </section>
      <section class="app-page" id="page-manage">
        <main class="document-layout">
          <section class="panel">
            <h2>文档库</h2>
            <div class="document-tools">
              <input id="documentFilterInput" type="text" placeholder="按文件名、类型、状态筛选" />
              <div class="actions settings-actions">
                <button class="secondary" id="refreshDocumentsBtn">刷新文档</button>
              </div>
              <div class="model-line" id="documentListStatus">正在读取文档列表</div>
            </div>
            <div class="document-list" id="documentList"></div>
          </section>
          <section class="panel results">
            <div class="tabs">
              <button class="tab active">文档详情</button>
            </div>
            <div class="content" id="documentDetail">
              <div class="diag">选择左侧文档后，可以查看 chunks、图片、索引计数，并执行启用、禁用、重建、删除。</div>
            </div>
          </section>
        </main>
      </section>
      <section class="app-page" id="page-settings">
        <div class="settings-page">
          <section class="panel">
            <h2>模型配置</h2>
            <div class="settings-form">
              <label for="apiKeyInput">百炼 API Key</label>
              <input id="apiKeyInput" type="password" autocomplete="off" placeholder="sk-..." />
              <div class="model-line" id="modelLine">EmbeddingGemma 2 / 768 维</div>
              <div class="settings-status" id="settingsStatus">读取配置中</div>
              <div class="actions settings-actions">
                <button id="saveApiKeyBtn">保存</button>
                <button class="secondary" id="testApiKeyBtn">检查</button>
                <button class="danger" id="deleteApiKeyBtn">删除</button>
              </div>
              <div hidden>
              <label for="deepseekApiKeyInput">DeepSeek API Key</label>
              <input id="deepseekApiKeyInput" type="password" autocomplete="off" placeholder="sk-..." />
              <div class="model-line" id="routerAnswerLine">qwen3.7-flash / Router · deepseek-v4-flash / Answer</div>
              <div class="settings-status" id="deepseekStatus">读取 DeepSeek 配置中</div>
              <div class="actions settings-actions">
                <button id="saveDeepseekApiKeyBtn">保存 DeepSeek</button>
                <button class="secondary" id="testDeepseekApiKeyBtn">检查 DeepSeek</button>
                <button class="danger" id="deleteDeepseekApiKeyBtn">删除 DeepSeek</button>
              </div>
              </div>
            </div>
            <h2>切分配置</h2>
            <div class="settings-form">
              <label for="chunkSizeInput">chunk 字符数</label>
              <input id="chunkSizeInput" type="number" min="500" max="8000" step="100" />
              <label for="chunkOverlapInput">overlap 字符数</label>
              <input id="chunkOverlapInput" type="number" min="0" max="2000" step="50" />
              <div class="model-line" id="chunkingLine">chunk 2400 / overlap 300</div>
              <div class="settings-status" id="chunkingStatus">读取切分配置中</div>
              <div class="actions settings-actions">
                <button id="saveChunkingBtn">保存切分</button>
              </div>
            </div>
            <h2>生成回复模型上下文</h2>
            <div class="settings-form">
              <label for="conversationRoundsInput">上下文注入轮数</label>
              <input id="conversationRoundsInput" type="number" min="1" max="100" step="1" value="40" />
              <div class="model-line" id="conversationHistoryLine">生成回复模型注入最近 40 轮；一问一答计 1 轮</div>
              <div class="settings-status" id="conversationHistoryStatus">读取上下文配置中</div>
              <div class="actions settings-actions">
                <button id="saveConversationHistoryBtn">保存上下文轮数</button>
              </div>
            </div>
            <h2>离线唤醒</h2>
            <div class="settings-form">
              <label for="wakeupWordInput">自定义唤醒词</label>
              <input id="wakeupWordInput" type="text" minlength="2" maxlength="32" placeholder="例如：小绿小绿" />
              <div class="model-line" id="wakeupSettingsLine">Sherpa-ONNX KWS / CPU</div>
              <div class="settings-status" id="wakeupSettingsStatus">读取唤醒配置中；建议使用 4～8 个中文字符</div>
              <div class="actions settings-actions">
                <button id="saveWakeupSettingsBtn">保存并重启唤醒</button>
              </div>
            </div>
            <h2>语音断句</h2>
            <div class="settings-form">
              <label class="toggle-row" for="vadEnabledInput">
                <span>启用流式 VAD</span>
                <input id="vadEnabledInput" type="checkbox" checked />
              </label>
              <label for="vadEndSilenceInput">断句静音间隔（毫秒）</label>
              <input id="vadEndSilenceInput" type="number" min="200" max="3000" step="100" value="400" />
              <div class="model-line" id="vadSettingsLine">FSMN-VAD / CPU / 400ms</div>
              <div class="settings-status" id="vadSettingsStatus">读取 VAD 配置中</div>
              <div class="actions settings-actions">
                <button id="saveVadSettingsBtn">保存语音断句</button>
              </div>
            </div>
            <h2>ASR 易错字纠错</h2>
            <div class="settings-form">
              <label class="toggle-row" for="asrCorrectionEnabledInput">
                <span>启用识别后处理纠错</span>
                <input id="asrCorrectionEnabledInput" type="checkbox" checked />
              </label>
              <label for="asrCorrectionRulesInput">自定义纠错表（每行：误识词 =&gt; 正确词）</label>
              <textarea id="asrCorrectionRulesInput" spellcheck="false" placeholder="绿色江和 => 绿色江河&#10;索兰达杰 => 索南达杰"></textarea>
              <div class="model-line" id="asrCorrectionLine">内置领域词表 + 0 条自定义规则</div>
              <div class="settings-status" id="asrCorrectionStatus">读取 ASR 纠错配置中</div>
              <div class="actions settings-actions">
                <button id="saveAsrCorrectionsBtn">保存 ASR 纠错表</button>
              </div>
            </div>
          </section>
          <section class="panel">
            <h2>回复提示词</h2>
            <div class="settings-form" hidden>
              <label for="deepseekTurnPromptInput">轮次控制提示词</label>
              <textarea id="deepseekTurnPromptInput" spellcheck="false"></textarea>
              <div class="settings-status" id="deepseekPromptStatus">读取提示词配置中</div>
              <div class="actions settings-actions">
                <button id="saveDeepseekPromptBtn">保存提示词</button>
                <button class="secondary" id="resetDeepseekPromptBtn">恢复默认</button>
              </div>
            </div>
            <div class="settings-form">
              <label>当前生效的完整 LLM 提示词</label>
              <div class="settings-status" id="llmPromptCatalogStatus">读取完整提示词中</div>
              <div class="prompt-catalog" id="llmPromptCatalog"></div>
            </div>
          </section>
        </div>
      </section>
    </main>
  </div>
  <script src="/api/display/client.js"></script>
  <script src="/api/voice/pcm-player.js"></script>
  <script>
    const navButtons = document.querySelectorAll('.nav-btn');
    const pages = {
      chat: document.getElementById('page-chat'),
      docs: document.getElementById('page-docs'),
      manage: document.getElementById('page-manage'),
      settings: document.getElementById('page-settings')
    };
    const chatMessages = document.getElementById('chatMessages');
    const chatEmpty = document.getElementById('chatEmpty');
    const chatInput = document.getElementById('chatInput');
    const voiceMicBtn = document.getElementById('voiceMicBtn');
    const autoSpeakBtn = document.getElementById('autoSpeakBtn');
    const wakeupToggle = document.getElementById('wakeupToggle');
    const wakeupLabel = document.getElementById('wakeupLabel');
    const wakeupStatus = document.getElementById('wakeupStatus');
    const chatSendBtn = document.getElementById('chatSendBtn');
    const chatStatus = document.getElementById('chatStatus');
    const llmTraceStatus = document.getElementById('llmTraceStatus');
    const llmTraceFeed = document.getElementById('llmTraceFeed');
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const fileList = document.getElementById('fileList');
    const uploadBtn = document.getElementById('uploadBtn');
    const clearBtn = document.getElementById('clearBtn');
    const autoIndexToggle = document.getElementById('autoIndexToggle');
    const jobStatus = document.getElementById('jobStatus');
    const tabs = document.getElementById('tabs');
    const content = document.getElementById('content');
    const apiKeyInput = document.getElementById('apiKeyInput');
    const deepseekApiKeyInput = document.getElementById('deepseekApiKeyInput');
    const deepseekTurnPromptInput = document.getElementById('deepseekTurnPromptInput');
    const deepseekPromptStatus = document.getElementById('deepseekPromptStatus');
    const llmPromptCatalog = document.getElementById('llmPromptCatalog');
    const llmPromptCatalogStatus = document.getElementById('llmPromptCatalogStatus');
    const chunkSizeInput = document.getElementById('chunkSizeInput');
    const chunkOverlapInput = document.getElementById('chunkOverlapInput');
    const conversationRoundsInput = document.getElementById('conversationRoundsInput');
    const wakeupWordInput = document.getElementById('wakeupWordInput');
    const wakeupSettingsLine = document.getElementById('wakeupSettingsLine');
    const wakeupSettingsStatus = document.getElementById('wakeupSettingsStatus');
    const vadEnabledInput = document.getElementById('vadEnabledInput');
    const vadEndSilenceInput = document.getElementById('vadEndSilenceInput');
    const asrCorrectionEnabledInput = document.getElementById('asrCorrectionEnabledInput');
    const asrCorrectionRulesInput = document.getElementById('asrCorrectionRulesInput');
    const asrCorrectionLine = document.getElementById('asrCorrectionLine');
    const asrCorrectionStatus = document.getElementById('asrCorrectionStatus');
    const saveChunkingBtn = document.getElementById('saveChunkingBtn');
    const saveConversationHistoryBtn = document.getElementById('saveConversationHistoryBtn');
    const saveWakeupSettingsBtn = document.getElementById('saveWakeupSettingsBtn');
    const saveVadSettingsBtn = document.getElementById('saveVadSettingsBtn');
    const saveAsrCorrectionsBtn = document.getElementById('saveAsrCorrectionsBtn');
    const saveApiKeyBtn = document.getElementById('saveApiKeyBtn');
    const deleteApiKeyBtn = document.getElementById('deleteApiKeyBtn');
    const testApiKeyBtn = document.getElementById('testApiKeyBtn');
    const saveDeepseekApiKeyBtn = document.getElementById('saveDeepseekApiKeyBtn');
    const deleteDeepseekApiKeyBtn = document.getElementById('deleteDeepseekApiKeyBtn');
    const testDeepseekApiKeyBtn = document.getElementById('testDeepseekApiKeyBtn');
    const saveDeepseekPromptBtn = document.getElementById('saveDeepseekPromptBtn');
    const resetDeepseekPromptBtn = document.getElementById('resetDeepseekPromptBtn');
    const settingsStatus = document.getElementById('settingsStatus');
    const deepseekStatus = document.getElementById('deepseekStatus');
    const modelLine = document.getElementById('modelLine');
    const routerAnswerLine = document.getElementById('routerAnswerLine');
    const chunkingLine = document.getElementById('chunkingLine');
    const chunkingStatus = document.getElementById('chunkingStatus');
    const conversationHistoryLine = document.getElementById('conversationHistoryLine');
    const conversationHistoryStatus = document.getElementById('conversationHistoryStatus');
    const vadSettingsLine = document.getElementById('vadSettingsLine');
    const vadSettingsStatus = document.getElementById('vadSettingsStatus');
    const indexBtn = document.getElementById('indexBtn');
    const refreshStatsBtn = document.getElementById('refreshStatsBtn');
    const indexStatus = document.getElementById('indexStatus');
    const indexStats = document.getElementById('indexStats');
    const imageIndexBtn = document.getElementById('imageIndexBtn');
    const refreshImageStatsBtn = document.getElementById('refreshImageStatsBtn');
    const imageIndexStats = document.getElementById('imageIndexStats');
    const imageIndexStatus = document.getElementById('imageIndexStatus');
    const imageSearchInput = document.getElementById('imageSearchInput');
    const imageQueryName = document.getElementById('imageQueryName');
    const imageSearchBtn = document.getElementById('imageSearchBtn');
    const imageSearchResults = document.getElementById('imageSearchResults');
    const documentFilterInput = document.getElementById('documentFilterInput');
    const refreshDocumentsBtn = document.getElementById('refreshDocumentsBtn');
    const documentListStatus = document.getElementById('documentListStatus');
    const documentList = document.getElementById('documentList');
    const documentDetail = document.getElementById('documentDetail');
    const AUTO_INDEX_STORAGE_KEY = 'rag.autoIndexUploads';
    const AUTO_SPEAK_STORAGE_KEY = 'rag.autoSpeak';
    const WAKEUP_STORAGE_KEY = 'rag.wakeupEnabled';
    let selectedFiles = [];
    let currentJob = null;
    let currentJobPayload = null;
    let currentSettings = null;
    let currentIndexJob = null;
    let currentImageIndexJob = null;
    let autoTextIndexStartedForJob = null;
    let autoImageIndexStartedForJob = null;
    let activeIndex = 0;
    let documentsCache = [];
    let activeDocumentId = null;
    let voiceStatusPayload = null;
    let isRecordingVoice = false;
    let isAnswering = false;
    let currentAnswerController = null;
    let answerGeneration = 0;
    let liveLlmInputs = [];
    let activeAnswerNode = null;
    let interruptInProgress = false;
    let voiceRecorder = null;
    let voiceStopTimer = null;
    let currentVoiceAudio = null;
    let currentTtsController = null;
    let currentTtsObjectUrl = null;
    let activeSpeakButton = null;
    let ttsPlaybackVersion = 0;
    let ttsSentenceQueue = [];
    let ttsQueueRunPromise = null;
    const voiceConversationId = crypto.randomUUID();
    let pendingRetrievalToken = null;
    let qwenAudioContext = null;
    let qwenPlayer = null;
    function unlockReplyAudio() {
      if (!qwenAudioContext) qwenAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      return qwenAudioContext.resume();
    }
    document.addEventListener('pointerdown', () => { void unlockReplyAudio().catch(() => {}); });
    let ttsActiveItem = null;
    let ttsPrefetchState = null;
    let displayEmotionState = null;
    let displayRestTimer = null;
    let displayTransient = null;
    let displayPlayingAudio = null;
    let displayWakeupPhase = '';
    let displayAsrPending = false;
    let displayTextResponse = false;
    const displaySync = new DisplaySessionClient();
    let wakeupRecorder = null;
    let wakeupResumeTimer = null;
    let wakeupStartToken = 0;
    const wakeupStartingSockets = new Set();
    let wakeupAutoStartAttempted = false;
    let wakeupEnabled = localStorage.getItem(WAKEUP_STORAGE_KEY) !== '0';
    let autoSpeakEnabled = resolveAutoSpeakPreference(
      localStorage.getItem(AUTO_SPEAK_STORAGE_KEY)
    );

    function setDisplayEmotion(state, restAfterMs = 0, claim = false) {
      clearTimeout(displayRestTimer);
      displayTransient = restAfterMs > 0 ? { state, until: Date.now() + restAfterMs } : null;
      if (['idle', 'success', 'error'].includes(state)) displayAsrPending = false;
      if (displayPlayingAudio) state = 'speaking';
      const takeControl = claim || (state === 'listening' && displayEmotionState !== state);
      displayEmotionState = state;
      displaySync.setState(state, takeControl);
      if (restAfterMs > 0) displayRestTimer = setTimeout(settleDisplayEmotion, restAfterMs);
    }

    function settleDisplayEmotion() {
      let state = 'idle';
      if (displayPlayingAudio) state = 'speaking';
      else if (displayTransient && displayTransient.until > Date.now()) state = displayTransient.state;
      else if (isRecordingVoice || displayWakeupPhase === 'listening') state = 'listening';
      else if (displayAsrPending || ['recognizing', 'acknowledging', 'interrupt_ack'].includes(displayWakeupPhase)) state = 'thinking';
      else if (isAnswering) state = displayTextResponse ? 'speaking' : 'thinking';
      else if (ttsActiveItem || ttsSentenceQueue.length || currentTtsController) state = 'thinking';
      displayEmotionState = state;
      displaySync.setState(state);
    }

    function trackDisplayAudio(audio, isCurrent) {
      const playing = () => {
        if (!isCurrent()) return;
        displayPlayingAudio = audio;
        settleDisplayEmotion();
      };
      const stopped = () => {
        if (displayPlayingAudio !== audio) return;
        displayPlayingAudio = null;
        settleDisplayEmotion();
      };
      audio.addEventListener('playing', playing);
      for (const event of ['waiting', 'pause', 'ended', 'error']) audio.addEventListener(event, stopped);
      return () => {
        audio.removeEventListener('playing', playing);
        for (const event of ['waiting', 'pause', 'ended', 'error']) audio.removeEventListener(event, stopped);
        stopped();
      };
    }

    function resolveAutoSpeakPreference(storedValue) {
      return storedValue === null || storedValue === '1';
    }

    navButtons.forEach(button => {
      button.addEventListener('click', () => switchPage(button.dataset.page));
    });

    chatSendBtn.addEventListener('click', sendChatMessage);
    voiceMicBtn.addEventListener('click', toggleVoiceRecording);
    autoSpeakBtn.addEventListener('click', toggleAutoSpeak);
    wakeupToggle.checked = wakeupEnabled;
    wakeupToggle.addEventListener('change', toggleWakeupListening);
    chatInput.addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
      }
    });

    function switchPage(pageName) {
      Object.entries(pages).forEach(([name, page]) => {
        page.classList.toggle('active', name === pageName);
      });
      navButtons.forEach(button => {
        button.classList.toggle('active', button.dataset.page === pageName);
      });
      if (pageName === 'docs') {
        jobStatus.textContent = currentJobPayload ? `${currentJobPayload.status} ${currentJobPayload.progress}/${currentJobPayload.total}` : '等待上传';
      } else if (pageName === 'chat') {
        jobStatus.textContent = '对话';
      } else if (pageName === 'manage') {
        jobStatus.textContent = '文档管理';
        loadDocuments();
      } else {
        jobStatus.textContent = '配置';
      }
    }

    function setFiles(files) {
      selectedFiles = Array.from(files);
      uploadBtn.disabled = selectedFiles.length === 0;
      fileList.innerHTML = selectedFiles.map(file => `
        <div class="file-row"><span class="file-name">${escapeHtml(file.name)}</span><span>${formatBytes(file.size)}</span></div>
      `).join('');
    }

    fileInput.addEventListener('change', event => setFiles(event.target.files));
    clearBtn.addEventListener('click', () => {
      selectedFiles = [];
      currentJob = null;
      currentJobPayload = null;
      currentIndexJob = null;
      currentImageIndexJob = null;
      autoTextIndexStartedForJob = null;
      autoImageIndexStartedForJob = null;
      fileInput.value = '';
      fileList.innerHTML = '';
      uploadBtn.disabled = true;
      indexStatus.textContent = '解析完成后可以入库';
      imageIndexStatus.textContent = '解析完成后可以入库图片';
      updateIndexButtonState();
      updateImageIndexButtonState();
    });
    ['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => {
      event.preventDefault();
      dropzone.classList.add('active');
    }));
    ['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => {
      event.preventDefault();
      dropzone.classList.remove('active');
    }));
    dropzone.addEventListener('drop', event => setFiles(event.dataTransfer.files));
    uploadBtn.addEventListener('click', upload);
    saveChunkingBtn.addEventListener('click', saveChunking);
    saveConversationHistoryBtn.addEventListener('click', saveConversationHistory);
    saveWakeupSettingsBtn.addEventListener('click', saveWakeupSettings);
    saveVadSettingsBtn.addEventListener('click', saveVadSettings);
    saveAsrCorrectionsBtn.addEventListener('click', saveAsrCorrections);
    saveApiKeyBtn.addEventListener('click', saveApiKey);
    deleteApiKeyBtn.addEventListener('click', deleteApiKey);
    testApiKeyBtn.addEventListener('click', testApiKey);
    saveDeepseekApiKeyBtn.addEventListener('click', saveDeepseekApiKey);
    deleteDeepseekApiKeyBtn.addEventListener('click', deleteDeepseekApiKey);
    testDeepseekApiKeyBtn.addEventListener('click', testDeepseekApiKey);
    saveDeepseekPromptBtn.addEventListener('click', saveDeepseekPrompt);
    resetDeepseekPromptBtn.addEventListener('click', resetDeepseekPrompt);
    indexBtn.addEventListener('click', startIndexing);
    refreshStatsBtn.addEventListener('click', loadIndexStats);
    imageIndexBtn.addEventListener('click', startImageIndexing);
    refreshImageStatsBtn.addEventListener('click', loadImageIndexStats);
    refreshDocumentsBtn.addEventListener('click', loadDocuments);
    documentFilterInput.addEventListener('input', () => renderDocumentList(documentsCache));
    documentDetail.addEventListener('click', handleDocumentAction);
    imageSearchBtn.addEventListener('click', searchImages);
    imageSearchInput.addEventListener('change', () => {
      const file = imageSearchInput.files && imageSearchInput.files[0];
      imageQueryName.textContent = file ? `${file.name} / ${formatBytes(file.size)}` : '未选择图片';
      imageSearchBtn.disabled = !file;
    });
    autoIndexToggle.checked = loadAutoIndexPreference();
    autoIndexToggle.addEventListener('change', () => {
      localStorage.setItem(AUTO_INDEX_STORAGE_KEY, autoIndexToggle.checked ? '1' : '0');
      if (autoIndexToggle.checked) {
        maybeStartAutoIndexing();
      }
    });
    window.addEventListener('pagehide', () => {
      answerGeneration += 1;
      currentAnswerController?.abort();
      currentAnswerController = null;
      isAnswering = false;
      isRecordingVoice = false;
      displayAsrPending = false;
      displayWakeupPhase = '';
      displayTransient = null;
      clearTimeout(displayRestTimer);
      stopCurrentTTS();
      for (const recorder of [voiceRecorder, wakeupRecorder]) {
        if (!recorder) continue;
        recorder.stopping = true;
        recorder.completed = true;
        recorder.promptController?.abort();
        recorder.promptAudio?.pause();
        recorder.stream?.getTracks().forEach(track => track.stop());
        recorder.socket?.close();
      }
      voiceRecorder = null;
      wakeupRecorder = null;
    });
    window.addEventListener('pageshow', event => {
      if (!event.persisted) return;
      setVoiceRecordingUi(false, '');
      settleDisplayEmotion();
      wakeupAutoStartAttempted = false;
      void loadVoiceStatus();
    });
    displaySync.start();
    switchPage('chat');
    loadSettings();
    loadLlmPromptCatalog();
    applyAutoSpeakState();
    loadVoiceStatus();
    loadIndexStats();
    loadImageIndexStats();
    loadDocuments();

    async function loadSettings() {
      settingsStatus.textContent = '读取配置中';
      try {
        const response = await fetch('/api/settings');
        const payload = await readJsonResponse(response);
        renderSettings(payload);
      } catch (error) {
        settingsStatus.textContent = `读取失败：${error.message}`;
        deepseekPromptStatus.textContent = `读取失败：${error.message}`;
      }
    }

    async function loadLlmPromptCatalog() {
      llmPromptCatalogStatus.textContent = '读取完整提示词中';
      try {
        const response = await fetch('/api/settings/llm-prompts');
        const payload = await readJsonResponse(response);
        renderLlmPromptCatalog(payload);
      } catch (error) {
        llmPromptCatalogStatus.textContent = `读取失败：${error.message}`;
        llmPromptCatalog.innerHTML = '';
      }
    }

    function renderLlmPromptCatalog(payload) {
      const prompts = Array.isArray(payload.prompts) ? payload.prompts : [];
      llmPromptCatalog.innerHTML = prompts.map((prompt, index) => {
        const messages = Array.isArray(prompt.messages) ? prompt.messages : [];
        const body = messages.map(message => {
          const repeatable = message.repeatable ? '（历史消息模板）' : '';
          return `[${String(message.role || '').toUpperCase()}]${repeatable}\n${String(message.content || '')}`;
        }).join('\n\n');
        const title = `${prompt.title || prompt.id || `提示词 ${index + 1}`} · ${prompt.model || '未指定模型'} · ${prompt.mode || ''}`;
        return `<details class="prompt-entry"><summary>${escapeHtml(title)} <span>${escapeHtml(prompt.id || '')}</span></summary><pre>${escapeHtml(body)}</pre></details>`;
      }).join('');
      llmPromptCatalogStatus.textContent = prompts.length
        ? `已加载 ${prompts.length} 组提示词；花括号占位符会在请求时替换为当前上下文`
        : '没有可展示的提示词';
    }

    function renderSettings(settings) {
      currentSettings = settings;
      modelLine.textContent = `${settings.embedding_model} / ${settings.embedding_dimensions} 维`;
      routerAnswerLine.textContent = `${settings.router_model || 'qwen3.7-flash'} / Router · ${settings.answer_model || 'deepseek-v4-flash'} / Answer`;
      deepseekTurnPromptInput.value = settings.deepseek_turn_prompt || '';
      deepseekPromptStatus.textContent = '轮次控制提示词已加载';
      chunkSizeInput.value = settings.chunk_size || 2400;
      chunkOverlapInput.value = settings.chunk_overlap || 300;
      chunkingLine.textContent = `chunk ${settings.chunk_size || 2400} / overlap ${settings.chunk_overlap || 300}`;
      chunkingStatus.textContent = '切分配置已加载';
      const rounds = Number(settings.conversation_max_rounds || 40);
      conversationRoundsInput.value = rounds;
      conversationHistoryLine.textContent = `生成回复模型注入最近 ${rounds} 轮；一问一答计 1 轮`;
      conversationHistoryStatus.textContent = '生成回复模型上下文配置已加载';
      wakeupWordInput.value = settings.wakeup_word || '小绿小绿';
      wakeupSettingsLine.textContent = `${settings.wakeup_model || 'Sherpa-ONNX KWS'} / ${(settings.wakeup_provider || 'cpu').toUpperCase()}`;
      wakeupSettingsStatus.textContent = `当前唤醒词：“${settings.wakeup_word || '小绿小绿'}”；建议使用 4～8 个中文字符`;
      const vadEndSilenceMs = Number(settings.vad_end_silence_ms || 400);
      vadEnabledInput.checked = settings.vad_enabled !== false;
      vadEndSilenceInput.value = vadEndSilenceMs;
      vadSettingsStatus.textContent = vadEnabledInput.checked
        ? '流式 VAD 已启用'
        : 'VAD 已关闭：手动语音使用整段录音，唤醒问答暂时停用';
      const correctionRules = Array.isArray(settings.asr_correction_rules)
        ? settings.asr_correction_rules : [];
      asrCorrectionEnabledInput.checked = settings.asr_correction_enabled !== false;
      asrCorrectionRulesInput.value = correctionRules
        .map(rule => `${rule.wrong} => ${rule.correct}`)
        .join('\n');
      asrCorrectionLine.textContent = `内置领域词表 + ${correctionRules.length} 条自定义规则`;
      asrCorrectionStatus.textContent = asrCorrectionEnabledInput.checked
        ? 'ASR 纠错已启用，云端流式与本地识别统一生效'
        : 'ASR 纠错已关闭，规则仍会保留';
      renderVoiceConfigSummary();
      if (settings.has_api_key) {
        const source = settings.api_key_source === 'env' ? '环境变量' : '本地保存';
        settingsStatus.textContent = `已配置 ${settings.masked_api_key}（${source}）`;
      } else {
        settingsStatus.textContent = '未配置 API Key';
      }
      if (settings.has_deepseek_api_key) {
        const source = settings.deepseek_api_key_source === 'env' ? '环境变量' : '本地保存';
        deepseekStatus.textContent = `已配置 ${settings.masked_deepseek_api_key}（${source}）`;
      } else {
        deepseekStatus.textContent = '未配置 DeepSeek API Key；搜索会跳过 Query Rewrite';
      }
      updateIndexButtonState();
      updateImageIndexButtonState();
      maybeStartAutoIndexing();
    }

    function renderVoiceConfigSummary() {
      if (!currentSettings) return;
      const silenceMs = Number(currentSettings.vad_end_silence_ms || 400);
      const provider = voiceStatusPayload?.asr_provider_active === 'dashscope_realtime'
        ? 'Qwen Server VAD'
        : 'FSMN-VAD / CPU';
      vadSettingsLine.textContent = `${provider} / ${silenceMs}ms`;
    }

    async function saveChunking() {
      chunkingStatus.textContent = '保存中';
      try {
        const response = await fetch('/api/settings/chunking', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(readChunkingConfig())
        });
        const payload = await readJsonResponse(response);
        renderSettings(payload);
      } catch (error) {
        chunkingStatus.textContent = `保存失败：${error.message}`;
      }
    }

    async function saveConversationHistory() {
      conversationHistoryStatus.textContent = '保存中';
      try {
        const response = await fetch('/api/settings/conversation-history', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ max_rounds: Number(conversationRoundsInput.value || 40) })
        });
        const payload = await readJsonResponse(response);
        renderSettings(payload);
      } catch (error) {
        conversationHistoryStatus.textContent = `保存失败：${error.message}`;
      }
    }

    async function restoreWakeupAfterSettingsChange(shouldRun) {
      if (shouldRun) wakeupAutoStartAttempted = true;
      await loadVoiceStatus();
      if (
        shouldRun &&
        !wakeupRecorder &&
        wakeupEnabled &&
        voiceStatusPayload?.wakeup_available &&
        voiceStatusPayload?.vad_enabled !== false
      ) {
        await startWakeupListening();
      }
    }

    async function saveWakeupSettings() {
      const word = wakeupWordInput.value.trim();
      const wasRunning = Boolean(wakeupRecorder);
      saveWakeupSettingsBtn.disabled = true;
      wakeupSettingsStatus.textContent = '正在校验并应用新唤醒词';
      if (wasRunning) await stopWakeupListening('正在应用新唤醒词');
      try {
        const response = await fetch('/api/settings/wakeup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ word })
        });
        const payload = await readJsonResponse(response);
        renderSettings(payload);
        await restoreWakeupAfterSettingsChange(wasRunning);
        wakeupSettingsStatus.textContent = `已应用新唤醒词：“${payload.wakeup_word}”`;
      } catch (error) {
        await restoreWakeupAfterSettingsChange(wasRunning);
        wakeupSettingsStatus.textContent = `保存失败：${error.message}`;
      } finally {
        saveWakeupSettingsBtn.disabled = false;
      }
    }

    async function saveVadSettings() {
      const wasRunning = Boolean(wakeupRecorder);
      const shouldRunWakeup = wakeupEnabled && vadEnabledInput.checked;
      saveVadSettingsBtn.disabled = true;
      vadSettingsStatus.textContent = '保存中';
      if (wasRunning) await stopWakeupListening('正在应用新的断句间隔');
      try {
        const response = await fetch('/api/settings/vad', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: vadEnabledInput.checked,
            end_silence_ms: Number(vadEndSilenceInput.value || 400)
          })
        });
        const payload = await readJsonResponse(response);
        renderSettings(payload);
        await restoreWakeupAfterSettingsChange(shouldRunWakeup);
        vadSettingsStatus.textContent = `已应用 ${payload.vad_end_silence_ms}ms 断句间隔`;
      } catch (error) {
        await restoreWakeupAfterSettingsChange(wasRunning);
        vadSettingsStatus.textContent = `保存失败：${error.message}`;
      } finally {
        saveVadSettingsBtn.disabled = false;
      }
    }

    function parseAsrCorrectionRules(value) {
      const rules = [];
      const seen = new Set();
      const lines = String(value || '').split(/\r?\n/);
      lines.forEach((rawLine, index) => {
        const line = rawLine.trim();
        if (!line) return;
        const match = line.match(/^(.+?)\s*(?:=>|->)\s*(.+)$/);
        if (!match) throw new Error(`第 ${index + 1} 行格式错误，应为“误识词 => 正确词”`);
        const wrong = match[1].trim();
        const correct = match[2].trim();
        if (!wrong || !correct) throw new Error(`第 ${index + 1} 行不能包含空词`);
        if (wrong.length > 64 || correct.length > 64) throw new Error(`第 ${index + 1} 行单项不能超过 64 个字符`);
        if (wrong === correct) throw new Error(`第 ${index + 1} 行纠错前后不能相同`);
        if (seen.has(wrong)) throw new Error(`第 ${index + 1} 行误识词“${wrong}”重复`);
        seen.add(wrong);
        rules.push({ wrong, correct });
      });
      if (rules.length > 200) throw new Error('自定义纠错规则不能超过 200 条');
      return rules;
    }

    async function saveAsrCorrections() {
      saveAsrCorrectionsBtn.disabled = true;
      asrCorrectionStatus.textContent = '正在保存 ASR 纠错表';
      try {
        const rules = parseAsrCorrectionRules(asrCorrectionRulesInput.value);
        const response = await fetch('/api/settings/asr-corrections', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: asrCorrectionEnabledInput.checked,
            rules
          })
        });
        const payload = await readJsonResponse(response);
        renderSettings(payload);
        asrCorrectionStatus.textContent = payload.asr_correction_enabled
          ? `已启用并加载 ${payload.asr_correction_rules.length} 条自定义纠错规则`
          : `纠错已关闭，已保留 ${payload.asr_correction_rules.length} 条自定义规则`;
      } catch (error) {
        asrCorrectionStatus.textContent = `保存失败：${error.message}`;
      } finally {
        saveAsrCorrectionsBtn.disabled = false;
      }
    }

    async function saveApiKey() {
      settingsStatus.textContent = '保存中';
      try {
        const response = await fetch('/api/settings/api-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: apiKeyInput.value })
        });
        const payload = await readJsonResponse(response);
        apiKeyInput.value = '';
        renderSettings(payload);
      } catch (error) {
        settingsStatus.textContent = `保存失败：${error.message}`;
      }
    }

    async function deleteApiKey() {
      settingsStatus.textContent = '删除中';
      try {
        const response = await fetch('/api/settings/api-key', { method: 'DELETE' });
        const payload = await readJsonResponse(response);
        apiKeyInput.value = '';
        renderSettings(payload);
      } catch (error) {
        settingsStatus.textContent = `删除失败：${error.message}`;
      }
    }

    async function saveDeepseekApiKey() {
      deepseekStatus.textContent = '保存中';
      try {
        const response = await fetch('/api/settings/deepseek-api-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: deepseekApiKeyInput.value })
        });
        const payload = await readJsonResponse(response);
        deepseekApiKeyInput.value = '';
        renderSettings(payload);
      } catch (error) {
        deepseekStatus.textContent = `保存失败：${error.message}`;
      }
    }

    async function deleteDeepseekApiKey() {
      deepseekStatus.textContent = '删除中';
      try {
        const response = await fetch('/api/settings/deepseek-api-key', { method: 'DELETE' });
        const payload = await readJsonResponse(response);
        deepseekApiKeyInput.value = '';
        renderSettings(payload);
      } catch (error) {
        deepseekStatus.textContent = `删除失败：${error.message}`;
      }
    }

    async function saveDeepseekPrompt() {
      deepseekPromptStatus.textContent = '保存中';
      saveDeepseekPromptBtn.disabled = true;
      resetDeepseekPromptBtn.disabled = true;
      try {
        const response = await fetch('/api/settings/deepseek-prompt', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: deepseekTurnPromptInput.value })
        });
        const payload = await readJsonResponse(response);
        renderSettings(payload);
        loadLlmPromptCatalog();
        deepseekPromptStatus.textContent = '提示词已保存，新请求立即生效';
      } catch (error) {
        deepseekPromptStatus.textContent = `保存失败：${error.message}`;
      } finally {
        saveDeepseekPromptBtn.disabled = false;
        resetDeepseekPromptBtn.disabled = false;
      }
    }

    async function resetDeepseekPrompt() {
      deepseekPromptStatus.textContent = '正在恢复默认提示词';
      saveDeepseekPromptBtn.disabled = true;
      resetDeepseekPromptBtn.disabled = true;
      try {
        const response = await fetch('/api/settings/deepseek-prompt', { method: 'DELETE' });
        const payload = await readJsonResponse(response);
        renderSettings(payload);
        loadLlmPromptCatalog();
        deepseekPromptStatus.textContent = '已恢复默认提示词，新请求立即生效';
      } catch (error) {
        deepseekPromptStatus.textContent = `恢复失败：${error.message}`;
      } finally {
        saveDeepseekPromptBtn.disabled = false;
        resetDeepseekPromptBtn.disabled = false;
      }
    }

    async function testDeepseekApiKey() {
      deepseekStatus.textContent = '检查中';
      const body = deepseekApiKeyInput.value ? { api_key: deepseekApiKeyInput.value } : {};
      try {
        const response = await fetch('/api/settings/test-deepseek-api-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const payload = await readJsonResponse(response);
        deepseekStatus.textContent = `格式有效：${payload.checked_api_key}`;
      } catch (error) {
        deepseekStatus.textContent = `检查失败：${error.message}`;
      }
    }

    async function testApiKey() {
      settingsStatus.textContent = '检查中';
      const body = apiKeyInput.value ? { api_key: apiKeyInput.value } : {};
      try {
        const response = await fetch('/api/settings/test-api-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const payload = await readJsonResponse(response);
        settingsStatus.textContent = `格式有效：${payload.checked_api_key}`;
      } catch (error) {
        settingsStatus.textContent = `检查失败：${error.message}`;
      }
    }

    function updateIndexButtonState() {
      const parseDone = currentJobPayload && ['completed', 'completed_with_errors'].includes(currentJobPayload.status);
      const hasParsedDocuments = parseDone && (currentJobPayload.results || []).some(result => result.status === 'success');
      indexBtn.disabled = !(currentJob && hasParsedDocuments);
      if (!currentJob) indexStatus.textContent = '解析完成后可以入库';
    }

    function updateImageIndexButtonState() {
      const parseDone = currentJobPayload && ['completed', 'completed_with_errors'].includes(currentJobPayload.status);
      const hasParsedDocuments = parseDone && (currentJobPayload.results || []).some(result => result.status === 'success');
      const hasImages = parseDone && hasIndexableImages(currentJobPayload);
      const hasApiKey = currentSettings && currentSettings.has_api_key;
      imageIndexBtn.disabled = !(currentJob && hasParsedDocuments && hasImages && hasApiKey);
      if (!hasApiKey) {
        imageIndexStatus.textContent = '请先配置百炼 API Key';
      } else if (!currentJob) {
        imageIndexStatus.textContent = '解析完成后可以入库图片';
      } else if (parseDone && hasParsedDocuments && !hasImages) {
        imageIndexStatus.textContent = '未发现可入库图片';
      }
    }

    function loadAutoIndexPreference() {
      return localStorage.getItem(AUTO_INDEX_STORAGE_KEY) !== '0';
    }

    function maybeStartAutoIndexing() {
      if (!autoIndexToggle.checked || !currentJob || !currentJobPayload) return;
      if (!['completed', 'completed_with_errors'].includes(currentJobPayload.status)) return;
      const successful = (currentJobPayload.results || []).filter(result => result.status === 'success').length;
      if (!successful) return;
      if (autoTextIndexStartedForJob !== currentJob && !currentIndexJob) {
        autoTextIndexStartedForJob = currentJob;
        startIndexing(true);
      }
      if (currentSettings?.has_api_key && hasIndexableImages(currentJobPayload)) {
        if (autoImageIndexStartedForJob !== currentJob && !currentImageIndexJob) {
          autoImageIndexStartedForJob = currentJob;
          startImageIndexing(true);
        }
      } else {
        autoImageIndexStartedForJob = currentJob;
        imageIndexStatus.textContent = '未发现可入库图片';
        updateImageIndexButtonState();
      }
    }

    function hasIndexableImages(job) {
      return (job.results || []).some(result => {
        if (result.status !== 'success' || !result.document) return false;
        const document = result.document;
        if (isImageFile(document.source || document.source_name || '')) return true;
        return (document.assets || []).some(asset => asset.path && isImageFile(asset.path));
      });
    }

    function isImageFile(path) {
      return /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(String(path || ''));
    }

    async function startImageIndexing(autoStarted = false) {
      if (!currentJob) return;
      imageIndexBtn.disabled = true;
      imageIndexStatus.textContent = autoStarted ? '自动图片入库任务创建中' : '图片入库任务创建中';
      try {
        const response = await fetch(`/api/image-index/jobs/${currentJob}`, { method: 'POST' });
        const payload = await readJsonResponse(response);
        currentImageIndexJob = payload.image_index_job_id;
        pollImageIndexJob();
      } catch (error) {
        imageIndexStatus.textContent = `图片入库失败：${error.message}`;
        updateImageIndexButtonState();
      }
    }

    async function pollImageIndexJob() {
      if (!currentImageIndexJob) return;
      try {
        const response = await fetch(`/api/image-index/jobs/${currentImageIndexJob}`);
        const job = await readJsonResponse(response);
        const vectorCount = (job.results || []).reduce((total, item) => total + indexResultVectorCount(item), 0);
        imageIndexStatus.textContent = `${job.status} ${job.progress}/${job.total}，已写入 ${vectorCount} 条图片向量`;
        if (job.status === 'queued' || job.status === 'running') {
          setTimeout(pollImageIndexJob, 1200);
        } else {
          renderImageIndexJob(job);
          loadImageIndexStats();
          loadDocuments();
          updateImageIndexButtonState();
        }
      } catch (error) {
        imageIndexStatus.textContent = `读取图片入库状态失败：${error.message}`;
        updateImageIndexButtonState();
      }
    }

    function renderImageIndexJob(job) {
      const results = job.results || [];
      const errors = job.errors || [];
      const totalImages = results.reduce((total, item) => total + (item.image_count || 0), 0);
      const totalVectors = results.reduce((total, item) => total + (item.vector_count || 0), 0);
      content.innerHTML = `
        <div class="summary">
          <div class="metric"><b>${totalImages}</b><span>images</span></div>
          <div class="metric"><b>${totalVectors}</b><span>vectors</span></div>
          <div class="metric"><b>${escapeHtml(results[0]?.dimensions || '')}</b><span>dimensions</span></div>
          <div class="metric"><b>${escapeHtml(results[0]?.model || '')}</b><span>model</span></div>
          <div class="metric"><b>${escapeHtml(job.status)}</b><span>status</span></div>
        </div>
        <h2>图片入库结果</h2>
        ${results.map(result => `<div class="section"><div class="section-head"><span>${escapeHtml(result.collection)}</span><span>${escapeHtml(result.vector_count)} vectors</span></div><pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre></div>`).join('')}
        ${errors.map(error => `<div class="diag error">${escapeHtml(JSON.stringify(error, null, 2))}</div>`).join('')}
      `;
    }

    async function startIndexing(autoStarted = false) {
      if (!currentJob) return;
      indexBtn.disabled = true;
      indexStatus.textContent = autoStarted ? '自动文本入库任务创建中' : '入库任务创建中';
      try {
        const response = await fetch(`/api/index/jobs/${currentJob}`, { method: 'POST' });
        const payload = await readJsonResponse(response);
        currentIndexJob = payload.index_job_id;
        pollIndexJob();
      } catch (error) {
        indexStatus.textContent = `入库失败：${error.message}`;
        updateIndexButtonState();
      }
    }

    async function pollIndexJob() {
      if (!currentIndexJob) return;
      try {
        const response = await fetch(`/api/index/jobs/${currentIndexJob}`);
        const job = await readJsonResponse(response);
        const vectorCount = (job.results || []).reduce((total, item) => total + (item.vector_count || 0), 0);
        indexStatus.textContent = `${job.status} ${job.progress}/${job.total}，已写入 ${vectorCount} 条向量`;
        if (job.status === 'queued' || job.status === 'running') {
          setTimeout(pollIndexJob, 1200);
        } else {
          renderIndexJob(job);
          loadIndexStats();
          loadDocuments();
          updateIndexButtonState();
        }
      } catch (error) {
        indexStatus.textContent = `读取入库状态失败：${error.message}`;
        updateIndexButtonState();
      }
    }

    function renderIndexJob(job) {
      const results = job.results || [];
      const errors = job.errors || [];
      if (results.length) {
        const totalVectors = results.reduce((total, item) => total + indexResultVectorCount(item), 0);
        const firstResult = normaliseIndexResult(results[0]);
        content.innerHTML = `
          <div class="summary">
            <div class="metric"><b>${results.length}</b><span>documents</span></div>
            <div class="metric"><b>${totalVectors}</b><span>vectors</span></div>
            <div class="metric"><b>${escapeHtml(firstResult.dimensions || '')}</b><span>dimensions</span></div>
            <div class="metric"><b>${escapeHtml(firstResult.model || '')}</b><span>model</span></div>
            <div class="metric"><b>${escapeHtml(job.status)}</b><span>status</span></div>
          </div>
          <h2>入库结果</h2>
          ${results.map(result => {
            const normalised = normaliseIndexResult(result);
            return `<div class="section"><div class="section-head"><span>${escapeHtml(normalised.source_name || 'reindex')}</span><span>${escapeHtml(normalised.document_id || '')}</span></div><pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre></div>`;
          }).join('')}
          ${errors.map(error => `<div class="diag error">${escapeHtml(JSON.stringify(error, null, 2))}</div>`).join('')}
        `;
      } else if (errors.length) {
        content.innerHTML = errors.map(error => `<div class="diag error">${escapeHtml(JSON.stringify(error, null, 2))}</div>`).join('');
      }
    }

    function normaliseIndexResult(result) {
      return result?.text_index || result || {};
    }

    function indexResultVectorCount(result) {
      return Number(result?.vector_count || result?.text_index?.vector_count || 0);
    }

    async function loadIndexStats() {
      indexStats.textContent = '索引统计读取中';
      try {
        const response = await fetch('/api/index/stats');
        const payload = await readJsonResponse(response);
        indexStats.textContent = `文档 ${payload.document_count} 个 / chunks ${payload.chunk_count} 个 / 向量 ${payload.vector_count} 条 / BM25 ${payload.bm25_chunk_count || 0} 条 / Sparse ${payload.sparse_chunk_count || 0} 条 / ${payload.model} ${payload.dimensions}维`;
      } catch (error) {
        indexStats.textContent = `索引统计读取失败：${error.message}`;
      }
    }

    async function loadImageIndexStats() {
      imageIndexStats.textContent = '图片索引统计读取中';
      try {
        const response = await fetch('/api/image-index/stats');
        const payload = await readJsonResponse(response);
        imageIndexStats.textContent = `图片 ${payload.image_count} 张 / 向量 ${payload.vector_count} 条 / ${payload.model} ${payload.dimensions}维`;
      } catch (error) {
        imageIndexStats.textContent = `图片索引统计读取失败：${error.message}`;
      }
    }

    async function loadDocuments() {
      documentListStatus.textContent = '读取文档列表中';
      try {
        const response = await fetch('/api/documents');
        const payload = await readJsonResponse(response);
        documentsCache = payload.documents || [];
        documentListStatus.textContent = `文档 ${documentsCache.length} 个`;
        renderDocumentList(documentsCache);
        if (activeDocumentId && documentsCache.some(doc => doc.document_id === activeDocumentId)) {
          loadDocumentDetail(activeDocumentId);
        }
      } catch (error) {
        documentListStatus.textContent = `读取失败：${error.message}`;
        documentList.innerHTML = `<div class="diag error">${escapeHtml(error.message)}</div>`;
      }
    }

    function renderDocumentList(documents) {
      const query = (documentFilterInput.value || '').trim().toLowerCase();
      const filtered = documents.filter(document => {
        const haystack = [
          document.source_name,
          document.file_type,
          document.status,
          document.parse_status,
          document.index_status,
          document.image_index_status,
          document.document_id
        ].join(' ').toLowerCase();
        return !query || haystack.includes(query);
      });
      if (!filtered.length) {
        documentList.innerHTML = '<div class="diag">没有匹配的文档</div>';
        return;
      }
      documentList.innerHTML = filtered.map(document => `
        <button class="document-row ${document.document_id === activeDocumentId ? 'active' : ''}" data-document-id="${escapeHtml(document.document_id)}">
          <b>${escapeHtml(document.source_name || document.document_id)}</b>
          <span>${renderStatusBadge(document.enabled ? document.status : 'disabled', document.enabled)}</span>
          <span class="document-meta">
            <span>${escapeHtml(document.file_type || '-')}</span>
            <span>${formatBytes(Number(document.file_size || 0))}</span>
            <span>chunks ${escapeHtml(document.chunk_count || 0)}</span>
            <span>vectors ${escapeHtml(document.vector_count || 0)}</span>
            <span>images ${escapeHtml(document.image_vector_count || 0)}</span>
          </span>
        </button>
      `).join('');
      documentList.querySelectorAll('.document-row').forEach(row => {
        row.addEventListener('click', () => loadDocumentDetail(row.dataset.documentId));
      });
    }

    function renderStatusBadge(status, enabled = true) {
      if (!enabled || status === 'disabled') return '<span class="badge off">disabled</span>';
      if (status === 'indexed' || status === 'success') return `<span class="badge ok">${escapeHtml(status)}</span>`;
      if (status === 'failed') return `<span class="badge error">${escapeHtml(status)}</span>`;
      return `<span class="badge warn">${escapeHtml(status || 'pending')}</span>`;
    }

    async function loadDocumentDetail(documentId) {
      if (!documentId) return;
      activeDocumentId = documentId;
      renderDocumentList(documentsCache);
      documentDetail.innerHTML = '<div class="diag">读取文档详情中</div>';
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(documentId)}`);
        const payload = await readJsonResponse(response);
        renderDocumentDetail(payload);
      } catch (error) {
        documentDetail.innerHTML = `<div class="diag error">读取详情失败：${escapeHtml(error.message)}</div>`;
      }
    }

    function renderDocumentDetail(payload) {
      const document = payload.document || {};
      const chunks = payload.chunks || [];
      const images = payload.images || [];
      const counts = payload.counts || {};
      documentDetail.innerHTML = `
        <div class="document-actions">
          <button class="secondary" data-doc-action="toggle" data-enabled="${document.enabled ? '0' : '1'}">${document.enabled ? '禁用检索' : '启用检索'}</button>
          <button class="secondary" data-doc-action="reparse">重新解析</button>
          <button class="secondary" data-doc-action="reindex">重新入库</button>
          <button class="danger" data-doc-action="delete">删除文档</button>
        </div>
        <div class="summary">
          <div class="metric"><b>${escapeHtml(document.chunk_count || 0)}</b><span>chunks</span></div>
          <div class="metric"><b>${escapeHtml(document.vector_count || 0)}</b><span>text vectors</span></div>
          <div class="metric"><b>${escapeHtml(counts.bm25_count || document.bm25_count || 0)}</b><span>BM25</span></div>
          <div class="metric"><b>${escapeHtml(counts.sparse_count || document.sparse_count || 0)}</b><span>sparse</span></div>
          <div class="metric"><b>${escapeHtml(document.image_vector_count || images.length || 0)}</b><span>image vectors</span></div>
        </div>
        <div class="section">
          <div class="section-head"><span>${escapeHtml(document.source_name || '')}</span><span>${renderStatusBadge(document.enabled ? document.status : 'disabled', document.enabled)}</span></div>
          <pre>${escapeHtml(JSON.stringify(document, null, 2))}</pre>
        </div>
        <h2>Chunks</h2>
        ${chunks.slice(0, 30).map((chunk, index) => `
          <div class="section">
            <div class="section-head"><span>#${index + 1} ${escapeHtml(chunk.metadata?.chunk_id || chunk.id)}</span><span>${escapeHtml(chunk.metadata?.heading_path || chunk.metadata?.source_name || '')}</span></div>
            <pre>${escapeHtml(chunk.text || '')}</pre>
          </div>
        `).join('') || '<div class="diag">没有可展示的 chunks</div>'}
        <h2>Images</h2>
        ${images.length ? `<div class="image-grid">${images.map(renderManagedImage).join('')}</div>` : '<div class="diag">没有已入库图片</div>'}
      `;
    }

    function renderManagedImage(image) {
      const metadata = image.metadata || {};
      return `
        <div class="image-card">
          ${image.image_data_url ? `<img src="${image.image_data_url}" alt="">` : ''}
          <div><b>${escapeHtml(metadata.source_name || image.id)}</b><br>${escapeHtml(metadata.origin || '')}<br>${escapeHtml(metadata.image_id || image.id)}</div>
        </div>
      `;
    }

    async function handleDocumentAction(event) {
      const button = event.target.closest('[data-doc-action]');
      if (!button || !activeDocumentId) return;
      const action = button.dataset.docAction;
      if (action === 'toggle') {
        await updateDocumentEnabled(button.dataset.enabled === '1');
      } else if (action === 'delete') {
        await deleteActiveDocument();
      } else if (action === 'reparse') {
        await reparseActiveDocument();
      } else if (action === 'reindex') {
        await reindexActiveDocument();
      }
    }

    async function updateDocumentEnabled(enabled) {
      documentListStatus.textContent = enabled ? '启用文档中' : '禁用文档中';
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(activeDocumentId)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled })
        });
        await readJsonResponse(response);
        await loadDocuments();
        await loadDocumentDetail(activeDocumentId);
      } catch (error) {
        documentListStatus.textContent = `更新失败：${error.message}`;
      }
    }

    async function deleteActiveDocument() {
      if (!window.confirm('确认删除这个文档及其所有索引吗？')) return;
      documentListStatus.textContent = '删除文档中';
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(activeDocumentId)}`, { method: 'DELETE' });
        await readJsonResponse(response);
        activeDocumentId = null;
        documentDetail.innerHTML = '<div class="diag">文档已删除</div>';
        await Promise.all([loadDocuments(), loadIndexStats(), loadImageIndexStats()]);
      } catch (error) {
        documentListStatus.textContent = `删除失败：${error.message}`;
      }
    }

    async function reparseActiveDocument() {
      documentListStatus.textContent = '创建重新解析任务中';
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(activeDocumentId)}/reparse`, { method: 'POST' });
        const payload = await readJsonResponse(response);
        currentJob = payload.job_id;
        currentJobPayload = null;
        switchPage('docs');
        pollJob();
      } catch (error) {
        documentListStatus.textContent = `重新解析失败：${error.message}`;
      }
    }

    async function reindexActiveDocument() {
      documentListStatus.textContent = '创建重新入库任务中';
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(activeDocumentId)}/reindex`, { method: 'POST' });
        const payload = await readJsonResponse(response);
        currentIndexJob = payload.index_job_id;
        switchPage('docs');
        pollIndexJob();
      } catch (error) {
        documentListStatus.textContent = `重新入库失败：${error.message}`;
      }
    }

    async function searchImages() {
      const file = imageSearchInput.files && imageSearchInput.files[0];
      if (!file) {
        imageSearchResults.innerHTML = '<div class="diag">请先选择查询图片</div>';
        return;
      }
      imageSearchResults.innerHTML = '<div class="diag">图片搜索中</div>';
      const form = new FormData();
      form.append('file', file);
      try {
        const response = await fetch('/api/image-search?top_k=6', { method: 'POST', body: form });
        const payload = await readJsonResponse(response);
        renderImageSearchResults(payload.results || []);
      } catch (error) {
        imageSearchResults.innerHTML = `<div class="diag error">图片搜索失败：${escapeHtml(error.message)}</div>`;
      }
    }

    function renderImageSearchResults(results) {
      if (!results.length) {
        imageSearchResults.innerHTML = '<div class="diag">没有搜到相似图片</div>';
        return;
      }
      const cards = results.map(result => `
        <div class="image-card">
          ${result.image_data_url ? `<img src="${result.image_data_url}" alt="">` : ''}
          <div><b>${escapeHtml(result.metadata?.source_name || result.image_id)}</b><br>score ${escapeHtml(result.score)}<br>${escapeHtml(result.metadata?.origin || '')}</div>
        </div>
      `).join('');
      imageSearchResults.innerHTML = `<div class="image-grid">${cards}</div>`;
      content.innerHTML = `<h2>以图搜图结果</h2><div class="image-grid">${cards}</div>`;
    }

    async function loadVoiceStatus() {
      try {
        const response = await fetch('/api/voice/status');
        const payload = await readJsonResponse(response);
        voiceStatusPayload = payload;
        const asrRuntime = describeAsrRuntime(payload);
        voiceMicBtn.disabled = !payload.stt_available;
        autoSpeakBtn.disabled = !payload.reply_audio_available;
        const wakeupWord = payload.wakeup_word || '小绿小绿';
        wakeupLabel.textContent = `唤醒“${wakeupWord}”`;
        const wakeupVadReady = payload.vad_enabled !== false;
        wakeupToggle.disabled = !payload.wakeup_available || !wakeupVadReady;
        if (!payload.wakeup_available) {
          wakeupStatus.className = 'wakeup-status error';
          wakeupStatus.textContent = `离线唤醒不可用：${payload.wakeup_error || '模型未就绪'}；仍可点击麦克风提问`;
        } else if (!wakeupVadReady) {
          wakeupStatus.className = 'wakeup-status';
          wakeupStatus.textContent = 'VAD 已关闭，唤醒问答暂停；浏览器唤醒偏好已保留';
        } else if (!wakeupRecorder) {
          wakeupStatus.className = 'wakeup-status';
          wakeupStatus.textContent = wakeupEnabled ? '等待麦克风许可' : '离线唤醒已关闭';
        }
        voiceMicBtn.title = payload.stt_available ? `语音输入 / ${asrRuntime}` : `语音输入不可用：${(payload.errors || []).join('; ')}`;
        autoSpeakBtn.title = `语音回复：${autoSpeakEnabled ? '开' : '关'} / ${payload.reply_model || 'Qwen Audio 3.1'}`;
        if (payload.voice_available) {
          chatStatus.textContent = `语音就绪 / ASR ${asrRuntime} / 回复 ${payload.reply_model || 'Qwen Audio 3.1'}`;
        }
        renderVoiceConfigSummary();
      } catch (error) {
        voiceStatusPayload = null;
        voiceMicBtn.disabled = true;
        autoSpeakBtn.disabled = true;
        voiceMicBtn.title = `语音状态读取失败：${error.message}`;
        autoSpeakBtn.title = `语音状态读取失败：${error.message}`;
        wakeupToggle.disabled = true;
        wakeupStatus.className = 'wakeup-status error';
        wakeupStatus.textContent = `离线唤醒状态读取失败：${error.message}`;
      }
      applyAutoSpeakState();
      if (
        !wakeupAutoStartAttempted &&
        wakeupEnabled &&
        voiceStatusPayload?.wakeup_available &&
        voiceStatusPayload?.vad_enabled !== false
      ) {
        wakeupAutoStartAttempted = true;
        void startWakeupListening();
      }
    }

    function describeAsrRuntime(payload) {
      const provider = String(payload?.asr_provider_active || '');
      const model = String(payload?.asr_model || payload?.stt_model || '').trim();
      const modelText = model ? `（${model}）` : '';
      if (provider === 'dashscope_realtime') return `Qwen Realtime${modelText}`;
      if (provider === 'local_whisper') return `本地 Whisper${modelText}`;
      return 'ASR 不可用';
    }

    function describeTtsRuntime(payload) {
      const provider = String(payload?.tts_provider_active || '');
      const connectionState = String(payload?.tts_connection_state || '');
      const model = String(payload?.tts_model || '').trim();
      const modelText = model ? `（${model}）` : '';
      if (provider === 'dashscope_realtime') {
        const stateText = connectionState === 'ready'
          ? '已就绪'
          : connectionState === 'fallback'
            ? '回退中'
            : connectionState === 'disconnected'
              ? '未连接'
              : '不可用';
        return `DashScope Realtime ${stateText}${modelText}`;
      }
      if (provider === 'edge') {
        if (connectionState === 'fallback' || connectionState === 'disconnected' || model) {
          return `Edge 在线回退${modelText}`;
        }
        return `Edge 在线 TTS ${connectionState === 'ready' ? '已就绪' : '不可用'}`;
      }
      if (provider === 'custom') return '自定义 TTS 已就绪';
      return 'TTS 不可用';
    }

    function applyAutoSpeakState() {
      autoSpeakBtn.classList.toggle('enabled', autoSpeakEnabled);
      autoSpeakBtn.textContent = autoSpeakEnabled ? '♫' : '♪';
      if (voiceStatusPayload?.reply_audio_available) {
        autoSpeakBtn.title = `语音回复：${autoSpeakEnabled ? '开' : '关'} / ${voiceStatusPayload.reply_model || 'Qwen Audio 3.1'} / ${voiceStatusPayload.reply_voice || ''}`;
      }
    }

    function toggleAutoSpeak() {
      if (autoSpeakBtn.disabled) return;
      autoSpeakEnabled = !autoSpeakEnabled;
      localStorage.setItem(AUTO_SPEAK_STORAGE_KEY, autoSpeakEnabled ? '1' : '0');
      if (!autoSpeakEnabled) stopCurrentTTS();
      applyAutoSpeakState();
    }

    function wakeupWebSocketUrl() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${protocol}//${window.location.host}/api/voice/session/stream`;
    }

    function setWakeupStatus(text, state = '') {
      wakeupStatus.className = `wakeup-status ${state}`.trim();
      wakeupStatus.textContent = text;
      displayWakeupPhase = state;
      if (state === 'listening') {
        setDisplayEmotion('listening');
      } else if (state === 'recognizing') {
        setDisplayEmotion('thinking');
      } else if (state === 'error' && !isAnswering &&
                 !isRecordingVoice && !ttsQueueRunPromise) {
        setDisplayEmotion('error', 1800);
      } else {
        settleDisplayEmotion();
      }
    }

    async function openWakeupSocket(onSocket = null) {
      const socket = new WebSocket(wakeupWebSocketUrl());
      onSocket?.(socket);
      socket.binaryType = 'arraybuffer';
      return await new Promise((resolve, reject) => {
        let settled = false;
        const timer = setTimeout(() => {
          if (settled) return;
          settled = true;
          socket.close();
          reject(new Error('离线唤醒连接超时'));
        }, 10000);
        socket.onopen = () => socket.send(JSON.stringify({
          type: 'start',
          sample_rate: 16000,
          channels: 1,
          encoding: 'pcm_s16le',
          end_silence_ms: Number(
            currentSettings?.vad_end_silence_ms ??
            voiceStatusPayload?.vad_end_silence_ms ??
            400
          ),
          wakeup_word: voiceStatusPayload?.wakeup_word || '小绿小绿'
        }));
        socket.onerror = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          reject(new Error('无法连接离线唤醒服务'));
        };
        socket.onmessage = event => {
          let payload;
          try { payload = JSON.parse(event.data); } catch (error) { return; }
          if (payload.type === 'armed' && !settled) {
            settled = true;
            clearTimeout(timer);
            resolve({ socket, payload });
          } else if (payload.type === 'error' && !settled) {
            settled = true;
            clearTimeout(timer);
            socket.close();
            reject(new Error(payload.detail || '离线唤醒启动失败'));
          }
        };
        socket.onclose = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          reject(new Error('离线唤醒连接已关闭'));
        };
      });
    }

    async function startWakeupListening() {
      if (wakeupRecorder || !wakeupEnabled || !voiceStatusPayload?.wakeup_available) return;
      if (voiceStatusPayload?.vad_enabled === false) {
        setWakeupStatus('VAD 已关闭，唤醒问答暂停；浏览器唤醒偏好已保留');
        return;
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        setWakeupStatus('当前浏览器不支持麦克风唤醒', 'error');
        return;
      }
      const startToken = ++wakeupStartToken;
      const isCurrentStart = () => startToken === wakeupStartToken && wakeupEnabled;
      const cancelStart = () => {
        throw new DOMException('离线唤醒启动已取消', 'AbortError');
      };
      let stream = null;
      let audioContext = null;
      let socket = null;
      try {
        setWakeupStatus('正在申请麦克风许可');
        stream = await navigator.mediaDevices.getUserMedia({ audio: voiceAudioConstraints() });
        if (!isCurrentStart()) cancelStart();
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContextClass();
        if (!audioContext.audioWorklet) throw new Error('浏览器不支持 AudioWorklet');
        if (audioContext.state === 'suspended') await audioContext.resume();
        if (!isCurrentStart()) cancelStart();
        await installVadAudioWorklet(audioContext);
        if (!isCurrentStart()) cancelStart();
        const opened = await openWakeupSocket(candidate => {
          socket = candidate;
          wakeupStartingSockets.add(candidate);
        });
        wakeupStartingSockets.delete(socket);
        if (!isCurrentStart()) cancelStart();
        const source = audioContext.createMediaStreamSource(stream);
        const worklet = new AudioWorkletNode(audioContext, 'vad-pcm-processor');
        const mute = audioContext.createGain();
        mute.gain.value = 0;
        source.connect(worklet);
        worklet.connect(mute);
        mute.connect(audioContext.destination);
        const recorder = {
          stream, audioContext, source, worklet, mute, socket,
          suspended: false, stopping: false, partialText: '', messageQueue: Promise.resolve(),
          promptAudio: null, promptController: null, pendingWakeupAction: null,
          pendingWakeupImmediate: false, pendingWakeupForce: false
        };
        wakeupRecorder = recorder;
        worklet.port.onmessage = event => {
          if (recorder.suspended || recorder.stopping || socket.readyState !== WebSocket.OPEN) return;
          socket.send(event.data);
        };
        socket.onmessage = event => {
          recorder.messageQueue = recorder.messageQueue
            .then(() => handleWakeupSocketMessage(recorder, event))
            .catch(error => failWakeupRecorder(recorder, error.message));
        };
        socket.onerror = () => void failWakeupRecorder(recorder, '离线唤醒连接异常');
        socket.onclose = () => {
          if (!recorder.stopping && wakeupRecorder === recorder) {
            void failWakeupRecorder(recorder, '离线唤醒连接已关闭');
          }
        };
        setWakeupStatus(`待唤醒：请说“${opened.payload.keyword || '小绿小绿'}”`, 'armed');
      } catch (error) {
        if (socket) wakeupStartingSockets.delete(socket);
        if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
        if (stream) stream.getTracks().forEach(track => track.stop());
        if (audioContext && audioContext.state !== 'closed') await audioContext.close();
        if (!isCurrentStart() || error.name === 'AbortError') return;
        setWakeupStatus(`离线唤醒启动失败：${error.message}；仍可点击麦克风提问`, 'error');
        wakeupEnabled = false;
        wakeupToggle.checked = false;
        localStorage.setItem(WAKEUP_STORAGE_KEY, '0');
      }
    }

    async function stopWakeupListening(statusText = '离线唤醒已关闭') {
      clearTimeout(wakeupResumeTimer);
      wakeupStartToken += 1;
      interruptInProgress = false;
      for (const socket of wakeupStartingSockets) {
        if (socket.readyState < WebSocket.CLOSING) socket.close();
      }
      wakeupStartingSockets.clear();
      const recorder = wakeupRecorder;
      wakeupRecorder = null;
      if (!recorder) {
        setWakeupStatus(statusText);
        return;
      }
      recorder.stopping = true;
      if (recorder.promptController) recorder.promptController.abort();
      if (recorder.promptAudio) recorder.promptAudio.pause();
      recorder.worklet.port.onmessage = null;
      if (recorder.socket.readyState === WebSocket.OPEN) {
        recorder.socket.send(JSON.stringify({ type: 'stop' }));
      }
      if (recorder.socket.readyState < WebSocket.CLOSING) recorder.socket.close();
      recorder.worklet.disconnect();
      recorder.source.disconnect();
      recorder.mute.disconnect();
      recorder.stream.getTracks().forEach(track => track.stop());
      if (recorder.audioContext.state !== 'closed') await recorder.audioContext.close();
      setWakeupStatus(statusText);
    }

    async function failWakeupRecorder(recorder, detail) {
      if (wakeupRecorder !== recorder || recorder.stopping) return;
      await stopWakeupListening(`${detail}；仍可点击麦克风提问`);
      setWakeupStatus(`${detail}；仍可点击麦克风提问`, 'error');
    }

    async function toggleWakeupListening() {
      wakeupEnabled = wakeupToggle.checked;
      localStorage.setItem(WAKEUP_STORAGE_KEY, wakeupEnabled ? '1' : '0');
      if (wakeupEnabled) {
        await startWakeupListening();
      } else {
        await stopWakeupListening();
      }
    }

    function suspendWakeupCapture(statusText = '问答处理中，唤醒已暂停') {
      clearTimeout(wakeupResumeTimer);
      if (!wakeupRecorder) return;
      wakeupRecorder.suspended = true;
      if (wakeupRecorder.socket.readyState === WebSocket.OPEN) {
        wakeupRecorder.socket.send(JSON.stringify({ type: 'pause' }));
      }
      setWakeupStatus(statusText);
    }

    // During answer generation and TTS the server-side KWS monitor must keep
    // receiving PCM.  Only the question/VAD and acknowledgement phases are
    // locally suspended.
    function keepWakeupCaptureActive(statusText = '回答处理中，唤醒监听中') {
      if (!wakeupRecorder || wakeupRecorder.stopping) return;
      wakeupRecorder.suspended = false;
      setWakeupStatus(statusText, 'answering');
    }

    function scheduleWakeupAction(action, { immediate = false, force = false } = {}) {
      clearTimeout(wakeupResumeTimer);
      if (!wakeupEnabled || !wakeupRecorder) return;
      wakeupRecorder.pendingWakeupAction = action;
      wakeupRecorder.pendingWakeupImmediate = Boolean(immediate);
      wakeupRecorder.pendingWakeupForce = Boolean(force);
      const resumeWhenIdle = () => {
        if (!wakeupRecorder || !wakeupEnabled) return;
        if (
          !wakeupRecorder.pendingWakeupForce &&
          (isAnswering || isRecordingVoice || ttsQueueRunPromise || currentVoiceAudio)
        ) {
          wakeupResumeTimer = setTimeout(resumeWhenIdle, 100);
          return;
        }
        const nextAction = wakeupRecorder.pendingWakeupAction;
        if (!nextAction) return;
        const cooldownMs = wakeupRecorder.pendingWakeupImmediate
          ? 0
          : nextAction === 'resume'
          ? Number(voiceStatusPayload?.wakeup_cooldown_ms || 800)
          : 0;
        wakeupResumeTimer = setTimeout(() => {
          if (!wakeupRecorder || !wakeupEnabled) return;
          const actionToSend = wakeupRecorder.pendingWakeupAction;
          if (!actionToSend) return;
          wakeupRecorder.pendingWakeupAction = null;
          wakeupRecorder.pendingWakeupImmediate = false;
          wakeupRecorder.pendingWakeupForce = false;
          if (wakeupRecorder.socket.readyState !== WebSocket.OPEN) {
            void failWakeupRecorder(wakeupRecorder, '离线唤醒连接已关闭');
            return;
          }
          wakeupRecorder.socket.send(JSON.stringify({ type: actionToSend }));
          if (actionToSend === 'continue') {
            wakeupRecorder.suspended = false;
            setWakeupStatus('继续对话：正在聆听问题', 'listening');
          } else {
            // Keep microphone frames paused until the server confirms that it
            // has reset the KWS session and emitted the next `armed` event.
            wakeupRecorder.suspended = true;
            setWakeupStatus('正在返回待唤醒状态', 'armed');
          }
        }, cooldownMs);
      };
      resumeWhenIdle();
    }

    function scheduleWakeupResume(options = {}) {
      scheduleWakeupAction('resume', options);
    }

    function scheduleWakeupContinuation() {
      scheduleWakeupAction('continue');
    }

    function schedulePendingWakeupAction() {
      const recorder = wakeupRecorder;
      const action = recorder?.pendingWakeupAction;
      if (action) {
        scheduleWakeupAction(action, {
          immediate: Boolean(recorder.pendingWakeupImmediate),
          force: Boolean(recorder.pendingWakeupForce)
        });
      }
    }

    async function playWakeupAcknowledgement(recorder, payload) {
      recorder.suspended = true;
      const isInterrupt = payload.reason === 'interrupt';
      const promptText = isInterrupt
        ? '我在'
        : String(payload.prompt_text || '嗨，我是小绿，有什么环保问题随时问我哦！');
      const promptUrl = isInterrupt
        ? '/api/voice/interrupt-acknowledgement'
        : String(payload.prompt_audio_url || '/api/voice/wakeup-acknowledgement');
      const controller = new AbortController();
      const audio = new Audio(promptUrl);
      recorder.promptController = controller;
      recorder.promptAudio = audio;
      const releaseDisplayAudio = trackDisplayAudio(audio, () =>
        wakeupRecorder === recorder && recorder.promptAudio === audio && !recorder.stopping);
      setWakeupStatus(
        isInterrupt ? '打断当前回答：正在播放“我在”' : `已唤醒：${promptText}`,
        'acknowledging'
      );
      try {
        await audio.play();
        if (!audio.ended) {
          await waitForMediaEvent(
            audio,
            'ended',
            controller.signal,
            '唤醒提示音播放失败',
            8000
          );
        }
      } finally {
        if (recorder.promptController === controller) recorder.promptController = null;
        if (recorder.promptAudio === audio) recorder.promptAudio = null;
        audio.pause();
        releaseDisplayAudio();
      }
    }

    async function interruptActiveTurn(recorder, payload) {
      if (wakeupRecorder !== recorder || recorder.stopping || interruptInProgress) return;
      interruptInProgress = true;
      answerGeneration += 1;
      clearTimeout(wakeupResumeTimer);
      if (currentAnswerController) {
        currentAnswerController.abort();
        currentAnswerController = null;
      }
      finishLiveLlmTrace('本轮已打断');
      isAnswering = false;
      if (activeAnswerNode) {
        activeAnswerNode.remove();
        activeAnswerNode = null;
      }
      stopCurrentTTS();
      recorder.pendingWakeupAction = null;
      recorder.pendingWakeupImmediate = false;
      recorder.pendingWakeupForce = false;
      recorder.suspended = true;
      chatSendBtn.disabled = isRecordingVoice;
      chatStatus.textContent = '已打断，正在播放“我在”';
      let promptCompleted = false;
      try {
        await playWakeupAcknowledgement(recorder, {
          ...payload,
          reason: 'interrupt',
          prompt_text: '我在',
          prompt_audio_url: '/api/voice/interrupt-acknowledgement'
        });
        promptCompleted = true;
      } catch (error) {
        if (error.name !== 'AbortError' && !recorder.stopping) {
          setWakeupStatus(`提示音播放失败：${error.message}`, 'error');
        }
      }
      if (wakeupRecorder !== recorder || recorder.stopping) return;
      if (!promptCompleted) {
        // Never open VAD when the acknowledgement was not confirmed as
        // played. Close this protocol instance and re-enter clean armed mode.
        if (recorder.socket.readyState === WebSocket.OPEN) {
          recorder.socket.send(JSON.stringify({ type: 'stop' }));
        }
        await stopWakeupListening('“我在”播放失败，正在恢复待唤醒');
        if (wakeupEnabled) await startWakeupListening();
        return;
      }
      // The server starts a fresh ASR/VAD session only after this marker.  Keep
      // the worklet muted until its `listening` event arrives.
      if (recorder.socket.readyState === WebSocket.OPEN) {
        recorder.socket.send(JSON.stringify({ type: 'prompt_finished' }));
      }
      recorder.suspended = true;
      setWakeupStatus('提示音完成，等待聆听状态', 'interrupt_ack');
    }

    function submitWakeupQuestion(recorder, text) {
      // Do not await the full answer here.  The WebSocket reader must remain
      // free to receive a later wake word while generation/TTS is running.
      chatInput.value = text;
      const answerPromise = sendChatMessage(true);
      void answerPromise.then(submitted => {
        if (submitted || wakeupRecorder !== recorder || recorder.stopping) return;
        chatInput.value = text;
        setWakeupStatus('问题已保留在输入框，请稍后发送', 'error');
        scheduleWakeupResume();
      });
    }

    async function handleWakeupSocketMessage(recorder, event) {
      if (wakeupRecorder !== recorder || recorder.stopping) return;
      let payload;
      try { payload = JSON.parse(event.data); } catch (error) {
        await failWakeupRecorder(recorder, '离线唤醒返回了无效消息');
        return;
      }
      if (payload.type === "armed") {
        interruptInProgress = false;
        recorder.pendingWakeupAction = null;
        recorder.pendingWakeupImmediate = false;
        recorder.pendingWakeupForce = false;
        recorder.suspended = false;
        recorder.partialText = '';
        setWakeupStatus(`待唤醒：请说“${payload.keyword || '小绿小绿'}”`, 'armed');
      } else if (payload.type === "wake_detected") {
        setDisplayEmotion('thinking', 0, true);
        recorder.partialText = '';
        // The server can already be in its outer `armed` phase when a user
        // starts a typed/manual answer.  In that case it cannot label the
        // subsequent wake hit as `reason=interrupt`, but it is still an
        // interruption whenever an answer or TTS playback is active.
        const activeTurn = Boolean(
          payload.reason === 'interrupt' ||
          isAnswering ||
          currentAnswerController ||
          ttsQueueRunPromise ||
          currentVoiceAudio
        );
        if (activeTurn) {
          await interruptActiveTurn(recorder, payload);
          return;
        }
        try {
          await playWakeupAcknowledgement(recorder, payload);
        } catch (error) {
          if (error.name === 'AbortError' || recorder.stopping) return;
          setWakeupStatus(`提示音播放失败：${error.message}；继续聆听问题`, 'error');
        }
        if (wakeupRecorder !== recorder || recorder.stopping) return;
        if (recorder.socket.readyState === WebSocket.OPEN) {
          recorder.socket.send(JSON.stringify({ type: 'prompt_finished' }));
        }
        recorder.suspended = false;
        setWakeupStatus('正在聆听问题', 'listening');
      } else if (payload.type === "listening") {
        interruptInProgress = false;
        recorder.partialText = '';
        chatInput.value = '';
        recorder.suspended = false;
        setWakeupStatus('正在聆听问题', 'listening');
      } else if (payload.type === "asr_fallback") {
        setWakeupStatus('千问 ASR 已断开，正在用本地 Whisper 继续识别', 'listening');
      } else if (payload.type === "speech_end" || payload.type === "forced_end") {
        setWakeupStatus('语音识别中', 'recognizing');
      } else if (payload.type === "partial") {
        const text = String(payload.text || '').trim();
        if (text) {
          recorder.partialText = text;
          chatInput.value = text;
        }
        setWakeupStatus(text ? `实时识别：${text}` : '正在聆听问题', 'listening');
      } else if (payload.type === "final") {
        pendingRetrievalToken = payload.retrieval_token || null;
        // The ASR turn is over; the server now switches to its KWS-only
        // waiting phase while the answer and TTS are running.
        recorder.suspended = false;
        const text = String(payload.text || recorder.partialText || '').trim();
        if (!text) {
          setWakeupStatus('只识别到唤醒词，请重新说“小绿小绿”', 'error');
          scheduleWakeupResume();
          return;
        }
        recorder.partialText = '';
        chatInput.value = text;
        setWakeupStatus(`已识别：${text}`);
        // The former `const submitted = await sendChatMessage(true)` waited
        // for the whole answer and starved this WebSocket's interrupt reader.
        submitWakeupQuestion(recorder, text);
      } else if (payload.type === "no_speech") {
        recorder.suspended = false;
        const text = String(recorder.partialText || '').trim();
        if (text) {
          recorder.partialText = '';
          chatInput.value = text;
          setWakeupStatus(`采用最后识别结果：${text}`);
          submitWakeupQuestion(recorder, text);
          return;
        }
        setWakeupStatus('没有识别到问题，请重新说“小绿小绿”', 'error');
        scheduleWakeupResume();
      } else if (payload.type === "error") {
        await failWakeupRecorder(recorder, payload.detail || '离线唤醒失败');
      }
    }

    async function toggleVoiceRecording() {
      if (isRecordingVoice) {
        await stopVoiceRecording();
      } else {
        await startVoiceRecording();
      }
    }

    async function startVoiceRecording() {
      if (voiceMicBtn.disabled) return;
      suspendWakeupCapture('手动语音输入中，唤醒已暂停');
      stopCurrentTTS();
      if (!navigator.mediaDevices?.getUserMedia) {
        chatStatus.textContent = '当前浏览器不支持麦克风录音';
        setDisplayEmotion('error', 1800);
        scheduleWakeupResume();
        return;
      }
      const supportsStreaming = Boolean(
        voiceStatusPayload?.streaming_stt_available &&
        window.AudioWorkletNode &&
        window.WebSocket
      );
      if (supportsStreaming) {
        try {
          await startStreamingVoiceRecording();
          return;
        } catch (error) {
          await startBatchVoiceRecording(`流式识别启动失败，已回退到整段录音：${error.message}`);
          return;
        }
      }
      await startBatchVoiceRecording('流式识别不可用，已回退到整段录音');
    }

    function voiceAudioConstraints() {
      return {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      };
    }

    function setVoiceRecordingUi(active, statusText) {
      isRecordingVoice = active;
      displayAsrPending = statusText === '语音识别中';
      voiceMicBtn.classList.toggle('recording', active);
      voiceMicBtn.textContent = active ? '■' : '◉';
      chatSendBtn.disabled = active || isAnswering;
      if (statusText) chatStatus.textContent = statusText;
      if (active) setDisplayEmotion('listening');
      else if (displayAsrPending) setDisplayEmotion('thinking');
      else settleDisplayEmotion();
    }

    function vadWebSocketUrl() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${protocol}//${window.location.host}/api/voice/stt/stream`;
    }

    async function openVadSocket() {
      const socket = new WebSocket(vadWebSocketUrl());
      socket.binaryType = 'arraybuffer';
      const endSilenceMs = Number(
        currentSettings?.vad_end_silence_ms || voiceStatusPayload?.vad_end_silence_ms || 400
      );
      return await new Promise((resolve, reject) => {
        let settled = false;
        const timer = setTimeout(() => {
          if (settled) return;
          settled = true;
          socket.close();
          reject(new Error('实时 ASR 连接超时'));
        }, 5000);
        socket.onopen = () => socket.send(JSON.stringify({
          type: 'start',
          sample_rate: 16000,
          channels: 1,
          encoding: 'pcm_s16le',
          end_silence_ms: endSilenceMs
        }));
        socket.onerror = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          reject(new Error('无法连接实时 ASR'));
        };
        socket.onmessage = event => {
          let payload;
          try {
            payload = JSON.parse(event.data);
          } catch (error) {
            return;
          }
          if (payload.type === 'ready' && !settled) {
            settled = true;
            clearTimeout(timer);
            resolve(socket);
          } else if (payload.type === 'error' && !settled) {
            settled = true;
            clearTimeout(timer);
            socket.close();
            reject(new Error(payload.detail || '实时 ASR 启动失败'));
          }
        };
        socket.onclose = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          reject(new Error('实时 ASR 连接已关闭'));
        };
      });
    }

    async function installVadAudioWorklet(audioContext) {
      const workletSource = `
        class VadPcmProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this.ratio = sampleRate / 16000;
            this.sourceSamples = [];
            this.readPosition = 0;
            this.frameSamples = [];
          }
          process(inputs) {
            const input = inputs[0] && inputs[0][0];
            if (!input || !input.length) return true;
            for (let index = 0; index < input.length; index += 1) {
              this.sourceSamples.push(input[index]);
            }
            while (this.readPosition + 1 < this.sourceSamples.length) {
              const leftIndex = Math.floor(this.readPosition);
              const fraction = this.readPosition - leftIndex;
              const left = this.sourceSamples[leftIndex];
              const right = this.sourceSamples[leftIndex + 1];
              this.frameSamples.push(left + (right - left) * fraction);
              this.readPosition += this.ratio;
              if (this.frameSamples.length === 1600) {
                const pcm = new Int16Array(1600);
                for (let cursor = 0; cursor < 1600; cursor += 1) {
                  const value = Math.max(-1, Math.min(1, this.frameSamples[cursor]));
                  pcm[cursor] = Math.round(value < 0 ? value * 32768 : value * 32767);
                }
                this.port.postMessage(pcm.buffer, [pcm.buffer]);
                this.frameSamples = [];
              }
            }
            const consumed = Math.min(
              Math.max(0, Math.floor(this.readPosition)),
              Math.max(0, this.sourceSamples.length - 1)
            );
            if (consumed) {
              this.sourceSamples.splice(0, consumed);
              this.readPosition -= consumed;
            }
            return true;
          }
        }
        registerProcessor('vad-pcm-processor', VadPcmProcessor);
      `;
      const workletUrl = URL.createObjectURL(new Blob([workletSource], { type: 'text/javascript' }));
      try {
        await audioContext.audioWorklet.addModule(workletUrl);
      } finally {
        URL.revokeObjectURL(workletUrl);
      }
    }

    async function startStreamingVoiceRecording() {
      let stream = null;
      let audioContext = null;
      let socket = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: voiceAudioConstraints() });
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContextClass();
        if (!audioContext.audioWorklet) throw new Error('浏览器不支持 AudioWorklet');
        socket = await openVadSocket();
        if (socket.readyState !== WebSocket.OPEN) throw new Error('VAD 连接已关闭');
        await installVadAudioWorklet(audioContext);
        const source = audioContext.createMediaStreamSource(stream);
        const worklet = new AudioWorkletNode(audioContext, 'vad-pcm-processor');
        const mute = audioContext.createGain();
        mute.gain.value = 0;
        source.connect(worklet);
        worklet.connect(mute);
        mute.connect(audioContext.destination);
        const recorder = {
          mode: 'stream', stream, audioContext, source, worklet, mute, socket,
          captureStopped: false, completed: false, messageQueue: Promise.resolve(),
          partialText: '', inputBeforeRecording: chatInput.value
        };
        worklet.port.onmessage = event => {
          if (recorder.captureStopped || socket.readyState !== WebSocket.OPEN) return;
          socket.send(event.data);
        };
        socket.onmessage = event => {
          recorder.messageQueue = recorder.messageQueue
            .then(() => handleVadSocketMessage(recorder, event))
            .catch(() => failStreamingRecorder(recorder, 'VAD 消息处理失败'));
        };
        socket.onerror = () => void failStreamingRecorder(recorder, '流式 VAD 连接异常');
        socket.onclose = () => {
          if (!recorder.completed && voiceRecorder === recorder) {
            failStreamingRecorder(recorder, '流式 VAD 连接已关闭');
          }
        };
        voiceRecorder = recorder;
        setVoiceRecordingUi(true, '等待说话');
        armVoiceStopTimer(10000);
      } catch (error) {
        if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
        if (stream) stream.getTracks().forEach(track => track.stop());
        if (audioContext && audioContext.state !== 'closed') await audioContext.close();
        throw error;
      }
    }

    async function handleVadSocketMessage(recorder, event) {
      if (voiceRecorder !== recorder || recorder.completed) return;
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (error) {
        await failStreamingRecorder(recorder, 'VAD 返回了无效消息');
        return;
      }
      if (payload.type === 'asr_fallback') {
        armVoiceStopTimer(30000);
        chatStatus.textContent = '千问 ASR 已断开，正在用本地 Whisper 继续识别';
      } else if (payload.type === 'speech_start') {
        armVoiceStopTimer(30000);
        chatStatus.textContent = '正在聆听';
      } else if (payload.type === 'partial') {
        const text = (payload.text || '').trim();
        if (!text) return;
        armVoiceStopTimer(30000);
        recorder.partialText = text;
        chatInput.value = text;
        chatInput.classList.add('stt-partial');
        chatStatus.textContent = `实时识别（尚未发送）：${text}`;
      } else if (payload.type === 'speech_end' || payload.type === 'forced_end') {
        voiceMicBtn.disabled = true;
        chatSendBtn.disabled = true;
        chatStatus.textContent = '检测到停顿，识别中';
        await stopStreamingCapture(recorder);
        displayAsrPending = true;
        setDisplayEmotion('thinking');
        chatSendBtn.disabled = true;
      } else if (payload.type === 'final') {
        pendingRetrievalToken = payload.retrieval_token || null;
        recorder.completed = true;
        const text = (payload.text || recorder.partialText || '').trim();
        clearStreamingPartial(recorder, false);
        await finishVoiceRecorder(recorder, true);
        if (!text) {
          restoreStreamingDraft(recorder);
          chatStatus.textContent = '没有识别到内容';
          setDisplayEmotion('idle');
          return;
        }
        chatInput.value = text;
        chatStatus.textContent = `已识别：${text}`;
        await sendChatMessage();
      } else if (payload.type === 'no_speech') {
        recorder.completed = true;
        restoreStreamingDraft(recorder);
        await finishVoiceRecorder(recorder, true);
        chatStatus.textContent = '没有检测到语音';
        setDisplayEmotion('idle');
        scheduleWakeupResume();
      } else if (payload.type === 'error') {
        recorder.completed = true;
        restoreStreamingDraft(recorder);
        await finishVoiceRecorder(recorder, true);
        chatStatus.textContent = `流式语音识别失败：${payload.detail || '未知错误'}；请重新点击麦克风`;
        setDisplayEmotion('error', 1800);
        scheduleWakeupResume();
      }
    }

    function clearStreamingPartial(recorder, restoreDraft) {
      chatInput.classList.remove('stt-partial');
      if (restoreDraft && chatInput.value === recorder.partialText) {
        chatInput.value = recorder.inputBeforeRecording || '';
      }
      recorder.partialText = '';
    }

    function restoreStreamingDraft(recorder) {
      clearStreamingPartial(recorder, true);
    }

    async function stopStreamingCapture(recorder) {
      if (recorder.captureStopped) return;
      clearTimeout(voiceStopTimer);
      recorder.captureStopped = true;
      recorder.worklet.port.onmessage = null;
      recorder.worklet.disconnect();
      recorder.source.disconnect();
      recorder.mute.disconnect();
      recorder.stream.getTracks().forEach(track => track.stop());
      if (recorder.audioContext.state !== 'closed') await recorder.audioContext.close();
      setVoiceRecordingUi(false, null);
    }

    async function finishVoiceRecorder(recorder, closeSocket) {
      await stopStreamingCapture(recorder);
      if (closeSocket && recorder.socket.readyState < WebSocket.CLOSING) recorder.socket.close();
      if (voiceRecorder === recorder) voiceRecorder = null;
      voiceMicBtn.disabled = !voiceStatusPayload?.stt_available;
      chatSendBtn.disabled = isAnswering;
    }

    async function failStreamingRecorder(recorder, detail) {
      if (recorder.completed) return;
      recorder.completed = true;
      restoreStreamingDraft(recorder);
      await finishVoiceRecorder(recorder, true);
      chatStatus.textContent = `${detail}；请重新点击麦克风`;
      setDisplayEmotion('error', 1800);
      scheduleWakeupResume();
    }

    async function startBatchVoiceRecording(fallbackDetail = '') {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: voiceAudioConstraints() });
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        const audioContext = new AudioContextClass();
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        const mute = audioContext.createGain();
        mute.gain.value = 0;
        const chunks = [];
        processor.onaudioprocess = event => {
          chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        };
        source.connect(processor);
        processor.connect(mute);
        mute.connect(audioContext.destination);
        voiceRecorder = {
          mode: 'batch', stream, audioContext, source, processor, mute, chunks,
          sampleRate: audioContext.sampleRate
        };
        const statusText = fallbackDetail
          ? `${fallbackDetail}；整段录音中，再次点击停止`
          : '整段录音中，再次点击停止';
        setVoiceRecordingUi(true, statusText);
        armVoiceStopTimer(30000);
      } catch (error) {
        chatStatus.textContent = fallbackDetail
          ? `${fallbackDetail}；麦克风启动失败：${error.message}`
          : `麦克风启动失败：${error.message}`;
        setDisplayEmotion('error', 1800);
        scheduleWakeupResume();
      }
    }

    function armVoiceStopTimer(delayMs) {
      clearTimeout(voiceStopTimer);
      voiceStopTimer = setTimeout(() => void stopVoiceRecording(), delayMs);
    }

    async function stopVoiceRecording() {
      if (!voiceRecorder) return;
      if (voiceRecorder.mode === 'stream') {
        const recorder = voiceRecorder;
        await stopStreamingCapture(recorder);
        voiceMicBtn.disabled = true;
        chatSendBtn.disabled = true;
        chatStatus.textContent = '语音识别中';
        displayAsrPending = true;
        setDisplayEmotion('thinking');
        if (recorder.socket.readyState === WebSocket.OPEN) {
          recorder.socket.send(JSON.stringify({ type: 'stop' }));
        } else {
          await failStreamingRecorder(recorder, '实时 ASR 连接已关闭');
        }
        return;
      }
      await stopBatchVoiceRecording();
    }

    async function stopBatchVoiceRecording() {
      if (!voiceRecorder || voiceRecorder.mode !== 'batch') return;
      clearTimeout(voiceStopTimer);
      const recorder = voiceRecorder;
      voiceRecorder = null;
      setVoiceRecordingUi(false, '语音识别中');
      try {
        recorder.processor.disconnect();
        recorder.source.disconnect();
        recorder.mute.disconnect();
        recorder.stream.getTracks().forEach(track => track.stop());
        await recorder.audioContext.close();
        const merged = mergeAudioChunks(recorder.chunks);
        if (!merged.length) {
          chatStatus.textContent = '没有检测到语音';
          setDisplayEmotion('idle');
          chatSendBtn.disabled = false;
          scheduleWakeupResume();
          return;
        }
        const wav = encodeWav(downsampleAudio(merged, recorder.sampleRate, 16000), 16000);
        const form = new FormData();
        form.append('audio', new Blob([wav], { type: 'audio/wav' }), 'recording.wav');
        const response = await fetch('/api/voice/stt', { method: 'POST', body: form });
        const payload = await readJsonResponse(response);
        const text = (payload.text || '').trim();
        if (!text) {
          chatStatus.textContent = '没有识别到内容';
          setDisplayEmotion('idle');
          chatSendBtn.disabled = false;
          scheduleWakeupResume();
          return;
        }
        chatInput.value = text;
        chatStatus.textContent = `已识别：${text}`;
        await sendChatMessage();
      } catch (error) {
        chatStatus.textContent = `语音识别失败：${error.message}`;
        setDisplayEmotion('error', 1800);
        chatSendBtn.disabled = false;
        scheduleWakeupResume();
      }
    }

    function mergeAudioChunks(chunks) {
      const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
      const merged = new Float32Array(length);
      let offset = 0;
      chunks.forEach(chunk => {
        merged.set(chunk, offset);
        offset += chunk.length;
      });
      return merged;
    }

    function downsampleAudio(buffer, inputSampleRate, outputSampleRate) {
      if (outputSampleRate === inputSampleRate) return buffer;
      const ratio = inputSampleRate / outputSampleRate;
      const length = Math.round(buffer.length / ratio);
      const output = new Float32Array(length);
      for (let index = 0; index < length; index += 1) {
        const start = Math.floor(index * ratio);
        const end = Math.min(Math.floor((index + 1) * ratio), buffer.length);
        let sum = 0;
        for (let cursor = start; cursor < end; cursor += 1) sum += buffer[cursor];
        output[index] = sum / Math.max(1, end - start);
      }
      return output;
    }

    function encodeWav(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      writeAscii(view, 0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeAscii(view, 8, 'WAVE');
      writeAscii(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeAscii(view, 36, 'data');
      view.setUint32(40, samples.length * 2, true);
      let offset = 44;
      for (let index = 0; index < samples.length; index += 1, offset += 2) {
        const sample = Math.max(-1, Math.min(1, samples[index]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      }
      return buffer;
    }

    function writeAscii(view, offset, value) {
      for (let index = 0; index < value.length; index += 1) {
        view.setUint8(offset + index, value.charCodeAt(index));
      }
    }

    function getVisibleAssistantText(node) {
      const answer = node.querySelector('.assistant-answer-text');
      return (answer?.innerText || answer?.textContent || '').trim();
    }

    function resetSpeakButton(button) {
      if (!button) return;
      button.disabled = !voiceStatusPayload?.tts_available;
      button.textContent = '♪';
    }

    function stopCurrentTTS() {
      if (qwenPlayer) { qwenPlayer.stop(); qwenPlayer = null; }
      ttsPlaybackVersion += 1;
      ttsQueueRunPromise = null;
      ttsSentenceQueue = [];
      ttsActiveItem = null;
      const stalePrefetch = ttsPrefetchState;
      ttsPrefetchState = null;
      if (stalePrefetch) {
        stalePrefetch.controller.abort();
        if (stalePrefetch.objectUrl) URL.revokeObjectURL(stalePrefetch.objectUrl);
      }
      if (currentTtsController) {
        currentTtsController.abort();
        currentTtsController = null;
      }
      if (currentVoiceAudio) {
        if (displayPlayingAudio === currentVoiceAudio) displayPlayingAudio = null;
        currentVoiceAudio.pause();
        currentVoiceAudio = null;
      }
      if (currentTtsObjectUrl) {
        URL.revokeObjectURL(currentTtsObjectUrl);
        currentTtsObjectUrl = null;
      }
      resetSpeakButton(activeSpeakButton);
      activeSpeakButton = null;
      settleDisplayEmotion();
    }

    function wireAssistantVoiceControls(node) {
      const button = node.querySelector('.speak-btn');
      if (!button) return;
      button.disabled = !voiceStatusPayload?.tts_available;
      button.addEventListener('click', () => playTTS(getVisibleAssistantText(node), button));
    }

    function cleanTtsText(text) {
      const cleaned = String(text || '')
        .replace(/\[(?:\s*\d+\s*(?:[,，、-]\s*\d+\s*)*)\]/g, '')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/https?:\/\/\S+/gi, '')
        .replace(/^\s{0,3}(?:#{1,6}|>|[-*+]\s)\s*/gm, '')
        .replace(/[`*_~]/g, '')
        .replace(/\s+/g, ' ')
        .replace(/\s+([，。！？!?、,.;；:：])/g, '$1')
        .trim();
      return /[\p{L}\p{N}]/u.test(cleaned) ? cleaned : '';
    }

    function splitTtsSentences(text, flush = false) {
      const source = String(text || '');
      const sentences = [];
      let sentenceStart = 0;
      for (let index = 0; index < source.length; index += 1) {
        const character = source[index];
        let isBoundary = '。！？!?；;\n'.includes(character);
        if (character === '.') {
          const previous = source[index - 1] || '';
          const next = source[index + 1] || '';
          if (!flush && index === source.length - 1) continue;
          isBoundary = !(/\d/.test(previous) && /\d/.test(next));
        }
        if (!isBoundary) continue;
        let sentenceEnd = index + 1;
        while (sentenceEnd < source.length && /[”’"'）)】》」』]/.test(source[sentenceEnd])) {
          sentenceEnd += 1;
        }
        const sentence = cleanTtsText(source.slice(sentenceStart, sentenceEnd));
        if (sentence) sentences.push(sentence);
        sentenceStart = sentenceEnd;
        index = sentenceEnd - 1;
      }
      const remainder = source.slice(sentenceStart);
      if (flush) {
        const tail = cleanTtsText(remainder);
        if (tail) sentences.push(tail);
        return { sentences, remainder: '' };
      }
      return { sentences, remainder };
    }

    function waitForMediaEvent(target, eventName, signal, errorMessage, timeoutMs = 0) {
      return new Promise((resolve, reject) => {
        let timeoutId = null;
        const cleanup = () => {
          clearTimeout(timeoutId);
          target.removeEventListener(eventName, onSuccess);
          target.removeEventListener('error', onError);
          signal?.removeEventListener('abort', onAbort);
        };
        const onSuccess = () => {
          cleanup();
          resolve();
        };
        const onError = () => {
          cleanup();
          reject(new Error(errorMessage));
        };
        const onAbort = () => {
          cleanup();
          reject(new DOMException('TTS playback aborted', 'AbortError'));
        };
        const onTimeout = () => {
          cleanup();
          reject(new Error(`${errorMessage}（等待超时）`));
        };
        if (signal?.aborted) {
          onAbort();
          return;
        }
        target.addEventListener(eventName, onSuccess, { once: true });
        target.addEventListener('error', onError, { once: true });
        signal?.addEventListener('abort', onAbort, { once: true });
        if (timeoutMs > 0) timeoutId = setTimeout(onTimeout, timeoutMs);
      });
    }

    async function appendTtsAudioChunk(sourceBuffer, chunk, signal) {
      if (!chunk?.byteLength) return;
      const updated = waitForMediaEvent(sourceBuffer, 'updateend', signal, '浏览器未能解码流式 MP3');
      sourceBuffer.appendBuffer(chunk);
      await updated;
    }

    async function playTtsBlob(response, controller, playbackVersion) {
      const audioBlob = await response.blob();
      if (!audioBlob.size) throw new Error('语音服务没有返回音频');
      if (controller.signal.aborted || playbackVersion !== ttsPlaybackVersion) return;
      const objectUrl = URL.createObjectURL(audioBlob);
      currentTtsObjectUrl = objectUrl;
      const audio = new Audio(objectUrl);
      currentVoiceAudio = audio;
      const releaseDisplayAudio = trackDisplayAudio(audio, () =>
        playbackVersion === ttsPlaybackVersion && currentVoiceAudio === audio);
      try {
        await audio.play();
        chatStatus.textContent = '正在流式朗读回答';
        if (!audio.ended) {
          await waitForMediaEvent(audio, 'ended', controller.signal, '音频播放失败');
        }
      } finally {
        if (currentVoiceAudio === audio) currentVoiceAudio = null;
        audio.pause();
        releaseDisplayAudio();
        if (currentTtsObjectUrl === objectUrl) {
          URL.revokeObjectURL(objectUrl);
          currentTtsObjectUrl = null;
        }
      }
    }

    async function playTtsMediaStream(response, controller, playbackVersion) {
      const mediaSource = new MediaSource();
      const objectUrl = URL.createObjectURL(mediaSource);
      currentTtsObjectUrl = objectUrl;
      const audio = new Audio(objectUrl);
      currentVoiceAudio = audio;
      const releaseDisplayAudio = trackDisplayAudio(audio, () =>
        playbackVersion === ttsPlaybackVersion && currentVoiceAudio === audio);
      let reader = null;
      let readerFinished = false;
      try {
        await waitForMediaEvent(mediaSource, 'sourceopen', controller.signal, '无法启动流式音频播放器');
        if (controller.signal.aborted || playbackVersion !== ttsPlaybackVersion) return;

        const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
        reader = response.body.getReader();
        let playbackStarted = false;
        let playPromise = null;
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            readerFinished = true;
            break;
          }
          if (controller.signal.aborted || playbackVersion !== ttsPlaybackVersion) {
            throw new DOMException('TTS playback aborted', 'AbortError');
          }
          await appendTtsAudioChunk(sourceBuffer, value, controller.signal);
          if (!playbackStarted) {
            playbackStarted = true;
            playPromise = audio.play();
            void playPromise.catch(() => {});
            chatStatus.textContent = '正在流式朗读回答';
          }
        }
        if (!playbackStarted) throw new Error('语音服务没有返回音频');
        if (sourceBuffer.updating) {
          await waitForMediaEvent(sourceBuffer, 'updateend', controller.signal, '浏览器未能解码流式 MP3');
        }
        if (mediaSource.readyState === 'open') mediaSource.endOfStream();
        await playPromise;
        if (!audio.ended) {
          await waitForMediaEvent(audio, 'ended', controller.signal, '音频播放失败');
        }
      } finally {
        if (reader && !readerFinished) await reader.cancel().catch(() => {});
        if (currentVoiceAudio === audio) currentVoiceAudio = null;
        audio.pause();
        releaseDisplayAudio();
        if (currentTtsObjectUrl === objectUrl) {
          URL.revokeObjectURL(objectUrl);
          currentTtsObjectUrl = null;
        }
      }
    }

    async function streamAndPlayTtsSentence(text, playbackVersion) {
      const controller = new AbortController();
      currentTtsController = controller;
      try {
        const response = await fetch('/api/voice/tts/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
          signal: controller.signal
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || `流式语音请求失败（${response.status}）`);
        }
        if (playbackVersion !== ttsPlaybackVersion) return;
        const canUseMediaSource = Boolean(
          response.body &&
          window.MediaSource &&
          MediaSource.isTypeSupported('audio/mpeg')
        );
        if (canUseMediaSource) {
          await playTtsMediaStream(response, controller, playbackVersion);
        } else {
          await playTtsBlob(response, controller, playbackVersion);
        }
      } finally {
        if (currentTtsController === controller) currentTtsController = null;
      }
    }

    async function prefetchTtsSentenceBlob(item, state) {
      const response = await fetch('/api/voice/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: item.text }),
        signal: state.controller.signal
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `流式语音预取失败（${response.status}）`);
      }
      const audioBlob = await response.blob();
      if (!audioBlob.size) throw new Error('语音预取没有返回音频');
      if (state.controller.signal.aborted || item.playbackVersion !== ttsPlaybackVersion) {
        throw new DOMException('TTS prefetch aborted', 'AbortError');
      }
      state.objectUrl = URL.createObjectURL(audioBlob);
      return state;
    }

    function maybeStartTtsPrefetch() {
      if (!ttsActiveItem || ttsPrefetchState || !ttsSentenceQueue.length) return;
      const item = ttsSentenceQueue[0];
      if (!item || item.playbackVersion !== ttsPlaybackVersion) return;
      const state = {
        item,
        controller: new AbortController(),
        objectUrl: null,
        promise: null
      };
      ttsPrefetchState = state;
      state.promise = prefetchTtsSentenceBlob(item, state);
      void state.promise.catch(() => {});
    }

    async function playPrefetchedTtsBlob(state, playbackVersion) {
      const objectUrl = state.objectUrl;
      if (!objectUrl) throw new Error('预取音频资源不可用');
      if (playbackVersion !== ttsPlaybackVersion) {
        URL.revokeObjectURL(objectUrl);
        state.objectUrl = null;
        return;
      }
      const controller = new AbortController();
      currentTtsController = controller;
      currentTtsObjectUrl = objectUrl;
      state.objectUrl = null;
      const audio = new Audio(objectUrl);
      currentVoiceAudio = audio;
      const releaseDisplayAudio = trackDisplayAudio(audio, () =>
        playbackVersion === ttsPlaybackVersion && currentVoiceAudio === audio);
      try {
        await audio.play();
        chatStatus.textContent = '正在流式朗读回答';
        if (!audio.ended) {
          await waitForMediaEvent(audio, 'ended', controller.signal, '音频播放失败');
        }
      } finally {
        if (currentTtsController === controller) currentTtsController = null;
        if (currentVoiceAudio === audio) currentVoiceAudio = null;
        audio.pause();
        releaseDisplayAudio();
        if (currentTtsObjectUrl === objectUrl) {
          URL.revokeObjectURL(objectUrl);
          currentTtsObjectUrl = null;
        }
      }
    }

    function enqueueTtsSentences(sentences, playbackVersion = ttsPlaybackVersion) {
      if (sentences.length) keepWakeupCaptureActive('正在朗读，唤醒监听中');
      for (const sentence of sentences) {
        const cleaned = cleanTtsText(sentence);
        if (cleaned) ttsSentenceQueue.push({ text: cleaned, playbackVersion });
      }
      if (!ttsQueueRunPromise && ttsSentenceQueue.length) {
        ttsQueueRunPromise = drainTtsSentenceQueue();
      } else {
        maybeStartTtsPrefetch();
      }
    }

    async function drainTtsSentenceQueue() {
      const startedVersion = ttsPlaybackVersion;
      let finishedNormally = false;
      try {
        while (ttsSentenceQueue.length) {
          const item = ttsSentenceQueue.shift();
          if (!item || item.playbackVersion !== ttsPlaybackVersion) continue;
          ttsActiveItem = item;
          settleDisplayEmotion();
          try {
            const prefetchedState = ttsPrefetchState?.item === item ? ttsPrefetchState : null;
            if (prefetchedState) {
              const readyState = await prefetchedState.promise;
              if (item.playbackVersion !== ttsPlaybackVersion) return;
              if (ttsPrefetchState === prefetchedState) ttsPrefetchState = null;
              maybeStartTtsPrefetch();
              await playPrefetchedTtsBlob(readyState, item.playbackVersion);
            } else {
              maybeStartTtsPrefetch();
              await streamAndPlayTtsSentence(item.text, item.playbackVersion);
            }
          } catch (error) {
            if (error.name === 'AbortError' || item.playbackVersion !== ttsPlaybackVersion) return;
            const detail = `流式语音合成失败：${error.message}`;
            stopCurrentTTS();
            chatStatus.textContent = detail;
            setDisplayEmotion('error', 2000);
            return;
          } finally {
            if (ttsActiveItem === item) ttsActiveItem = null;
          }
        }
        finishedNormally = true;
      } finally {
        if (startedVersion !== ttsPlaybackVersion) return;
        ttsActiveItem = null;
        ttsQueueRunPromise = null;
        if (ttsSentenceQueue.length) {
          ttsQueueRunPromise = drainTtsSentenceQueue();
        } else if (activeSpeakButton) {
          const completedButton = activeSpeakButton;
          activeSpeakButton = null;
          resetSpeakButton(completedButton);
        }
        if (finishedNormally && startedVersion === ttsPlaybackVersion &&
            !ttsSentenceQueue.length && !isAnswering && !isRecordingVoice) {
          setDisplayEmotion('success', 1400);
        }
        schedulePendingWakeupAction();
      }
    }

    function playTTS(text, button = null) {
      const split = splitTtsSentences(text, true);
      if (!split.sentences.length) return;
      stopCurrentTTS();
      if (wakeupRecorder) wakeupRecorder.pendingWakeupAction = 'resume';
      setDisplayEmotion('thinking', 0, true);
      const playbackVersion = ttsPlaybackVersion;
      activeSpeakButton = button;
      if (button) {
        button.disabled = true;
        button.textContent = '...';
      }
      enqueueTtsSentences(split.sentences, playbackVersion);
    }

    function beginLiveLlmTrace(query) {
      liveLlmInputs = [];
      llmTraceStatus.textContent = `正在准备本轮输入：${query}`;
      llmTraceFeed.textContent = '';
      const pending = document.createElement('p');
      pending.className = 'llm-trace-empty';
      pending.textContent = '正在准备参考资料与对话上下文';
      llmTraceFeed.appendChild(pending);
    }

    function appendLiveLlmInput(event) {
      const messages = Array.isArray(event.messages) ? event.messages : [];
      liveLlmInputs.push({
        stage: String(event.stage || 'llm'),
        title: String(event.title || 'LLM 输入'),
        model: String(event.model || ''),
        messages
      });
      renderLiveLlmInputs();
    }

    function renderLiveLlmInputs() {
      if (!liveLlmInputs.length) {
        llmTraceFeed.innerHTML = '<p class="llm-trace-empty">本轮尚无 LLM 输入</p>';
        return;
      }
      llmTraceFeed.innerHTML = liveLlmInputs.map((input, inputIndex) => {
        const messages = input.messages.map((message, messageIndex) => {
          const rawRole = String(message?.role || 'message').toLowerCase();
          const roleClass = ['system', 'user', 'assistant'].includes(rawRole) ? rawRole : 'message';
          return `
            <section class="llm-message ${roleClass}">
              <span class="llm-message-role">${escapeHtml(rawRole)} · ${messageIndex + 1}</span>
              <pre>${escapeHtml(String(message?.content || ''))}</pre>
            </section>
          `;
        }).join('');
        const isLatest = inputIndex === liveLlmInputs.length - 1;
        return `
          <details class="llm-trace-entry"${isLatest ? ' open' : ''}>
            <summary>
              ${inputIndex + 1}. ${escapeHtml(input.title)}
              <span>${escapeHtml(input.model || 'unknown')} / ${input.messages.length} messages</span>
            </summary>
            <div class="llm-message-list">${messages}</div>
          </details>
        `;
      }).join('');
      const latest = liveLlmInputs[liveLlmInputs.length - 1];
      llmTraceStatus.textContent = `${liveLlmInputs.length} 组实际输入 / ${latest.title}`;
      llmTraceFeed.scrollTop = llmTraceFeed.scrollHeight;
    }

    function finishLiveLlmTrace(statusText) {
      const suffix = liveLlmInputs.length ? ` / ${liveLlmInputs.length} 组实际输入` : '';
      llmTraceStatus.textContent = `${statusText}${suffix}`;
    }

    async function sendChatMessage(continuousVoice = false) {
      const query = chatInput.value.trim();
      if (!query || isAnswering || isRecordingVoice || interruptInProgress) return false;
      const retrievalToken = pendingRetrievalToken;
      pendingRetrievalToken = null;
      const continueConversation = continuousVoice === true;
      const turnGeneration = ++answerGeneration;
      const controller = new AbortController();
      const audioEnabled = autoSpeakEnabled;
      let endSession = false;
      let terminal = false;
      let ignored = false;
      let failed = false;
      if (wakeupRecorder) wakeupRecorder.pendingWakeupAction = continueConversation ? 'continue' : 'resume';
      isAnswering = true;
      displayAsrPending = false;
      displayTextResponse = false;
      displayWakeupPhase = '';
      keepWakeupCaptureActive('正在回答，唤醒监听中');
      stopCurrentTTS();
      chatInput.value = '';
      beginLiveLlmTrace(query);
      appendChatMessage('user', escapeHtml(query));
      chatStatus.textContent = '正在检索并生成语音回复';
      chatSendBtn.disabled = true;
      setDisplayEmotion('thinking', 0, true);
      const assistantNode = appendChatMessage('assistant', '<span class="stream-cursor"></span>');
      activeAnswerNode = assistantNode;
      currentAnswerController = controller;
      let streamedText = '';
      let player = null;
      const isCurrentTurn = () => turnGeneration === answerGeneration;
      try {
        if (audioEnabled) {
          await unlockReplyAudio();
          if (!isCurrentTurn()) return false;
          player = new QwenPcmPlayer(qwenAudioContext);
          qwenPlayer = player;
        }
        const response = await fetch('/api/answer/stream', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: 4, audio: audioEnabled, retrieval_token: retrievalToken, conversation_id: voiceConversationId }),
          signal: controller.signal
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || '请求失败');
        }
        await readAnswerStream(response, {
          onLlmInput: event => { if (isCurrentTurn()) appendLiveLlmInput(event); },
          onMeta: event => {
            if (!isCurrentTurn()) return;
            endSession = event.retrieval?.reason === 'end_session';
            if (endSession && wakeupRecorder) wakeupRecorder.pendingWakeupAction = 'resume';
          },
          onAudio: event => {
            if (!isCurrentTurn() || !player) return;
            if (event.format !== 'pcm16' || event.channels !== 1) throw new Error('回复音频格式不匹配');
            player.push(event.audio, event.sample_rate);
            setDisplayEmotion('speaking', 0, true);
          },
          onIgnored: event => {
            if (!isCurrentTurn()) return;
            terminal = true;
            ignored = true;
            assistantNode.remove();
            if (activeAnswerNode === assistantNode) activeAnswerNode = null;
            chatStatus.textContent = event.reason || '继续聆听';
            finishLiveLlmTrace('本地空语音过滤');
          },
          onDelta: text => {
            if (!isCurrentTurn()) return;
            streamedText += text;
            assistantNode.innerHTML = `${formatAnswerText(streamedText)}<span class="stream-cursor"></span>`;
            chatMessages.scrollTop = chatMessages.scrollHeight;
            if (!audioEnabled) { displayTextResponse = true; setDisplayEmotion('speaking'); }
          },
          onFinal: result => {
            if (!isCurrentTurn()) return;
            terminal = true;
            assistantNode.innerHTML = `<div class="assistant-answer-text">${formatAnswerText(result.answer || streamedText)}</div>`;
            finishLiveLlmTrace('本轮完成');
            chatStatus.textContent = audioEnabled ? '语音播放中' : '回答完成';
          },
          onError: detail => { throw new Error(detail || '流式生成失败'); }
        });
        if (!terminal && isCurrentTurn()) throw new Error('回复连接提前结束');
        if (audioEnabled && !ignored && isCurrentTurn() && !player?.frames) throw new Error('回复缺少语音数据');
        if (player && isCurrentTurn()) await player.drain();
        if (isCurrentTurn()) { chatStatus.textContent = '回答完成'; setDisplayEmotion('success', 1400); }
      } catch (error) {
        failed = true;
        controller.abort();
        if (player) player.stop();
        if (isCurrentTurn() && error?.name !== 'AbortError') {
          setDisplayEmotion('error', 2000);
          assistantNode.innerHTML = `<div class="diag error">生成回答失败：${escapeHtml(error.message)}</div>`;
          finishLiveLlmTrace('本轮失败');
          chatStatus.textContent = '生成失败';
        }
      } finally {
        if (qwenPlayer === player) qwenPlayer = null;
        if (isCurrentTurn()) {
          if (currentAnswerController === controller) currentAnswerController = null;
          isAnswering = false;
          chatSendBtn.disabled = false;
          chatInput.focus();
          if (continueConversation && !endSession && !failed) scheduleWakeupContinuation();
          else scheduleWakeupResume();
        }
      }
      return true;
    }

    function appendChatMessage(role, html) {
      if (chatEmpty) chatEmpty.remove();
      const node = document.createElement('div');
      node.className = `chat-message ${role}`;
      node.innerHTML = html;
      chatMessages.appendChild(node);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return node;
    }

    function appendChatAnswer(payload) {
      const citations = payload.citations || [];
      const answerText = payload.answer || '';
      const html = `
        <div class="chat-message-tools"><button class="secondary speak-btn" title="朗读回答">♪</button></div>
        <div class="assistant-answer-text">${formatAnswerText(answerText)}</div>
      `;
      const node = appendChatMessage('assistant', html);
      wireAssistantVoiceControls(node);
    }

    async function readAnswerStream(response, handlers) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === 'llm_input') handlers.onLlmInput?.(event);
          if (event.type === 'meta') handlers.onMeta?.(event);
          if (event.type === 'audio') handlers.onAudio?.(event);
          if (event.type === 'delta') handlers.onDelta?.(event.text || '');
          else if (event.type === 'ignored') handlers.onIgnored?.(event);
          if (event.type === 'final') handlers.onFinal?.(event.result || {});
          if (event.type === 'error') handlers.onError?.(event.detail || '');
        }
      }
      const tail = buffer.trim();
      if (tail) {
        const event = JSON.parse(tail);
        if (event.type === 'llm_input') handlers.onLlmInput?.(event);
        else if (event.type === 'meta') handlers.onMeta?.(event);
        else if (event.type === 'audio') handlers.onAudio?.(event);
        else if (event.type === 'delta') handlers.onDelta?.(event.text || '');
        else if (event.type === 'ignored') handlers.onIgnored?.(event);
        else if (event.type === 'final') handlers.onFinal?.(event.result || {});
        if (event.type === 'error') handlers.onError?.(event.detail || '');
      }
    }

    function formatAnswerText(value) {
      return escapeHtml(value || '')
        .replace(/\n/g, '<br>');
    }

    function renderCitations(citations) {
      if (!citations.length) {
        return '<div class="diag">模型没有引用证据，或当前资料不足。</div>';
      }
      return `<div class="citation-grid">${citations.map(citation => `
        <div class="citation-card">
          <div class="citation-badge">[${escapeHtml(citation.id)}]</div>
          <b>${escapeHtml(citation.source_name || '')}</b>
          <span>${escapeHtml(citation.heading_path || citation.heading || '')}</span>
          <span>${escapeHtml(locationText(citation))}</span>
          <div class="citation-meta">
            <span>${escapeHtml(citation.retrieval_source || 'unknown')}</span>
            <span>RRF ${formatMetric(citation.rrf_score ?? citation.score)}</span>
            <span>match ${formatMetric(citation.matched_token_ratio)}</span>
            <span>routes ${formatMetric(citation.route_count)}</span>
          </div>
          <code>${escapeHtml(citation.chunk_id || '')}</code>
          <p>${escapeHtml(citation.quote || '')}</p>
        </div>
      `).join('')}</div>`;
    }

    function locationText(item) {
      return `page ${item.page || '-'} / section ${item.section_index ?? '-'} / chunk ${item.chunk_index ?? '-'}`;
    }

    function renderAnswerCitations(payload) {
      const citations = payload.citations || [];
      if (citations.length) return renderCitations(citations);
      if (payload.retrieval?.bypassed) return '';
      return '<div class="diag">没有可展示引用，可能是证据不足。</div>';
    }

    function formatMetric(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
      return String(value);
    }

    async function upload() {
      if (!selectedFiles.length) return;
      const form = new FormData();
      selectedFiles.forEach(file => form.append('files', file));
      const chunking = readChunkingConfig();
      form.append('chunk_size', String(chunking.chunk_size));
      form.append('chunk_overlap', String(chunking.chunk_overlap));
      uploadBtn.disabled = true;
      jobStatus.textContent = '上传中';
      const response = await fetch('/api/parse', { method: 'POST', body: form });
      const payload = await readJsonResponse(response);
      currentJob = payload.job_id;
      currentJobPayload = null;
      currentIndexJob = null;
      currentImageIndexJob = null;
      autoTextIndexStartedForJob = null;
      autoImageIndexStartedForJob = null;
      indexStatus.textContent = '等待解析完成';
      imageIndexStatus.textContent = '等待解析完成';
      activeIndex = 0;
      updateIndexButtonState();
      updateImageIndexButtonState();
      pollJob();
    }

    function readChunkingConfig() {
      const chunkSize = Number.parseInt(chunkSizeInput.value || '2400', 10);
      const chunkOverlap = Number.parseInt(chunkOverlapInput.value || '300', 10);
      return {
        chunk_size: Number.isFinite(chunkSize) ? chunkSize : 2400,
        chunk_overlap: Number.isFinite(chunkOverlap) ? chunkOverlap : 300
      };
    }

    async function pollJob() {
      if (!currentJob) return;
      const response = await fetch(`/api/jobs/${currentJob}`);
      const job = await response.json();
      currentJobPayload = job;
      jobStatus.textContent = `${job.status} ${job.progress}/${job.total}`;
      renderJob(job);
      if (job.status === 'queued' || job.status === 'running') {
        setTimeout(pollJob, 1200);
      } else {
        uploadBtn.disabled = selectedFiles.length === 0;
        const successful = (job.results || []).filter(result => result.status === 'success').length;
        const hasImages = hasIndexableImages(job);
        indexStatus.textContent = successful ? `解析完成，可入库 ${successful} 个文档` : '没有可入库的解析结果';
        imageIndexStatus.textContent = successful && hasImages ? `解析完成，可入库图片` : '未发现可入库图片';
        updateIndexButtonState();
        updateImageIndexButtonState();
        maybeStartAutoIndexing();
        loadDocuments();
      }
    }

    function renderJob(job) {
      const results = job.results || [];
      tabs.innerHTML = results.map((result, index) => {
        const doc = result.document || {};
        const label = doc.source_name || result.source || `文件 ${index + 1}`;
        return `<button class="tab ${index === activeIndex ? 'active' : ''}" data-index="${index}">${escapeHtml(label)}</button>`;
      }).join('');
      tabs.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
        activeIndex = Number(tab.dataset.index);
        renderJob(job);
      }));
      if (!results.length) {
        content.innerHTML = `<div class="diag">任务已创建，等待解析结果。</div>`;
        return;
      }
      renderDocument(results[Math.min(activeIndex, results.length - 1)].document, job.errors || []);
    }

    function renderDocument(document, errors) {
      if (!document) {
        content.innerHTML = `<div class="diag error">${escapeHtml(JSON.stringify(errors, null, 2))}</div>`;
        return;
      }
      const sections = document.sections || [];
      const chunks = document.chunks || [];
      const tablesData = document.tables || [];
      const assets = document.assets || [];
      const diagnostics = document.diagnostics || [];
      content.innerHTML = `
        <div class="summary">
          <div class="metric"><b>${sections.length}</b><span>sections</span></div>
          <div class="metric"><b>${chunks.length}</b><span>chunks</span></div>
          <div class="metric"><b>${tablesData.length}</b><span>tables</span></div>
          <div class="metric"><b>${assets.length}</b><span>assets</span></div>
          <div class="metric"><b>${document.cache_hit ? '是' : '否'}</b><span>cache hit</span></div>
          <div class="metric"><b>${escapeHtml(chunks[0]?.metadata?.chunk_size || currentJobPayload?.chunking?.chunk_size || '')}</b><span>chunk size</span></div>
        </div>
        <h2>诊断</h2>
        ${diagnostics.map(d => `<div class="diag ${d.level === 'error' ? 'error' : ''}"><b>${escapeHtml(d.code)}</b><br>${escapeHtml(d.message)}</div>`).join('') || '<div class="diag">无诊断信息</div>'}
        <h2>Sections</h2>
        ${sections.slice(0, 50).map((section, index) => renderSection(index, section)).join('')}
        <h2>Tables</h2>
        ${tablesData.map(renderTable).join('') || '<div class="diag">未提取到表格</div>'}
        <h2>Chunks</h2>
        ${chunks.slice(0, 50).map((chunk, index) => renderChunk(index, chunk)).join('')}
        <h2>Assets</h2>
        ${assets.map(asset => `<div class="section"><div class="section-head"><span>${escapeHtml(asset.kind)}</span><span>${escapeHtml(asset.page || '')}</span></div><pre>${escapeHtml(JSON.stringify(asset, null, 2))}</pre></div>`).join('') || '<div class="diag">未提取到图片资产</div>'}
        <h2>Raw JSON</h2>
        <div class="section"><pre>${escapeHtml(JSON.stringify(document, null, 2))}</pre></div>
      `;
    }

    function renderSection(index, section) {
      const meta = section.metadata || {};
      return `<div class="section"><div class="section-head"><span>#${index + 1} ${escapeHtml(meta.heading_path || meta.element_type || meta.parser || '')}</span><span>${escapeHtml(meta.page_number || meta.page || '')}</span></div><pre>${escapeHtml(section.text || '')}</pre></div>`;
    }

    function renderChunk(index, chunk) {
      return `<div class="section"><div class="section-head"><span>#${index + 1}</span><span>${escapeHtml(chunk.id || '')}</span></div><pre>${escapeHtml(chunk.text || '')}</pre></div>`;
    }

    function renderTable(table) {
      const columns = table.columns || [];
      const rows = table.rows || [];
      return `<table><thead><tr>${columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }

    async function readJsonResponse(response) {
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || '请求失败');
      }
      return payload;
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
    }

    function formatBytes(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }
  </script>
</body>
</html>"""
