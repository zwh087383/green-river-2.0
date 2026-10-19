from __future__ import annotations

import importlib.util
import json
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping


DependencyFinder = Callable[[str], object | None]


_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "wakeup": ("numpy", "pypinyin", "sherpa_onnx"),
    "vad": ("funasr_onnx", "numpy", "onnxruntime"),
    "asr-local": ("faster_whisper", "numpy", "opencc"),
    "asr-qwen": ("aiohttp", "opencc"),
    "tts-edge": ("aiohttp", "edge_tts"),
    "tts-qwen": ("aiohttp",),
    "rag-cloud": ("chromadb", "docling", "jieba", "openai", "rank_bm25", "requests"),
    "rag-edge": ("chromadb", "jieba", "openai", "rank_bm25", "requests"),
    "agent": ("chromadb", "jieba", "openai", "rank_bm25", "requests"),
}


def build_doctor_report(
    *,
    project_root: Path | str | None = None,
    dependency_finder: DependencyFinder = importlib.util.find_spec,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a read-only, offline readiness report for customer module tests."""

    root = Path(project_root or _project_root()).expanduser().resolve()
    environment = os.environ if environ is None else environ
    dependencies = _dependency_status(dependency_finder)
    models = _model_status()
    configuration = _configuration_status(root, environment)
    modules = _module_status(dependencies, models, configuration)
    ready_count = sum(bool(status["ready"]) for status in modules.values())

    return {
        "ok": ready_count == len(modules),
        "command": "doctor",
        "offline": True,
        "project": {
            "root": str(root),
            "version": _project_version(),
            "python": platform.python_version(),
            "python_ready": tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10),
        },
        "checks": {
            "dependencies": dependencies,
            "models": models,
            "configuration": configuration,
            "test_data": _test_data_status(root),
        },
        "modules": modules,
        "summary": {
            "ready_modules": ready_count,
            "module_count": len(modules),
        },
    }


def report_exit_code(report: Mapping[str, Any]) -> int:
    return 0 if bool(report.get("ok")) else 1


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _project_version() -> str:
    try:
        return metadata.version("rag-system")
    except metadata.PackageNotFoundError:
        return "unknown"


def _dependency_status(finder: DependencyFinder) -> dict[str, bool]:
    names = sorted({name for values in _DEPENDENCIES.values() for name in values})
    status: dict[str, bool] = {}
    for name in names:
        try:
            status[name] = finder(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            status[name] = False
    return status


def _model_status() -> dict[str, dict[str, Any]]:
    # These helpers inspect file names only. They never load a model, contact a
    # model hub, compute large hashes, or create runtime directories.
    from rag_system.parsing.model_assets import (
        default_docling_artifacts_dir,
        docling_model_files_present,
    )
    from rag_system.voice.models import default_asr_model_dir, asr_model_files_present
    from rag_system.voice.vad_models import default_vad_model_dir, vad_model_files_present
    from rag_system.voice.wakeup_models import (
        default_wakeup_model_dir,
        wakeup_model_files_present,
    )

    probes = {
        "wakeup": (default_wakeup_model_dir, wakeup_model_files_present),
        "vad": (default_vad_model_dir, vad_model_files_present),
        "asr-local": (default_asr_model_dir, asr_model_files_present),
        "docling": (default_docling_artifacts_dir, docling_model_files_present),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, (path_factory, presence_check) in probes.items():
        try:
            path = Path(path_factory()).expanduser().resolve()
            results[name] = {"ready": bool(presence_check(path)), "path": str(path)}
        except Exception as exc:
            results[name] = {
                "ready": False,
                "path": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return results


def _configuration_status(
    root: Path,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    settings_path = root / "data" / "settings.json"
    payload: dict[str, Any] = {}
    error: str | None = None
    if settings_path.is_file():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
            else:
                error = "settings root must be a JSON object"
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            error = str(exc)

    dashscope_source = _secret_source(
        environ.get("DASHSCOPE_API_KEY"), payload.get("dashscope_api_key")
    )
    deepseek_source = _secret_source(
        environ.get("DEEPSEEK_API_KEY"), payload.get("deepseek_api_key")
    )
    result: dict[str, Any] = {
        "settings_file": str(settings_path),
        "settings_file_present": settings_path.is_file(),
        "dashscope_api_key": dashscope_source != "none",
        "dashscope_api_key_source": dashscope_source,
        "deepseek_api_key": deepseek_source != "none",
        "deepseek_api_key_source": deepseek_source,
        "asr_provider": str(payload.get("asr_provider") or "dashscope_realtime"),
    }
    if error:
        result["error"] = error
    return result


def _secret_source(environment_value: object, file_value: object) -> str:
    if str(environment_value or "").strip():
        return "env"
    if str(file_value or "").strip():
        return "file"
    return "none"


def _module_status(
    dependencies: Mapping[str, bool],
    models: Mapping[str, Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    dashscope_ready = bool(configuration.get("dashscope_api_key"))
    deepseek_ready = bool(configuration.get("deepseek_api_key"))

    wakeup = _readiness(
        dependencies,
        _DEPENDENCIES["wakeup"],
        model=("wakeup", models),
    )
    vad = _readiness(
        dependencies,
        _DEPENDENCIES["vad"],
        model=("vad", models),
    )
    asr_local = _readiness(
        dependencies,
        _DEPENDENCIES["asr-local"],
        model=("asr-local", models),
    )
    asr_qwen = _readiness(
        dependencies,
        _DEPENDENCIES["asr-qwen"],
        required_settings=(("dashscope_api_key", dashscope_ready),),
    )
    tts_edge = _readiness(dependencies, _DEPENDENCIES["tts-edge"])
    tts_qwen = _readiness(
        dependencies,
        _DEPENDENCIES["tts-qwen"],
        required_settings=(("dashscope_api_key", dashscope_ready),),
    )
    rag_cloud = _readiness(
        dependencies,
        _DEPENDENCIES["rag-cloud"],
        model=("docling", models),
        required_settings=(("dashscope_api_key", dashscope_ready),),
    )
    rag_edge = _readiness(
        dependencies,
        _DEPENDENCIES["rag-edge"],
        required_settings=(("dashscope_api_key", dashscope_ready),),
    )
    agent = _readiness(
        dependencies,
        _DEPENDENCIES["agent"],
        required_settings=(
            ("dashscope_api_key", dashscope_ready),
            ("deepseek_api_key", deepseek_ready),
        ),
    )

    return {
        "wakeup": wakeup,
        "vad": vad,
        "asr": {
            "ready": bool(asr_local["ready"] or asr_qwen["ready"]),
            "providers": {"local": asr_local, "qwen": asr_qwen},
        },
        "tts": {
            "ready": bool(tts_edge["ready"] or tts_qwen["ready"]),
            "providers": {"edge": tts_edge, "qwen": tts_qwen},
        },
        "rag-cloud": rag_cloud,
        "rag-edge": rag_edge,
        "agent": agent,
    }


def _readiness(
    dependency_status: Mapping[str, bool],
    required_dependencies: tuple[str, ...],
    *,
    model: tuple[str, Mapping[str, Mapping[str, Any]]] | None = None,
    required_settings: tuple[tuple[str, bool], ...] = (),
) -> dict[str, Any]:
    missing = [
        f"dependency:{name}"
        for name in required_dependencies
        if not dependency_status.get(name, False)
    ]
    if model is not None:
        model_name, model_status = model
        if not bool(model_status.get(model_name, {}).get("ready")):
            missing.append(f"model:{model_name}")
    missing.extend(
        f"setting:{name}" for name, available in required_settings if not available
    )
    return {"ready": not missing, "missing": missing}


def _test_data_status(root: Path) -> dict[str, Any]:
    audio_root = root / "samples" / "audio"
    document_root = root / "samples" / "acceptance"
    audio_count = _count_files(audio_root, {".wav"})
    document_count = _count_files(
        document_root,
        {".docx", ".html", ".md", ".pdf", ".pptx", ".txt", ".xlsx"},
    )
    return {
        "ready": audio_count > 0 and document_count > 0,
        "required": False,
        "audio_path": str(audio_root.resolve()),
        "document_path": str(document_root.resolve()),
        "audio_files": audio_count,
        "document_files": document_count,
    }


def _count_files(root: Path, suffixes: set[str]) -> int:
    if not root.is_dir():
        return 0
    try:
        return sum(
            1
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
    except OSError:
        return 0


__all__ = ["build_doctor_report", "report_exit_code"]
