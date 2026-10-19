from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    command: str
    package: str
    input: str
    output: str
    network: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


MODULE_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec("wakeup", "rag_system.wakeup", "WAV", "wakeup event JSON", "offline"),
    ModuleSpec("vad", "rag_system.vad", "WAV", "speech segment JSON", "offline"),
    ModuleSpec("asr", "rag_system.asr", "WAV", "recognized text JSON", "provider-dependent"),
    ModuleSpec("tts", "rag_system.tts", "text", "MP3 file + JSON", "online"),
    ModuleSpec("rag-cloud", "rag_system.rag_cloud", "document/ID", "parse/index result JSON", "provider-dependent"),
    ModuleSpec("rag-edge", "rag_system.rag_edge", "query", "local retrieval result JSON", "provider-dependent"),
    ModuleSpec("agent", "rag_system.agent", "question", "answer event JSON", "online"),
)

MODULE_BY_COMMAND = {spec.command: spec for spec in MODULE_SPECS}


__all__ = ["MODULE_BY_COMMAND", "MODULE_SPECS", "ModuleSpec"]
