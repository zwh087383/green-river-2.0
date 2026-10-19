"""
Interactive CLI for Green River Knowledge Base.

Input: type text OR :rec to record voice.
Output: text answer + auto TTS playback.

Usage:
  python -m rag_system              # interactive mode
  python -m rag_system --no-color   # no ANSI colors
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

# 禁掉 chromadb 的 posthog telemetry 报错
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEMP_DIR = Path(tempfile.gettempdir()) / "rag-system-cli"

_USE_COLOR = os.environ.get("NO_COLOR", "").strip() == ""
_AUTO_TTS = True  # auto speak every answer


def _clr(code: int, text: str) -> str:
    return "\033[{}m{}\033[0m".format(code, text) if _USE_COLOR else text


def _dim(t): return _clr(2, t)
def _bold(t): return _clr(1, t)
def _green(t): return _clr(32, t)
def _cyan(t): return _clr(36, t)
def _yellow(t): return _clr(33, t)
def _red(t): return _clr(31, t)


BANNER = _green("""
  +--------------------------------------------------+
  |   Green River Knowledge Base CLI                 |
  |                                                  |
  |   Type question  |  :h help  |  :q quit          |
  +--------------------------------------------------+
""")

HELP_TEXT = """{bold_cmd}  Commands start with {y}:

  {y}:q{y}  {y}:quit{y}        Quit
  {y}:h{y}  {y}:help{y}        This help
  {y}:rec{y}             Record voice -> STT -> ask
  {y}:stt{y} <wav>       Transcribe wav file
  {y}:tts{y} on|off      Toggle auto TTS playback
  {y}:stats{y}           Index statistics
  {y}:docs{y}            List indexed documents
  {y}:search{y} <query>  Search only (no answer)
  {y}:parse{y} <file>    Parse and preview chunks
  {y}:index{y} <file>    Parse and index
  {y}:rm{y} <doc-id>     Remove from index
  {y}:status{y}          API / voice status
  {y}:clear{y}           Clear screen

  Anything without : is treated as a RAG question.
  Answers are spoken aloud by default (:tts off to disable).
""".format(bold_cmd=_bold("Commands"), y=_yellow(":"), y2=_yellow(":"))


# ---- service singletons ----

_settings_store = None
_indexing_service = None
_answer_service = None
_voice_service = None


def _get_settings_store():
    global _settings_store
    if _settings_store is None:
        from rag_system.settings import SettingsStore
        _settings_store = SettingsStore()
    return _settings_store


def _get_indexing():
    global _indexing_service
    if _indexing_service is None:
        from rag_system.indexing.service import IndexingService
        _indexing_service = IndexingService(settings_store=_get_settings_store())
    return _indexing_service


def _get_answer():
    global _answer_service
    if _answer_service is None:
        from rag_system.generation.service import AnswerGenerationService
        _answer_service = AnswerGenerationService(
            indexing_service=_get_indexing(),
            settings_store=_get_settings_store(),
        )
    return _answer_service


def _get_voice():
    global _voice_service
    if _voice_service is None:
        from rag_system.voice.service import VoiceService
        _voice_service = VoiceService(settings_store=_get_settings_store())
    return _voice_service


# ---- TTS playback ----

def _play_audio_async(filepath: str) -> None:
    """Play audio file without blocking the CLI."""
    def _play():
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["powershell", "-c",
                     '(New-Object Media.SoundPlayer "{}").PlaySync()'.format(filepath)],
                    capture_output=True, timeout=30,
                )
            elif sys.platform == "darwin":
                subprocess.run(["afplay", filepath], capture_output=True, timeout=30)
            else:
                # Linux: try multiple common audio players
                _play_linux(filepath)
        except Exception:
            pass

    def _play_linux(filepath: str) -> None:
        """Try available Linux audio players in order of preference."""
        players = [
            # ffplay handles MP3 natively, widely available via ffmpeg
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
            # mpg123 is a lightweight MP3 player
            ["mpg123", "-q", filepath],
            # mpg321 is a drop-in replacement for mpg123
            ["mpg321", "-q", filepath],
            # aplay with sox for systems with ALSA (requires WAV, not MP3 - unlikely for MP3 but kept as last resort)
        ]
        for args in players:
            try:
                subprocess.run(args, capture_output=True, timeout=30)
                return  # success, stop trying
            except Exception:
                continue

    t = threading.Thread(target=_play, daemon=True)
    t.start()


def _speak_text(text: str) -> None:
    """Generate TTS and play asynchronously."""
    if not _AUTO_TTS:
        return
    try:
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        voice = _get_voice()
        mp3 = voice.text_to_speech(text)
        tmp = _TEMP_DIR / "answer.mp3"
        tmp.write_bytes(mp3)
        _play_audio_async(str(tmp))
    except Exception:
        pass  # TTS is non-critical


# ---- voice recording ----

def _record_mic(duration: float = 10.0, sample_rate: int = 16000) -> bytes | None:
    """Record from microphone. Returns WAV bytes or None on failure."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("  " + _yellow("sounddevice not installed. Run: pip install sounddevice"))
        return None

    try:
        print("  " + _dim("Recording... (speak now, max {}s)".format(int(duration))))
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("  " + _dim("Recording finished."))

        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()
    except Exception as exc:
        print("  " + _red("Recording failed: {}".format(exc)))
        return None


def cmd_rec() -> str | None:
    """Record voice, transcribe, return text. Returns None on failure."""
    wav = _record_mic(duration=10.0)
    if wav is None:
        return None
    print("  " + _dim("Transcribing..."))
    try:
        text = _get_voice().speech_to_text(wav, filename="recording.wav")
        print("  " + _green("Recognized") + ": " + _bold(text))
        return text
    except Exception as exc:
        print("  " + _red("STT failed: {}".format(exc)))
        return None


# ---- command handlers ----

def cmd_stats() -> None:
    s = _get_indexing().stats()
    print("  {}: {}  |  {}: {}  |  {}: {}".format(
        _bold("Documents"), s.document_count,
        _bold("Chunks"), s.chunk_count,
        _bold("Vectors"), s.vector_count,
    ))
    print("  {}: {}  ({} dim)".format(_bold("Model"), s.model, s.dimensions))
    if s.bm25_chunk_count:
        print("  {}: {} chunks".format(_bold("BM25"), s.bm25_chunk_count))
    if s.sparse_chunk_count:
        print("  {}: {} chunks".format(_bold("Sparse"), s.sparse_chunk_count))


def cmd_docs() -> None:
    docs = _get_indexing().list_documents()
    if not docs:
        print("  " + _dim("(no documents indexed)"))
        return
    for d in docs:
        icon = _green("[enabled]") if d.enabled else _red("[disabled]")
        print("  {} {}".format(icon, _bold(d.source_name)))
        print("    ID: {}  chunks: {}  vectors: {}".format(
            _dim(d.document_id), d.chunk_count, d.vector_count))
        print("    size: {:,} bytes  type: {}  indexed: {}".format(
            d.file_size, d.file_type, d.indexed_at[:19]))


def cmd_parse(filepath: str) -> None:
    p = Path(filepath).expanduser().resolve()
    if not p.exists():
        print("  " + _red("File not found: {}".format(p)))
        return
    from rag_system.parsing import ParsingService
    svc = ParsingService.with_chunking(backend="auto", enable_cache=True)
    print("  " + _dim("Parsing..."))
    try:
        doc = svc.parse_document(p)
        print("  {}: {} chunks".format(_green("Parsed"), len(doc.chunks)))
        for i, chunk in enumerate(doc.chunks):
            preview = chunk.text[:120].replace("\n", " ")
            print("  [{}] {}...".format(i, _dim(preview)))
    except Exception as exc:
        print("  " + _red("Parse failed: {}".format(exc)))


def cmd_index(filepath: str) -> None:
    p = Path(filepath).expanduser().resolve()
    if not p.exists():
        print("  " + _red("File not found: {}".format(p)))
        return
    from rag_system.parsing import ParsingService
    svc = ParsingService.with_chunking(backend="auto", enable_cache=True)
    print("  " + _dim("Parsing..."))
    try:
        doc = svc.parse_document(p)
        print("  {}: {} chunks, {}...".format(
            _green("Parsed"), len(doc.chunks), _dim("indexing...")))
        result = _get_indexing().index_document(doc)
        print("  {}: {} chunks, {} vectors".format(
            _green("Indexed"), result.chunk_count, result.vector_count))
    except Exception as exc:
        print("  " + _red("Failed: {}".format(exc)))


def cmd_rm(doc_id: str) -> None:
    result = _get_indexing().delete_document(doc_id)
    print("  {}: vectors={} bm25={} sparse={}".format(
        _green("Deleted"), result["text_vectors"],
        result["bm25_chunks"], result["sparse_chunks"]))


def cmd_search(query: str) -> None:
    results = _get_indexing().search(query, top_k=3)
    if not results:
        print("  " + _dim("(no results)"))
        return
    for i, r in enumerate(results):
        src = r.metadata.get("retrieval_source", "?")
        preview = r.text[:200].replace("\n", " ")
        print("  {} {} {}  {}".format(
            _bold("[{}]".format(i + 1)),
            _green("{:.4f}".format(r.score)),
            _dim("({})".format(src)),
            preview))
        print()


def cmd_stt(filepath: str) -> None:
    p = Path(filepath).expanduser().resolve()
    if not p.exists():
        print("  " + _red("File not found: {}".format(p)))
        return
    print("  " + _dim("Transcribing..."))
    try:
        audio = p.read_bytes()
        text = _get_voice().speech_to_text(audio, filename=p.name)
        print("  {}: {}".format(_green("Result"), _bold(text)))
    except Exception as exc:
        print("  " + _red("STT failed: {}".format(exc)))


def cmd_tts_export(text: str, output: str) -> None:
    print("  " + _dim("Generating speech..."))
    try:
        mp3 = _get_voice().text_to_speech(text)
        Path(output).expanduser().resolve().write_bytes(mp3)
        print("  {}: {} ({:,} bytes)".format(_green("Saved"), output, len(mp3)))
    except Exception as exc:
        print("  " + _red("TTS failed: {}".format(exc)))


def cmd_status() -> None:
    settings = _get_settings_store().load()
    voice = _get_voice().status()
    print("  {}".format(_bold("API")))
    print("    DashScope : {}".format(
        _green("OK") if settings.has_api_key else _red("not configured")))
    print("    DeepSeek  : {}".format(
        _green("OK") if settings.has_deepseek_api_key else _red("not configured")))
    print("    Embedding : {} ({}d)".format(
        settings.embedding_model, settings.embedding_dimensions))
    print("    Answer    : {}".format(settings.answer_model))
    print("    Reranker  : {} ({})".format(
        "enabled" if settings.reranker_enabled else "disabled",
        settings.reranker_model))
    print("  {}".format(_bold("Voice")))
    print("    STT       : {}  ({})".format(
        _green("OK") if voice.stt_available else _red("unavailable"),
        voice.stt_model))
    print("    TTS       : {}  ({})".format(
        _green("OK") if voice.tts_available else _red("unavailable"),
        voice.tts_voice))
    print("    Auto-TTS  : {}".format(
        _green("ON") if _AUTO_TTS else _dim("OFF")))


# ---- RAG ask ----

def ask(query: str) -> None:
    svc = _get_answer()
    in_progress = ""
    is_tty = sys.stdout.isatty()
    got_final = False
    try:
        for event in svc.stream_answer_events(query):
            etype = event.get("type", "")
            if etype == "meta":
                identity = event.get("identity")
                if identity:
                    persona = identity.get("persona", "")
                    if persona not in ("general_public", ""):
                        label = identity.get("label", "")
                        print("  {}".format(_dim("[{}]".format(label))))
            elif etype == "delta":
                text = event.get("text") or ""
                if is_tty:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                in_progress += text
            elif etype == "final":
                got_final = True
                result = event.get("result", {})
                answer_text = result.get("answer") or in_progress
                citations = result.get("citations") or []
                if is_tty:
                    print()
                print("  {} {}".format(_bold(_cyan(">")), answer_text))
                if citations:
                    c_str = " ".join("[{}]".format(c.get("id", "")) for c in citations)
                    print("  {}".format(_dim(c_str)))
                # auto TTS playback
                if answer_text.strip():
                    _speak_text(answer_text)
    except KeyboardInterrupt:
        print("\n  " + _yellow("[interrupted]"))
        return
    except Exception as exc:
        print("\n  " + _red("[error] {}".format(exc)))
        # 丢了 final 但已有部分内容，兜底显示
        if in_progress.strip() and not got_final:
            print("  {} {}".format(_bold(_cyan(">")), in_progress))


# ---- REPL ----

def _dispatch_cmd(cmd: str, rest: str, rest2: str) -> bool:
    """Handle a colon command. Returns True if should quit."""
    global _AUTO_TTS
    if cmd in ("q", "quit", "exit"):
        print("  " + _yellow("Goodbye"))
        return True
    elif cmd in ("h", "help"):
        print(HELP_TEXT)
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "docs":
        cmd_docs()
    elif cmd == "clear":
        os.system("cls" if os.name == "nt" else "clear")
        print(BANNER)
    elif cmd == "search":
        if not rest:
            print("  " + _yellow("Usage: :search <query>"))
        else:
            cmd_search(rest)
    elif cmd == "parse":
        if not rest:
            print("  " + _yellow("Usage: :parse <file>"))
        else:
            cmd_parse(rest)
    elif cmd == "index":
        if not rest:
            print("  " + _yellow("Usage: :index <file>"))
        else:
            cmd_index(rest)
    elif cmd == "rm":
        if not rest:
            print("  " + _yellow("Usage: :rm <doc-id>"))
        else:
            cmd_rm(rest)
    elif cmd == "stt":
        if not rest:
            print("  " + _yellow("Usage: :stt <wav-file>"))
        else:
            cmd_stt(rest)
    elif cmd == "tts":
        sub = rest.strip().lower()
        if sub in ("on", "1", "true", "yes"):
            _AUTO_TTS = True
            print("  " + _green("Auto-TTS: ON"))
        elif sub in ("off", "0", "false", "no"):
            _AUTO_TTS = False
            print("  " + _dim("Auto-TTS: OFF"))
        elif rest and rest2:
            cmd_tts_export(rest, rest2)
        else:
            print("  " + _yellow("Usage: :tts on|off   or   :tts <text> <output.mp3>"))
    elif cmd == "rec":
        text = cmd_rec()
        if text:
            ask(text)
    elif cmd == "status":
        cmd_status()
    else:
        print("  {}  Type {} for help".format(
            _red("Unknown command: {}".format(cmd)), _yellow(":h")))
    return False


def repl() -> None:
    print(BANNER)
    print(HELP_TEXT)

    while True:
        try:
            raw = input("\n{}> ".format(_bold("Q")))
        except (EOFError, KeyboardInterrupt):
            print("\n  " + _yellow("Goodbye"))
            break

        line = raw.strip()
        if not line:
            continue

        if line.startswith(":"):
            parts = line[1:].split(maxsplit=2)
            cmd = parts[0].strip().lower()
            rest = parts[1] if len(parts) > 1 else ""
            rest2 = parts[2] if len(parts) > 2 else ""
            if _dispatch_cmd(cmd, rest, rest2):
                break
        else:
            ask(line)


# ---- entry ----

def main() -> None:
    global _USE_COLOR
    if "--no-color" in sys.argv:
        _USE_COLOR = False
        sys.argv.remove("--no-color")

    # 过滤 chromadb/jieba 的 stderr 刷屏
    import io
    _stderr_filter = io.TextIOWrapper(
        sys.stderr.buffer, line_buffering=True,
    )
    _stderr_filter_stash = sys.stderr

    class _FilterStderr:
        def write(self, s):
            if not s:
                return 0
            if any(tag in s for tag in (
                "Number of requested results",
                "Building prefix dict",
                "Loading model from cache",
                "Loading model cost",
                "Prefix dict has been built",
            )):
                return len(s)
            return _stderr_filter_stash.write(s)
        def flush(self):
            _stderr_filter_stash.flush()
    sys.stderr = _FilterStderr()

    src_path = _PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    repl()


if __name__ == "__main__":
    main()
