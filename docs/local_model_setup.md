# Local model setup

The project keeps all configured offline neural-network weights under `data/models/`.
The downloader uses ModelScope (魔搭) for offline weights. DashScope provides
the online router, embedding, realtime ASR, realtime TTS, and model APIs.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[voice,heavy]"
```

The heavy parser dependencies pin `unstructured==0.18.32` and
`unstructured-inference==1.1.1`. This is the latest compatible pair for the
VAD runtime's NumPy 1.26 and OpenCV 4.11 constraints; newer inference releases
require OpenCV 4.12 or 4.13.

## Download and verify

```powershell
# Download every local model used by the current code.
.\.venv\Scripts\python.exe -m rag_system.model_setup download

# Verify file sizes and SHA-256 digests without accessing the network.
.\.venv\Scripts\python.exe -m rag_system.model_setup verify
```

Components can be managed independently:

```powershell
.\.venv\Scripts\python.exe -m rag_system.model_setup download --component asr
.\.venv\Scripts\python.exe -m rag_system.model_setup download --component vad
.\.venv\Scripts\python.exe -m rag_system.model_setup download --component wakeup
.\.venv\Scripts\python.exe -m rag_system.model_setup download --component docling
```

## Selected models

- Realtime ASR (default): DashScope `qwen3-asr-flash-realtime`, 16 kHz mono
  PCM16 input with streaming partial/final text output.
- Local ASR fallback: `Systran/faster-whisper-base`, multilingual Whisper
  `base`, CPU INT8.
- VAD: `iic/speech_fsmn_vad_zh-cn-16k-common-onnx@v2.0.6`, quantized
  FSMN-VAD ONNX model running on CPU. Its default end-of-speech silence is
  400 ms and can be configured from 200 to 3000 ms in 100 ms steps.
- Wake word: ModelScope
  `pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`, using only
  the INT8 encoder, decoder, joiner, and token table on one CPU thread. The
  default phrase is `小绿小绿`; a configured Chinese phrase is converted to
  the model's tone-aware partial-pinyin tokens without retraining.
- Document layout: `ds4sd/docling-layout-heron`.
- Table extraction: the accurate TableFormer weights from
  `AI-ModelScope/docling-models`.
- OCR: the Chinese ONNX Runtime files from `RapidAI/RapidOCR@v3.9.0`.

The runtime never passes a remote model ID to Faster-Whisper. It resolves an
absolute local directory and loads it with `device="cpu"`, `compute_type="int8"`,
and `local_files_only=True`. Docling receives the local artifact root and an
explicit CPU accelerator configuration.

Optional overrides:

```powershell
$env:RAG_SYSTEM_ASR_MODEL_DIR = "D:\models\faster-whisper-base"
$env:RAG_SYSTEM_VAD_MODEL_DIR = "D:\models\fsmn-vad-onnx"
$env:RAG_SYSTEM_WAKEUP_MODEL_DIR = "D:\models\sherpa-kws-wenetspeech"
$env:RAG_SYSTEM_DOCLING_ARTIFACTS_DIR = "D:\models\docling"
```

## Offline wake word

The Web page keeps one 16 kHz mono PCM16 microphone stream open after the user
grants permission. While it is armed, audio is decoded only by the local
sherpa-onnx keyword spotter and is not sent to DashScope. Detection of
`小绿小绿` opens Qwen Realtime ASR, replays a short local pre-roll so a question
spoken immediately after the wake phrase is not clipped, and then uses Qwen's
400 ms server VAD to finish the utterance.

The persistent endpoint is `WS /api/voice/session/stream`. It moves through
`armed`, `wake_detected`, `listening`, `partial`, and `final` events. KWS is not
fed while ASR/VAD is collecting the user's question. After ASR emits `final`,
the browser keeps forwarding microphone PCM to the local KWS monitor throughout
answer generation and TTS playback; that audio is not forwarded to ASR. A wake
word in this phase aborts the active answer request and queued/current speech,
plays the fixed local `我在` acknowledgement, and opens a fresh listening turn
only after the acknowledgement finishes.

When an uninterrupted voice answer finishes, the browser sends `continue` and
starts the next listening turn. An explicit end-of-session result sends `resume`
and returns to `armed` after its farewell playback. The original manual
microphone endpoint remains available if the KWS runtime or model is absent.

## Streaming ASR

With a DashScope API key configured, the Web voice flow creates a browser-scoped
`qwen3-asr-flash-realtime` WebSocket and streams 16 kHz mono PCM16 chunks to it.
Qwen's server VAD uses the configured silence interval (400 ms by default), and
the UI displays `text + stash` previews without submitting them. Only the
`completed.transcript` result is finalized and sent to chat. The final text is
then normalized to Simplified Chinese and passed through the local correction
glossary.

If the cloud WebSocket cannot be opened, the same browser endpoint falls back
to the local FSMN-VAD plus Faster-Whisper path. The complete-WAV `POST
/api/voice/stt` endpoint also remains a local fallback. Set the ASR provider via
`POST /api/settings/asr` when an explicit local-only mode is needed.

## Streaming TTS

When a DashScope API key is configured, application startup opens and configures
a persistent `qwen3-tts-flash-realtime` WebSocket. The default voice is the
female `Cherry` voice, using commit mode and 24 kHz MP3 output. Consecutive
sentences reuse this connection; Edge-TTS remains the fallback if DashScope
fails before the first audio byte is emitted.

`POST /api/voice/tts/stream` returns the provider audio as chunked
`audio/mpeg`. The Web chat enables auto-speak by default, keeps sentence state
across LLM deltas, starts synthesis as soon as a complete sentence is
available, and plays sentences in order. Browsers with MediaSource MP3 support
play incoming audio chunks immediately; other browsers fall back to completing
one short sentence at a time instead of waiting for the whole answer. Starting
a new question or microphone recording, disabling auto-speak, or a wake-word
interruption cancels the active stream and clears queued speech.
