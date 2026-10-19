# Linux And ROS2 Deployment

The robot runs Ubuntu 22.04 and ROS2 Humble. Its vendor SDK is precompiled in
`~/wheeled_robot_pkg/install`; do not write RAG code into that directory.

The deployable source tree belongs at:

```text
~/Voice
```

`rag_interfaces` provides typed request, response, and stream contracts.
`voice` contains five ROS2 nodes:

| Node | Responsibility |
| --- | --- |
| `voice_bridge_node` | Converts vendor speech recognition results into RAG queries and sends final answers to vendor TTS. |
| `rag_answer_node` | Maintains isolated RAG conversation state per `session_id` and publishes streamed answer events. |
| `rag_knowledge_node` | Exposes parse/index actions plus search, index-stat, and document-enable services. |
| `robot_audio_input_node` | Captures the robot microphone, detects `小绿小绿`, segments speech with FSMN VAD, and publishes local Whisper transcripts. |
| `voice_arm_motion_node` | Serializes voice gestures, publishes SDK `Byte` motion commands, and requests preset recovery after interruption or timeout. |

The complete Python stack is isolated in `~/Voice/.venv-linux`. The generated
ROS2 executables delegate into that environment, so parsing, indexing, ASR, and
TTS all use one dependency set without changing the vendor SDK's Python install.
Colcon installs only the ROS2 interfaces and node wrappers.

The vendor speech interface remains supported:

```text
/speech_interaction/isr_result -> /voice/query -> /voice/answer
                                                -> /speech_interaction/text_to_audio
```

While ASR is transcribing or an answer is being generated or played, the local
audio node keeps the wake-word detector armed. KWS is not fed while FSMN VAD is
collecting the user's question, so the two recognizers never compete for the
same speech. A wake-word hit during a busy phase publishes
`VoiceControl(command=interrupt)` on `/voice/interrupt` with a non-zero
process-incarnation `generation`. The answer and bridge nodes invalidate that generation, send an
empty `TextToAudio(is_play=false)` stop frame, then queue the fixed `我在`
acknowledgement.

The bridge publishes `VoicePlayback(state=started, purpose=interrupt_ack)` for
the acknowledgement. The vendor playback adapter must return a terminal
`completed`, `failed`, or `interrupted` event with the same `request_id`,
`session_id`, `generation`, and `purpose`; only then does the audio node enter
`listening`. If no terminal event arrives before
`interrupt_ack_timeout_seconds`, the bridge publishes `return_to_wakeup` and
the audio node returns to KWS-only `armed` mode without opening question VAD.
Ordinary answer playback has the same bounded fallback through
`playback_timeout_seconds`, so a missing vendor completion callback cannot
permanently block later transcripts.
Late answer events from the cancelled request are dropped by both nodes.

The full launch can instead use the robot microphone directly:

```text
HK-MIC -> 小绿小绿 -> FSMN VAD -> local Whisper -> /voice/query
```

Local audio is disabled by default. On the current robot, PulseAudio's default
source is `HK-MIC`; leave `robot_audio_input_node.device` empty so PulseAudio
owns and resamples the device safely.

## Install

From the development machine, copy this project with:

```bash
./scripts/deploy_voice_to_robot.sh
```

Synchronize the local models once:

```bash
./scripts/deploy_models_to_robot.sh
```

On the robot, install every optional dependency and build the overlay:

```bash
bash ~/Voice/ros2/voice/scripts/install_linux.sh
```

The installer creates `~/Voice/.venv-linux`, verifies `rclpy`, and writes RAG
state beneath `RAG_DATA_DIR` (default: `~/Voice/data`). It does not overwrite
the vendor SDK installation or configure API keys.

Verify the deployment offline:

```bash
cd ~/Voice
.venv-linux/bin/python -m rag_system.model_setup verify
./scripts/run_linux.sh doctor
```

## Run

Set keys outside ROS messages, then start the launch file:

```bash
export DASHSCOPE_API_KEY='...'
export DEEPSEEK_API_KEY='...'
bash ~/Voice/ros2/voice/scripts/run_voice.sh
```

The standard launch starts the three bridge/RAG nodes and continues using the
vendor speech topics. Use `speak_answers:=false` to test without commanding the
vendor TTS node.

Start the full local microphone chain with:

```bash
bash ~/Voice/ros2/voice/scripts/run_full_voice.sh speak_answers:=false
```

This additionally starts `robot_audio_input_node`. It uses local wakeup, VAD,
and Whisper without API keys. Cloud routing, embeddings, answer generation,
and the agent still require valid `DASHSCOPE_API_KEY` and
`DEEPSEEK_API_KEY` values.

## Verify

```bash
source /opt/ros/humble/setup.bash
source ~/wheeled_robot_pkg/install/setup.bash
source ~/wheeled_robot_pkg/sample/install/setup.bash

ros2 node list | grep -E 'voice|rag_'
ros2 interface list | grep rag_interfaces
ros2 topic info /voice/audio_status -v
ros2 topic echo /voice/display_text
ros2 service call /rag/index_stats rag_interfaces/srv/GetIndexStats '{}'
```

The active ROS graph exposes these topics:

```text
/voice/query, /voice/answer, /voice/control, /voice/interrupt,
/voice/arm_motion_request, /voice/playback, /voice/display_text,
/voice/audio_status, /arm_control/motion, /arm_control/arm_state
```

`rag_answer_node` uses the DashScope Qwen routing request to return one of
three control markers before retrieval: `[[IGNORE_TURN]]`, `[[END_SESSION]]`,
or `[[ANSWER]]`. An ignored filler emits `resume_listening` on `/voice/control`;
an explicit end bypasses retrieval and streams a DeepSeek farewell. No separate
turn-gate request is made. `robot_audio_input_node` keeps the microphone
pipeline locked after a normal or closing turn until it receives a matching
terminal `/voice/playback` message. `voice_bridge_node` publishes `started`
when it forwards text to the vendor TTS. The vendor playback adapter must
publish `completed`, `failed`, or `interrupted` with the same `request_id`,
`session_id`, `generation`, and `purpose`. For an interruption acknowledgement,
the adapter must correlate the bridge's `VoicePlayback(started,
purpose="interrupt_ack")` token with the immediately following `TextToAudio`
request and echo that exact token in its terminal event. This repository has no
vendor playback-completion producer and does not modify the vendor-side node;
without that callback, the acknowledgement times out after 8 seconds and
safely returns to wake-word-only `armed` mode instead of opening VAD.

The added fields change the ROSIDL wire types. Rebuild and source
`rag_interfaces` plus every publisher/subscriber that uses these messages as one
deployment; an older generated message package is not wire-compatible with the
new protocol.

The local audio node seeds `generation` from its process incarnation, so a
standalone restart advances beyond generations emitted before the restart.
Delayed controls, answers, ASR results, and playback events with any other
generation are discarded.

```text
Services: /rag/ask, /rag/search, /rag/index_stats, /rag/set_document_enabled
Actions:  /rag/parse_document, /rag/index_document
```

The arm gesture integration, PDF references, manual arm-enable command, and
physical acceptance checklist are recorded in
`docs/voice_arm_motion_integration.md`.

Use `ros2 action send_goal` for document parsing or indexing only after the
cloud/API settings and Python dependencies have been configured.
