from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rag_interfaces.srv import SetDisplayMode
from rag_system.display_state import DisplayStateStore
from rag_system.display_emotion import EMOTION_TONES, infer_reply_emotion


class DisplayNode(Node):
    """Persist robot speech events for the local kiosk display page."""

    def __init__(self) -> None:
        super().__init__("display_node")
        self.declare_parameter("state_path", "data/runtime/display_state.json")
        self.declare_parameter("display_topic", "/voice/display_text")
        self.declare_parameter("mode_service", "/display/set_mode")
        self._store = DisplayStateStore(Path(str(self.get_parameter("state_path").value)))
        self._answer_text = ""
        self._query_text = ""
        self._tone = "neutral"
        display_topic = str(self.get_parameter("display_topic").value)
        mode_service = str(self.get_parameter("mode_service").value)
        self.create_subscription(String, display_topic, self._on_display_text, 20)
        self.create_service(SetDisplayMode, mode_service, self._on_set_mode)
        # Always come up in the robot-facing emotion mode after a reboot.
        self._store.set_mode("emotion", reason="display node startup")
        self._store.set_emotion("idle", reason="display node startup", source="ros")
        self.get_logger().info(f"Display node ready: topic={display_topic} service={mode_service}")

    def _on_display_text(self, message: String) -> None:
        text = str(message.data or "").strip()
        if not text:
            return
        if text.startswith("[tone]"):
            tone = text.split("]", 1)[1].strip()
            if tone in EMOTION_TONES:
                self._tone = tone
                snapshot = self._store.snapshot()
                if snapshot.get("source") == "ros" and snapshot["state"] in {"speaking", "success"}:
                    self._store.set_emotion(snapshot["state"], text=snapshot.get("text", ""),
                                            source="ros", tone=tone, reason="reply emotion")
            return
        state = "idle"
        lowered = text.lower()
        if lowered.startswith("[query]"):
            state = "listening"
        elif lowered.startswith("[delta]"):
            state = "speaking"
        elif lowered.startswith("[answer]"):
            state = "success"
        elif lowered.startswith("[error]"):
            state = "error"
        elif lowered.startswith("[interrupt]"):
            state = "listening"
        elif lowered.startswith("[ignored]"):
            state = "idle"
        body = text.split("]", 1)[1].strip() if "]" in text else text
        if lowered.startswith("[query]") or state in {"idle", "listening", "error"}:
            self._answer_text = ""
            self._query_text = body if lowered.startswith("[query]") else ""
            self._tone = "neutral"
        elif lowered.startswith("[delta]"):
            self._answer_text = (self._answer_text + body)[:1200]
        elif lowered.startswith("[answer]"):
            self._answer_text = body[:1200]
        # Robot display uses the same offline cues; it never waits for an API
        # or changes the gateway's PCM/output-drain timing.
        tone = infer_reply_emotion(self._query_text, self._answer_text).tone if state in {"speaking", "success"} else "neutral"
        if tone == "neutral" and state in {"speaking", "success"}:
            tone = self._tone
        try:
            self._store.set_emotion(state, text=body, reason=text.split("]", 1)[0] + "]", source="ros", tone=tone)
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def _on_set_mode(self, request: SetDisplayMode.Request, response: SetDisplayMode.Response):
        try:
            snapshot = self._store.set_mode(request.mode, reason=request.reason)
        except ValueError as exc:
            response.accepted = False
            response.active_mode = self._store.snapshot()["mode"]
            response.message = str(exc)
            return response
        response.accepted = True
        response.active_mode = str(snapshot["mode"])
        response.message = "display mode changed"
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
