"""Legacy voice-event adapter; hardware calls live in the native SDK process."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rag_interfaces.msg import VoiceControl
from rag_system.robot_motion import MotionClient
from voice.runtime import nonempty_session_id


class VoiceArmMotionNode(Node):
    def __init__(self) -> None:
        super().__init__("voice_arm_motion_node")
        self.declare_parameter("enabled", True)
        self.declare_parameter("motion_request_topic", "/voice/arm_motion_request")
        self.declare_parameter("interrupt_topic", "/voice/interrupt")
        self.declare_parameter("control_topic", "/voice/control")
        self.declare_parameter("session_id", "robot")
        self._enabled = bool(self.get_parameter("enabled").value)
        self._session_id = nonempty_session_id(self.get_parameter("session_id").value)
        self._client = MotionClient(report=self.get_logger().warning)
        self.create_subscription(VoiceControl, str(self.get_parameter("motion_request_topic").value),
                                 self._on_motion_request, 20)
        for topic in {str(self.get_parameter(name).value) for name in ("interrupt_topic", "control_topic")}:
            self.create_subscription(VoiceControl, topic, self._on_control, 20)
        self.get_logger().info("Voice gesture adapter ready: native SDK over Unix socket")

    def _matches(self, message: VoiceControl) -> bool:
        return str(message.session_id or self._session_id).strip() == self._session_id

    def _on_motion_request(self, message: VoiceControl) -> None:
        if self._enabled and self._matches(message) and message.command == "play_motion":
            self._client.request(message.request_id, int(message.generation), message.purpose, self._session_id)

    def _on_control(self, message: VoiceControl) -> None:
        if self._matches(message) and message.command in {"interrupt", "return_to_wakeup"}:
            self._client.advance(int(message.generation))

    def destroy_node(self):
        self._client.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoiceArmMotionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
