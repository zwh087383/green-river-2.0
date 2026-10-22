from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    data_dir = LaunchConfiguration("data_dir")
    python = EnvironmentVariable("RAG_SYSTEM_PYTHON", default_value="python3")
    config = PathJoinSubstitution([FindPackageShare("voice"), "config", "voice_rag.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("data_dir", default_value=EnvironmentVariable("RAG_DATA_DIR")),
        DeclareLaunchArgument("backend_url", default_value="http://127.0.0.1:8000"),
        DeclareLaunchArgument("input_device", default_value=""),
        DeclareLaunchArgument("output_device", default_value=""),
        DeclareLaunchArgument("wake_source", default_value=EnvironmentVariable("RAG_ROBOT_WAKE_SOURCE", default_value="software")),
        DeclareLaunchArgument("interrupt_source", default_value="hardware"),
        DeclareLaunchArgument("echo_cancel_enabled", default_value="false"),
        DeclareLaunchArgument("wake_serial_port", default_value=EnvironmentVariable("RAG_ROBOT_WAKE_PORT", default_value="/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0004-if00")),
        DeclareLaunchArgument("wake_keyword", default_value="小绿小绿"),
        DeclareLaunchArgument("playback_guard_ms", default_value="300"),
        DeclareLaunchArgument("native_motion_script", default_value=PathJoinSubstitution([
            EnvironmentVariable("GREENRIVER_PROJECT_ROOT", default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"), "green-river-4.0"])),
            "scripts", "run_sdk_motion.sh",
        ])),
        ExecuteProcess(cmd=["bash", LaunchConfiguration("native_motion_script"), "--takeover"], output="screen"),
        Node(package="voice", executable="realtime_voice_gateway", prefix=[python], output="screen", parameters=[{
            "backend_url": LaunchConfiguration("backend_url"),
            "input_device": LaunchConfiguration("input_device"),
            "output_device": LaunchConfiguration("output_device"),
            "wake_source": LaunchConfiguration("wake_source"),
            "interrupt_source": LaunchConfiguration("interrupt_source"),
            "echo_cancel_enabled": LaunchConfiguration("echo_cancel_enabled"),
            "wake_serial_port": LaunchConfiguration("wake_serial_port"),
            "wake_keyword": LaunchConfiguration("wake_keyword"),
            "playback_guard_ms": LaunchConfiguration("playback_guard_ms"),
        }], additional_env={"PULSE_SOURCE": EnvironmentVariable("RAG_ROBOT_PULSE_SOURCE", default_value="alsa_input.usb-iflytek_XFM-DP-V0.0.18_bc001440821449423d0-01.mono-fallback")} ),
        Node(package="voice", executable="display_node", prefix=[python], output="screen", parameters=[config, {
            "state_path": PathJoinSubstitution([data_dir, "runtime", "display_state.json"]),
        }]),
        Node(package="voice", executable="rag_knowledge_node", prefix=[python], output="screen", parameters=[config, {
            "data_dir": data_dir,
            "settings_path": PathJoinSubstitution([data_dir, "settings.json"]),
            "index_dir": PathJoinSubstitution([data_dir, "index"]),
            "parser_cache_dir": PathJoinSubstitution([data_dir, "parser_cache"]),
        }]),
    ])
