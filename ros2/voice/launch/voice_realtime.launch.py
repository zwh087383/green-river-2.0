from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    data_dir = LaunchConfiguration("data_dir")
    config = PathJoinSubstitution([FindPackageShare("voice"), "config", "voice_rag.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("data_dir", default_value=EnvironmentVariable("RAG_DATA_DIR")),
        DeclareLaunchArgument("backend_url", default_value="http://127.0.0.1:8000"),
        DeclareLaunchArgument("input_device", default_value=""),
        DeclareLaunchArgument("output_device", default_value=""),
        Node(package="voice", executable="realtime_voice_gateway", output="screen", parameters=[{
            "backend_url": LaunchConfiguration("backend_url"),
            "input_device": LaunchConfiguration("input_device"),
            "output_device": LaunchConfiguration("output_device"),
        }]),
        Node(package="voice", executable="display_node", output="screen", parameters=[config, {
            "state_path": PathJoinSubstitution([data_dir, "runtime", "display_state.json"]),
        }]),
        Node(package="voice", executable="voice_arm_motion_node", output="screen", parameters=[config]),
        Node(package="voice", executable="rag_knowledge_node", output="screen", parameters=[config, {
            "data_dir": data_dir,
            "settings_path": PathJoinSubstitution([data_dir, "settings.json"]),
            "index_dir": PathJoinSubstitution([data_dir, "index"]),
            "parser_cache_dir": PathJoinSubstitution([data_dir, "parser_cache"]),
        }]),
    ])
