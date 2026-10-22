from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    data_dir = LaunchConfiguration("data_dir")
    full_python = LaunchConfiguration("full_python")
    params_file = PathJoinSubstitution(
        [FindPackageShare("voice"), "config", "voice_rag.yaml"]
    )
    default_data_dir = EnvironmentVariable(
        "RAG_DATA_DIR",
        default_value=PathJoinSubstitution(
            [EnvironmentVariable("HOME"), ".local", "share", "green-river"]
        ),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("data_dir", default_value=default_data_dir),
            DeclareLaunchArgument(
                "settings_path",
                default_value=PathJoinSubstitution([data_dir, "settings.json"]),
            ),
            DeclareLaunchArgument(
                "index_dir",
                default_value=PathJoinSubstitution([data_dir, "index"]),
            ),
            DeclareLaunchArgument(
                "parser_cache_dir",
                default_value=PathJoinSubstitution([data_dir, "parser_cache"]),
            ),
            DeclareLaunchArgument(
                "display_state_path",
                default_value=PathJoinSubstitution([data_dir, "runtime", "display_state.json"]),
            ),
            DeclareLaunchArgument("speak_answers", default_value="true"),
            DeclareLaunchArgument("native_motion_script", default_value=PathJoinSubstitution([
                EnvironmentVariable("GREENRIVER_PROJECT_ROOT", default_value=PathJoinSubstitution([
                    EnvironmentVariable("HOME"), "green-river-4.0"])),
                "scripts", "run_sdk_motion.sh",
            ])),
            ExecuteProcess(cmd=["bash", LaunchConfiguration("native_motion_script"), "--takeover"], output="screen"),
            DeclareLaunchArgument("enable_local_audio", default_value="false"),
            DeclareLaunchArgument(
                "full_python",
                default_value=EnvironmentVariable(
                    "RAG_SYSTEM_PYTHON",
                    default_value=PathJoinSubstitution(
                        [
                            EnvironmentVariable("HOME"),
                            "Voice",
                            ".venv-linux",
                            "bin",
                            "python",
                        ]
                    ),
                ),
            ),
            Node(
                package="voice",
                executable="voice_bridge_node",
                name="voice_bridge_node",
                output="screen",
                parameters=[
                    params_file,
                    {"speak_answers": ParameterValue(LaunchConfiguration("speak_answers"), value_type=bool)},
                ],
            ),
            Node(
                package="voice",
                executable="voice_arm_motion_node",
                name="voice_arm_motion_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="voice",
                executable="rag_answer_node",
                name="rag_answer_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "data_dir": data_dir,
                        "settings_path": LaunchConfiguration("settings_path"),
                        "index_dir": LaunchConfiguration("index_dir"),
                    },
                ],
            ),
            Node(
                package="voice",
                executable="rag_knowledge_node",
                name="rag_knowledge_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "data_dir": data_dir,
                        "settings_path": LaunchConfiguration("settings_path"),
                        "index_dir": LaunchConfiguration("index_dir"),
                        "parser_cache_dir": LaunchConfiguration("parser_cache_dir"),
                    },
                ],
            ),
            Node(
                package="voice",
                executable="display_node",
                name="display_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "state_path": LaunchConfiguration("display_state_path"),
                    },
                ],
            ),
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration("enable_local_audio")),
                cmd=[
                    full_python,
                    "-m",
                    "voice.audio_input_node",
                    "--ros-args",
                    "--params-file",
                    params_file,
                ],
                output="screen",
            ),
        ]
    )
