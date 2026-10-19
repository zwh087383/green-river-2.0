from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
INTERFACES = ROOT / "ros2" / "rag_interfaces"
VOICE = ROOT / "ros2" / "voice"


def test_ros2_package_metadata_has_expected_names() -> None:
    interface_root = ElementTree.parse(INTERFACES / "package.xml").getroot()
    voice_root = ElementTree.parse(VOICE / "package.xml").getroot()

    assert interface_root.findtext("name") == "rag_interfaces"
    assert voice_root.findtext("name") == "voice"
    assert interface_root.findtext("export/build_type") == "ament_cmake"
    assert voice_root.findtext("export/build_type") == "ament_python"


def test_interface_files_cover_voice_and_knowledge_boundaries() -> None:
    expected = {
        "msg/VoiceQuery.msg",
        "msg/VoiceAnswer.msg",
        "msg/VoiceControl.msg",
        "msg/VoicePlayback.msg",
        "srv/Ask.srv",
        "srv/Search.srv",
        "srv/GetIndexStats.srv",
        "srv/SetDocumentEnabled.srv",
        "srv/SetDisplayMode.srv",
        "action/ParseDocument.action",
        "action/IndexDocument.action",
    }
    actual = {
        path.relative_to(INTERFACES).as_posix()
        for path in INTERFACES.rglob("*")
        if path.is_file() and path.suffix in {".msg", ".srv", ".action"}
    }

    assert actual == expected
    query_source = (INTERFACES / "msg" / "VoiceQuery.msg").read_text(encoding="utf-8")
    answer_source = (INTERFACES / "msg" / "VoiceAnswer.msg").read_text(encoding="utf-8")
    assert "string request_id" in query_source
    assert "uint64 generation" in query_source
    assert "bool final" in answer_source
    assert "uint64 generation" in answer_source
    control_source = (INTERFACES / "msg" / "VoiceControl.msg").read_text(
        encoding="utf-8"
    )
    playback_source = (INTERFACES / "msg" / "VoicePlayback.msg").read_text(
        encoding="utf-8"
    )
    assert "string command" in control_source
    assert "uint64 generation" in control_source
    assert "string purpose" in control_source
    assert "string state" in playback_source
    assert "uint64 generation" in playback_source
    assert "string purpose" in playback_source


def test_voice_interruption_contract_is_wired_end_to_end() -> None:
    answer_source = (VOICE / "voice" / "answer_node.py").read_text(encoding="utf-8")
    bridge_source = (VOICE / "voice" / "bridge_node.py").read_text(encoding="utf-8")
    audio_source = (VOICE / "voice" / "audio_input_node.py").read_text(encoding="utf-8")
    config_source = (VOICE / "config" / "voice_rag.yaml").read_text(encoding="utf-8")

    assert 'declare_parameter("interrupt_topic", "/voice/interrupt")' in answer_source
    assert 'declare_parameter("interrupt_topic", "/voice/interrupt")' in bridge_source
    assert 'declare_parameter("interrupt_topic", "/voice/interrupt")' in audio_source
    assert config_source.count("interrupt_topic: /voice/interrupt") == 4
    assert "_answer_event_is_stale" in answer_source
    assert 'stop.is_play = False' in bridge_source
    assert 'acknowledgement.text = "我在"' in bridge_source
    assert '"purpose", "interrupt_ack"' in bridge_source


def test_voice_arm_motion_contract_is_wired_end_to_end() -> None:
    arm_source = (VOICE / "voice" / "arm_motion_node.py").read_text(encoding="utf-8")
    bridge_source = (VOICE / "voice" / "bridge_node.py").read_text(encoding="utf-8")
    audio_source = (VOICE / "voice" / "audio_input_node.py").read_text(
        encoding="utf-8"
    )
    config_source = (VOICE / "config" / "voice_rag.yaml").read_text(encoding="utf-8")
    launch_source = (VOICE / "launch" / "voice_rag.launch.py").read_text(
        encoding="utf-8"
    )
    setup_source = (VOICE / "setup.py").read_text(encoding="utf-8")

    assert "from std_msgs.msg import Byte" in arm_source
    assert 'message.data = bytes([code])' in arm_source
    assert '"wake_wave":' in arm_source
    assert '"answer_gesture":' in arm_source
    assert '"closing_wave":' in arm_source
    assert '"default_motion_code", 1' in arm_source
    assert 'purpose="wake_wave"' in audio_source
    assert '"closing_wave" if disposition == "closing" else "answer_gesture"' in bridge_source
    assert config_source.count("arm_motion_request_topic: /voice/arm_motion_request") == 2
    assert "voice_arm_motion_node:" in config_source
    assert 'executable="voice_arm_motion_node"' in launch_source
    assert "voice_arm_motion_node = voice.venv_entrypoints:arm_main" in setup_source


def test_voice_python_modules_compile_without_ros2_runtime() -> None:
    node_sources = sorted((VOICE / "voice").glob("*_node.py"))
    for source in sorted((VOICE / "voice").glob("*.py")):
        py_compile.compile(str(source), doraise=True)
    py_compile.compile(str(VOICE / "launch" / "voice_rag.launch.py"), doraise=True)

    answer_source = (VOICE / "voice" / "answer_node.py").read_text(encoding="utf-8")
    launch_source = (VOICE / "launch" / "voice_rag.launch.py").read_text(
        encoding="utf-8"
    )
    audio_source = (VOICE / "voice" / "audio_input_node.py").read_text(
        encoding="utf-8"
    )
    config_source = (VOICE / "config" / "voice_rag.yaml").read_text(
        encoding="utf-8"
    )
    run_source = (VOICE / "scripts" / "run_voice.sh").read_text(encoding="utf-8")
    assert "self._services" not in answer_source
    assert '"RAG_DATA_DIR"' in launch_source
    assert "default_value=PathJoinSubstitution" in launch_source
    assert 'data_dir:="${RAG_DATA_DIR}"' in run_source
    assert 'export RAG_SYSTEM_PYTHON="${FULL_PYTHON}"' in run_source
    assert '"enable_local_audio"' in launch_source
    assert '"voice_arm_motion_node"' in launch_source
    assert '"voice.audio_input_node"' in launch_source
    assert '"interrupt_topic"' in audio_source
    assert "/voice/interrupt" in config_source
    for source in node_sources:
        assert "if rclpy.ok():" in source.read_text(encoding="utf-8")


def test_voice_setup_uses_linux_installed_rag_source() -> None:
    completed = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=VOICE,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "voice"
    setup_source = (VOICE / "setup.py").read_text(encoding="utf-8")
    install_source = (VOICE / "scripts" / "install_linux.sh").read_text(
        encoding="utf-8"
    )
    deploy_source = (ROOT / "scripts" / "deploy_voice_to_robot.sh").read_text(
        encoding="utf-8"
    )
    assert "RAG_SOURCE" not in setup_source
    assert 'scripts/install_all_linux.sh' in install_source
    assert '--base-paths "${PROJECT_ROOT}/ros2"' in install_source
    assert "voice.venv_entrypoints:bridge_main" in setup_source
    assert "voice.venv_entrypoints:answer_main" in setup_source
    assert "voice.venv_entrypoints:knowledge_main" in setup_source
    assert "/home/nerv/Voice" in deploy_source


def test_ros_entrypoints_delegate_to_full_linux_environment(
    monkeypatch, tmp_path: Path
) -> None:
    sys.path.insert(0, str(VOICE))
    try:
        from voice import venv_entrypoints

        base_python = tmp_path / "python3"
        base_python.write_text("", encoding="utf-8")
        base_python.chmod(0o755)
        python = tmp_path / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.symlink_to(base_python)
        monkeypatch.setenv("RAG_SYSTEM_PYTHON", str(python))
        monkeypatch.setattr(sys, "argv", ["rag_answer_node", "--ros-args"])

        captured: dict[str, object] = {}

        def fake_execv(executable: str, argv: list[str]) -> None:
            captured["executable"] = executable
            captured["argv"] = argv

        monkeypatch.setattr(venv_entrypoints.os, "execv", fake_execv)
        venv_entrypoints.answer_main()

        assert captured["executable"] == str(python.absolute())
        argv = captured["argv"]
        assert isinstance(argv, list)
        assert argv[:2] == [str(python.absolute()), "-c"]
        assert "executable={sys.executable} prefix={sys.prefix}" in argv[2]
        assert "from voice.answer_node import main; main()" in argv[2]
        assert argv[3:] == ["--ros-args"]
    finally:
        sys.path.remove(str(VOICE))


def test_ros_runtime_helpers_bound_external_values(monkeypatch, tmp_path: Path) -> None:
    sys.path.insert(0, str(VOICE))
    try:
        from voice.runtime import default_data_dir, nonempty_session_id, normalize_top_k

        monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "robot-data"))
        assert default_data_dir() == (tmp_path / "robot-data").resolve()
        assert normalize_top_k(0, default=4) == 4
        assert normalize_top_k(100) == 20
        assert nonempty_session_id("") == "robot"
        assert nonempty_session_id("  robot  ") == "robot"
    finally:
        sys.path.remove(str(VOICE))
