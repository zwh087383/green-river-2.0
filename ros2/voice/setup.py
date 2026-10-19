from __future__ import annotations

from glob import glob

from setuptools import find_packages, setup

setup(
    name="voice",
    version="0.1.0",
    packages=find_packages(include=["voice", "voice.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/voice"]),
        ("share/voice", ["package.xml"]),
        ("share/voice/launch", glob("launch/*.launch.py")),
        ("share/voice/config", glob("config/*.yaml")),
        ("share/voice/scripts", glob("scripts/*.sh")),
        ("share/voice/systemd", glob("systemd/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Green River",
    maintainer_email="noreply@example.invalid",
    description="Robot speech bridge and ROS2 wrappers for the Green River RAG system.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "voice_bridge_node = voice.venv_entrypoints:bridge_main",
            "voice_arm_motion_node = voice.venv_entrypoints:arm_main",
            "rag_answer_node = voice.venv_entrypoints:answer_main",
            "rag_knowledge_node = voice.venv_entrypoints:knowledge_main",
            "display_node = voice.venv_entrypoints:display_main",
        ],
    },
)
