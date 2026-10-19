from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_system.agent import AgentNode
from rag_system.agent.cli import main


@dataclass
class FakeAnswer:
    query: str
    answer: str
    model: str


class FakeAnswerService:
    def __init__(self) -> None:
        self.answer_calls: list[tuple[str, int | None]] = []
        self.stream_calls: list[tuple[str, int | None]] = []

    def answer(self, query: str, *, top_k: int | None = None):
        self.answer_calls.append((query, top_k))
        return FakeAnswer(query, "绿色江河是一家环保组织。", "fake-model")

    def stream_answer_events(self, query: str, *, top_k: int | None = None):
        self.stream_calls.append((query, top_k))
        yield {"type": "delta", "text": "绿色江河"}
        yield {"type": "final", "result": {"answer": "绿色江河是一家环保组织。"}}


def test_agent_node_hides_generation_pipeline_behind_answer_and_stream():
    service = FakeAnswerService()
    node = AgentNode(service=service)

    answer = node.answer(" 介绍一下绿色江河 ", top_k=2)
    events = list(node.stream("介绍一下绿色江河", top_k=2))

    assert service.answer_calls == [("介绍一下绿色江河", 2)]
    assert service.stream_calls == [("介绍一下绿色江河", 2)]
    assert answer["data"] == {
        "query": "介绍一下绿色江河",
        "answer": "绿色江河是一家环保组织。",
        "model": "fake-model",
    }
    assert [event["data"]["type"] for event in events] == ["delta", "final"]
    assert all(event["module"] == "agent" for event in events)


def test_agent_node_rejects_empty_question_and_invalid_top_k():
    node = AgentNode(service=FakeAnswerService())

    with pytest.raises(ValueError, match="query"):
        node.answer("  ")
    with pytest.raises(ValueError, match="top_k"):
        list(node.stream("问题", top_k=101))


def test_agent_stream_cli_outputs_json_lines(capsys):
    node = AgentNode(service=FakeAnswerService())

    exit_code = main(["stream", "介绍一下绿色江河"], node=node)

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["data"]["type"] for line in lines] == ["delta", "final"]


def test_agent_stream_pretty_flag_still_outputs_valid_json_lines(capsys):
    node = AgentNode(service=FakeAnswerService())

    exit_code = main(["--pretty", "stream", "介绍一下绿色江河"], node=node)

    assert exit_code == 0
    output_lines = capsys.readouterr().out.splitlines()
    assert len(output_lines) == 2
    assert [json.loads(line)["data"]["type"] for line in output_lines] == ["delta", "final"]


def test_agent_cli_defaults_to_an_isolated_acceptance_index(monkeypatch, capsys):
    captured: dict[str, Path] = {}
    fake_node = AgentNode(service=FakeAnswerService())

    def fake_builder(*, index_dir):
        captured["index_dir"] = Path(index_dir)
        return fake_node

    monkeypatch.setattr("rag_system.agent.cli.build_default_node", fake_builder)

    assert main(["answer", "介绍一下绿色江河"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["index_dir"] == Path("data/acceptance/index")
