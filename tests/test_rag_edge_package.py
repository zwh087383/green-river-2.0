from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_system.rag_edge import RAGEdgeNode
from rag_system.rag_edge.cli import main


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float


@dataclass
class FakeDocument:
    document_id: str
    source_name: str


@dataclass
class FakeStats:
    document_count: int
    chunk_count: int
    vector_count: int


class FakeReader:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return [FakeSearchResult("hit-1", "绿色江河介绍", 0.91)]

    def list_documents(self):
        return [FakeDocument("doc-1", "知识库.docx")]

    def stats(self):
        return FakeStats(document_count=1, chunk_count=3, vector_count=3)


def test_edge_node_exposes_only_simple_read_operations():
    reader = FakeReader()
    node = RAGEdgeNode(reader=reader)

    search = node.search("  绿色江河  ", top_k=3)
    documents = node.list()
    stats = node.stats()

    assert reader.search_calls == [("绿色江河", 3)]
    assert search["data"] == {
        "query": "绿色江河",
        "top_k": 3,
        "count": 1,
        "results": [{"id": "hit-1", "text": "绿色江河介绍", "score": 0.91}],
    }
    assert documents["data"] == {
        "count": 1,
        "documents": [{"document_id": "doc-1", "source_name": "知识库.docx"}],
    }
    assert stats["data"] == {"document_count": 1, "chunk_count": 3, "vector_count": 3}


def test_edge_node_validates_customer_inputs():
    node = RAGEdgeNode(reader=FakeReader())

    with pytest.raises(ValueError, match="query"):
        node.search(" ")
    with pytest.raises(ValueError, match="top_k"):
        node.search("test", top_k=0)


def test_edge_cli_is_a_single_json_command(capsys):
    node = RAGEdgeNode(reader=FakeReader())

    exit_code = main(["search", "绿色江河", "--top-k", "2"], node=node)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["module"] == "rag-edge"
    assert payload["data"]["top_k"] == 2


def test_edge_cli_defaults_to_an_isolated_acceptance_index(monkeypatch, capsys):
    captured: dict[str, Path] = {}
    fake_node = RAGEdgeNode(reader=FakeReader())

    def fake_builder(*, index_dir):
        captured["index_dir"] = Path(index_dir)
        return fake_node

    monkeypatch.setattr("rag_system.rag_edge.cli.build_default_node", fake_builder)

    assert main(["stats"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["index_dir"] == Path("data/acceptance/index")
