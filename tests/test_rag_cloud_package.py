from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_system.rag_cloud import RAGCloudNode
from rag_system.rag_cloud.cli import main


@dataclass
class FakeDocument:
    source: str
    source_name: str
    chunks: list[str]


@dataclass
class FakeIndexResult:
    document_id: str
    chunk_count: int


class FakeParser:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def parse_document(self, source):
        self.sources.append(str(source))
        return FakeDocument(str(source), str(source).split("/")[-1], ["第一段", "第二段"])


class FakeIndexer:
    def __init__(self) -> None:
        self.documents: list[FakeDocument] = []
        self.deletes: list[tuple[str, bool]] = []

    def index_document(self, document):
        self.documents.append(document)
        return FakeIndexResult("doc-001", len(document.chunks))

    def delete_document(self, document_id: str, *, remove_manifest: bool = True):
        self.deletes.append((document_id, remove_manifest))
        return {"document_id": document_id, "manifest": remove_manifest, "text_vectors": 2}


def test_cloud_node_parse_index_and_delete_are_small_json_facades():
    parser = FakeParser()
    indexer = FakeIndexer()
    node = RAGCloudNode(parser=parser, indexer=indexer)

    parsed = node.parse("customer.docx")
    indexed = node.index("customer.docx")
    deleted = node.delete("doc-001")

    assert parsed == {
        "ok": True,
        "module": "rag-cloud",
        "operation": "parse",
        "data": {
            "source": "customer.docx",
            "source_name": "customer.docx",
            "chunks": ["第一段", "第二段"],
        },
    }
    assert indexed["data"] == {"document_id": "doc-001", "chunk_count": 2}
    assert parser.sources == ["customer.docx", "customer.docx"]
    assert indexer.documents[0].source_name == "customer.docx"
    assert deleted["data"]["text_vectors"] == 2
    assert indexer.deletes == [("doc-001", True)]


def test_cloud_node_can_index_an_already_parsed_document():
    parser = FakeParser()
    indexer = FakeIndexer()
    node = RAGCloudNode(parser=parser, indexer=indexer)
    document = FakeDocument("memory", "memory.txt", ["内容"])

    result = node.index(document)

    assert result["data"]["chunk_count"] == 1
    assert parser.sources == []
    assert indexer.documents == [document]


def test_cloud_cli_prints_one_json_result(capsys):
    node = RAGCloudNode(parser=FakeParser(), indexer=FakeIndexer())

    exit_code = main(["parse", "sample.txt"], node=node)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["operation"] == "parse"


def test_cloud_node_rejects_empty_identifiers():
    node = RAGCloudNode(parser=FakeParser(), indexer=FakeIndexer())

    with pytest.raises(ValueError, match="source"):
        node.parse("  ")
    with pytest.raises(ValueError, match="document_id"):
        node.delete("")


def test_cloud_cli_defaults_to_an_isolated_acceptance_index(monkeypatch, capsys):
    captured: dict[str, Path] = {}
    fake_node = RAGCloudNode(parser=FakeParser(), indexer=FakeIndexer())

    def fake_builder(*, index_dir):
        captured["index_dir"] = Path(index_dir)
        return fake_node

    monkeypatch.setattr("rag_system.rag_cloud.cli.build_default_node", fake_builder)

    assert main(["parse", "sample.txt"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["index_dir"] == Path("data/acceptance/index")
