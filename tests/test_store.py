from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

from rag_system.indexing.store import ChromaVectorStore


def test_chroma_first_use_is_serialized_for_concurrent_collections(monkeypatch, tmp_path) -> None:
    start = Barrier(3)
    first_client_entered = Event()
    release_first_client = Event()
    second_client_entered = Event()
    state_lock = Lock()
    state = {"calls": 0, "active": 0, "overlapped": False}

    class FakeClient:
        def get_or_create_collection(self, *, name, metadata):
            return SimpleNamespace(name=name, metadata=metadata)

    def persistent_client(*, path):
        del path
        with state_lock:
            state["calls"] += 1
            call_number = state["calls"]
            state["active"] += 1
            if state["active"] > 1:
                state["overlapped"] = True

        try:
            if call_number == 1:
                first_client_entered.set()
                assert release_first_client.wait(timeout=1)
            else:
                second_client_entered.set()
            return FakeClient()
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=persistent_client))

    def create_store(collection_name: str) -> ChromaVectorStore:
        start.wait(timeout=1)
        return ChromaVectorStore(path=tmp_path / "chroma", collection_name=collection_name)

    with ThreadPoolExecutor(max_workers=2) as executor:
        text_store = executor.submit(create_store, "rag_chunks")
        image_store = executor.submit(create_store, "rag_images")
        start.wait(timeout=1)
        assert first_client_entered.wait(timeout=1)
        try:
            assert not second_client_entered.wait(timeout=0.1)
        finally:
            release_first_client.set()

        assert text_store.result(timeout=1).collection_name == "rag_chunks"
        assert image_store.result(timeout=1).collection_name == "rag_images"

    assert state == {"calls": 2, "active": 0, "overlapped": False}
