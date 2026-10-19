from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import RLock

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from rag_interfaces.action import IndexDocument, ParseDocument
from rag_interfaces.srv import GetIndexStats, Search, SetDocumentEnabled
from rag_system.indexing import IndexingService
from rag_system.parsing import ParsingService
from rag_system.settings import SettingsStore

from voice.runtime import default_data_dir, normalize_top_k


class RagKnowledgeNode(Node):
    """ROS2 boundary around the current parse, index, search, and document APIs."""

    def __init__(self) -> None:
        super().__init__("rag_knowledge_node")
        data_dir = default_data_dir()
        self.declare_parameter("data_dir", str(data_dir))
        self.declare_parameter("settings_path", str(data_dir / "settings.json"))
        self.declare_parameter("index_dir", str(data_dir / "index"))
        self.declare_parameter("parser_cache_dir", str(data_dir / "parser_cache"))
        self.declare_parameter("chunk_size", 2400)
        self.declare_parameter("chunk_overlap", 300)
        self.declare_parameter("search_service", "/rag/search")
        self.declare_parameter("index_stats_service", "/rag/index_stats")
        self.declare_parameter("set_document_enabled_service", "/rag/set_document_enabled")
        self.declare_parameter("parse_action", "/rag/parse_document")
        self.declare_parameter("index_action", "/rag/index_document")

        self._settings_path = Path(
            self.get_parameter("settings_path").value
        ).expanduser().resolve()
        self._index_dir = Path(self.get_parameter("index_dir").value).expanduser().resolve()
        self._cache_dir = Path(
            self.get_parameter("parser_cache_dir").value
        ).expanduser().resolve()
        self._chunk_size = max(1, int(self.get_parameter("chunk_size").value))
        self._chunk_overlap = max(0, int(self.get_parameter("chunk_overlap").value))
        self._data_lock = RLock()
        self._index_service: IndexingService | None = None
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.create_service(
            Search,
            str(self.get_parameter("search_service").value),
            self._on_search,
        )
        self.create_service(
            GetIndexStats,
            str(self.get_parameter("index_stats_service").value),
            self._on_get_stats,
        )
        self.create_service(
            SetDocumentEnabled,
            str(self.get_parameter("set_document_enabled_service").value),
            self._on_set_document_enabled,
        )
        self._parse_server = ActionServer(
            self,
            ParseDocument,
            str(self.get_parameter("parse_action").value),
            self._execute_parse,
        )
        self._index_server = ActionServer(
            self,
            IndexDocument,
            str(self.get_parameter("index_action").value),
            self._execute_index,
        )
        self.get_logger().info(
            f"RAG knowledge node ready: index={self._index_dir} cache={self._cache_dir}"
        )

    def destroy_node(self) -> bool:
        self._parse_server.destroy()
        self._index_server.destroy()
        return super().destroy_node()

    def _on_search(self, request: Search.Request, response: Search.Response) -> Search.Response:
        query = str(request.query or "").strip()
        if not query:
            response.success = False
            response.message = "query cannot be empty"
            return response
        try:
            with self._data_lock:
                service = self._indexing_service()
                rewrite, results = service.search_with_rewrite(
                    query,
                    top_k=normalize_top_k(int(request.top_k)),
                )
            response.success = True
            response.message = "ok"
            response.diagnostics_json = json.dumps(rewrite.to_dict(), ensure_ascii=False)
            response.ids = [result.id for result in results]
            response.texts = [result.text for result in results]
            response.scores = [float(result.score) for result in results]
            response.source_names = [
                str(result.metadata.get("source_name") or "") for result in results
            ]
        except Exception as exc:
            self.get_logger().error(f"Search failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _on_get_stats(
        self,
        _request: GetIndexStats.Request,
        response: GetIndexStats.Response,
    ) -> GetIndexStats.Response:
        try:
            with self._data_lock:
                stats = self._indexing_service().stats()
            response.success = True
            response.message = "ok"
            response.document_count = int(stats.document_count)
            response.chunk_count = int(stats.chunk_count)
            response.vector_count = int(stats.vector_count)
            response.collection = stats.collection
            response.model = stats.model
            response.dimensions = int(stats.dimensions)
            response.bm25_chunk_count = int(stats.bm25_chunk_count)
            response.sparse_chunk_count = int(stats.sparse_chunk_count)
        except Exception as exc:
            self.get_logger().error(f"Index stats failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _on_set_document_enabled(
        self,
        request: SetDocumentEnabled.Request,
        response: SetDocumentEnabled.Response,
    ) -> SetDocumentEnabled.Response:
        document_id = str(request.document_id or "").strip()
        if not document_id:
            response.success = False
            response.message = "document_id cannot be empty"
            return response
        try:
            with self._data_lock:
                document = self._indexing_service().set_document_enabled(
                    document_id,
                    bool(request.enabled),
                )
            if document is None:
                response.success = False
                response.message = "document not found"
            else:
                response.success = True
                response.message = "ok"
        except Exception as exc:
            self.get_logger().error(f"Set document enabled failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _execute_parse(self, goal_handle: object) -> ParseDocument.Result:
        goal = goal_handle.request
        feedback = ParseDocument.Feedback()
        feedback.progress = 0.1
        feedback.stage = "parsing"
        feedback.detail = Path(goal.file_path).name
        goal_handle.publish_feedback(feedback)
        result = ParseDocument.Result()
        try:
            parser = self._parser_for_goal(goal)
            document = parser.parse_document(goal.file_path, use_cache=bool(goal.enable_cache))
            feedback.progress = 1.0
            feedback.stage = "completed"
            feedback.detail = document.source_name
            goal_handle.publish_feedback(feedback)
            result.success = True
            result.message = "ok"
            result.document_json = json.dumps(asdict(document), ensure_ascii=False)
            result.elapsed_seconds = float(document.elapsed_seconds)
            result.diagnostics = [diagnostic.message for diagnostic in document.diagnostics]
            goal_handle.succeed()
        except Exception as exc:
            self.get_logger().error(f"Parse action failed: {exc}")
            result.success = False
            result.message = str(exc)
            goal_handle.abort()
        return result

    def _execute_index(self, goal_handle: object) -> IndexDocument.Result:
        goal = goal_handle.request
        feedback = IndexDocument.Feedback()
        feedback.progress = 0.1
        feedback.stage = "parsing"
        feedback.detail = Path(goal.file_path).name
        goal_handle.publish_feedback(feedback)
        result = IndexDocument.Result()
        try:
            parser = self._parser_for_goal(goal)
            document = parser.parse_document(goal.file_path, use_cache=bool(goal.enable_cache))
            feedback.progress = 0.7
            feedback.stage = "indexing"
            feedback.detail = document.source_name
            goal_handle.publish_feedback(feedback)
            with self._data_lock:
                indexed = self._indexing_service().index_document(document)
            feedback.progress = 1.0
            feedback.stage = "completed"
            feedback.detail = indexed.document_id
            goal_handle.publish_feedback(feedback)
            result.success = True
            result.message = indexed.message
            result.document_id = indexed.document_id
            result.chunk_count = int(indexed.chunk_count)
            result.vector_count = int(indexed.vector_count)
            goal_handle.succeed()
        except Exception as exc:
            self.get_logger().error(f"Index action failed: {exc}")
            result.success = False
            result.message = str(exc)
            goal_handle.abort()
        return result

    def _parser_for_goal(self, goal: object) -> ParsingService:
        chunk_size = int(goal.chunk_size) if int(goal.chunk_size) > 0 else self._chunk_size
        chunk_overlap = (
            int(goal.chunk_overlap) if int(goal.chunk_overlap) > 0 else self._chunk_overlap
        )
        backend = str(goal.backend or "auto").strip() or "auto"
        return ParsingService.with_chunking(
            backend=backend,
            enable_cache=bool(goal.enable_cache),
            cache_dir=self._cache_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _indexing_service(self) -> IndexingService:
        if self._index_service is None:
            self._index_service = IndexingService(
                settings_store=SettingsStore(self._settings_path),
                index_dir=self._index_dir,
            )
        return self._index_service


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RagKnowledgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
