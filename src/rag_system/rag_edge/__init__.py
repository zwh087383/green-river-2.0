"""Edge/read-side RAG facade with no write operations."""

from rag_system.rag_edge.node import KnowledgeReader, RAGEdgeNode, build_default_node

__all__ = ["KnowledgeReader", "RAGEdgeNode", "build_default_node"]
