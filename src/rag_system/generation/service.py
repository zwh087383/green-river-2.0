from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from rag_system.generation.context import build_evidence_chunks, citation_from_evidence
from rag_system.generation.deepseek_chat import (
    AnswerClient,
    DeepSeekAnswerClient,
    IncompleteAnswerStreamError,
    _citation_ids_from_text,
    _dedupe_ids,
    build_chat_answer_messages,
    build_farewell_messages,
    build_rag_answer_messages,
)
from rag_system.generation.guard import direct_answer_for_query
from rag_system.generation.history import ConversationHistory, ROUTER_HISTORY_MAX_ROUNDS
from rag_system.generation.identity import (
    CachedIdentityDetector,
    DeepSeekIdentityDetector,
    IdentityDetector,
    IdentityResult,
)
from rag_system.generation.intent_router import (
    TURN_SIGNAL_END_SESSION,
    TURN_SIGNAL_IGNORE,
    DeepSeekIntentRouter,
    IntentRouter,
    build_router_messages,
    fallback_intent,
)
from rag_system.generation.models import AnswerResult, Citation, ConversationTurn, EvidenceChunk, IntentRouteResult
from rag_system.indexing.models import SearchResult
from rag_system.indexing.service import IndexingService, QueryEmbeddingBundle
from rag_system.settings import SettingsStore


LOW_RRF_THRESHOLD = 0.012
MIN_MATCHED_TOKEN_RATIO = 0.15

_CITATION_RE = re.compile(r"\[\d+\]|【\d+】")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([，。！？；：,.!?;:])")


def _cancel_requested(should_cancel: Callable[[], bool] | None) -> bool:
    """Return whether cooperative cancellation has been requested."""
    if should_cancel is None:
        return False
    return bool(should_cancel())


def _remove_citation_markers(text: str) -> str:
    without_markers = _CITATION_RE.sub("", text)
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", without_markers).strip()


QualityLevel = Literal["high", "medium", "low"]


@dataclass(slots=True)
class RetrievalQuality:
    level: QualityLevel
    should_retry: bool
    reasons: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "should_retry": self.should_retry,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


class AnswerGenerationService:
    def __init__(
        self,
        *,
        settings_store: SettingsStore | None = None,
        indexing_service: IndexingService | None = None,
        answer_client: AnswerClient | None = None,
        intent_router: IntentRouter | None = None,
        identity_detector: IdentityDetector | None = None,
        conversation_history: ConversationHistory | None = None,
        index_dir: Path | str = Path("data/index"),
        context_top_k: int = 4,
    ) -> None:
        self.settings_store = settings_store or SettingsStore()
        self.index_dir = Path(index_dir)
        self.indexing_service = indexing_service
        self.answer_client = answer_client
        self.intent_router = intent_router
        self._intent_router_injected = intent_router is not None
        self.identity_detector = identity_detector
        self.conversation_history = (
            conversation_history
            if conversation_history is not None
            else ConversationHistory()
        )
        self.context_top_k = max(1, min(context_top_k, 4))

    def answer(self, query: str, *, top_k: int | None = None) -> AnswerResult:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Answer query cannot be empty")
        settings = self.settings_store.load()
        self.conversation_history.set_max_rounds(
            max(settings.conversation_max_rounds, ROUTER_HISTORY_MAX_ROUNDS)
        )
        history = self.conversation_history.snapshot(max_rounds=settings.conversation_max_rounds)
        router_history = self.conversation_history.snapshot(
            max_rounds=ROUTER_HISTORY_MAX_ROUNDS
        )
        direct_answer = direct_answer_for_query(cleaned)
        if direct_answer is not None:
            reason, answer = direct_answer
            result = _direct_answer_result(
                query=cleaned,
                answer=answer,
                reason=reason,
                history_rounds=len(history),
                router_history_rounds=len(router_history),
            )
            self.conversation_history.append(cleaned, answer)
            return result
        fast, query_embeddings = self._coordinate_request(cleaned, router_history)
        if fast.should_ignore:
            return _ignored_turn_result(
                query=cleaned,
                model=fast.intent.model,
                signal=fast.turn_signal,
                reason=fast.turn_reason,
                history_rounds=len(history),
                router_history_rounds=len(router_history),
            )
        if fast.should_end_session:
            return self._collect_farewell_result(
                cleaned,
                signal=fast.turn_signal,
                reason=fast.turn_reason,
            )
        intent = fast.intent
        identity = _identity_from_fast(fast)
        if not intent.needs_retrieval:
            generated = self._answer_client().generate_chat(cleaned, history=history, intent=intent, identity=identity)
            result = _chat_answer_result(
                query=cleaned,
                answer=generated.answer,
                model=self._answer_client().model,
                confidence=generated.confidence,
                intent=intent,
                history_rounds=len(history),
                router_history_rounds=len(router_history),
                identity=identity,
            )
            self.conversation_history.append(cleaned, generated.answer)
            return result
        context_top_k = max(1, min(top_k or self.context_top_k, 4))
        results, evidence, retrieval = self._retrieve_context(
            query=cleaned,
            intent=intent,
            context_top_k=context_top_k,
            history_rounds=len(history),
            router_history_rounds=len(router_history),
            query_embeddings=query_embeddings,
        )
        generated = self._answer_client().generate(cleaned, evidence, history=history, identity=identity)
        answer = _remove_citation_markers(generated.answer)
        citations = _ground_citations(generated.citation_ids, evidence)
        result = AnswerResult(
            query=cleaned,
            answer=answer,
            citations=citations,
            evidence=evidence,
            model=self._answer_client().model,
            confidence=generated.confidence,
            retrieval=retrieval,
            identity=identity.to_dict() if identity else {},
        )
        self.conversation_history.append(cleaned, answer)
        return result

    def stream_answer_events(
        self,
        query: str,
        *,
        top_k: int | None = None,
        should_cancel: Callable[[], bool] | None = None,
        include_llm_inputs: bool = False,
    ) -> Iterator[dict[str, Any]]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Answer query cannot be empty")
        if _cancel_requested(should_cancel):
            return
        settings = self.settings_store.load()
        self.conversation_history.set_max_rounds(
            max(settings.conversation_max_rounds, ROUTER_HISTORY_MAX_ROUNDS)
        )
        history = self.conversation_history.snapshot(max_rounds=settings.conversation_max_rounds)
        router_history = self.conversation_history.snapshot(
            max_rounds=ROUTER_HISTORY_MAX_ROUNDS
        )
        direct_answer = direct_answer_for_query(cleaned)
        if direct_answer is not None:
            if _cancel_requested(should_cancel):
                return
            reason, answer = direct_answer
            result = _direct_answer_result(
                query=cleaned,
                answer=answer,
                reason=reason,
                history_rounds=len(history),
                router_history_rounds=len(router_history),
            )
            retrieval = result.retrieval
            yield {
                "type": "meta",
                "query": cleaned,
                "model": result.model,
                "evidence": [],
                "retrieval": retrieval,
            }
            if _cancel_requested(should_cancel):
                self._append_interrupted_turn(cleaned, answer)
                return
            yield {"type": "delta", "text": answer}
            if _cancel_requested(should_cancel):
                self._append_interrupted_turn(cleaned, answer)
                return
            self.conversation_history.append(cleaned, answer)
            if _cancel_requested(should_cancel):
                return
            yield {"type": "final", "result": result.to_dict()}
            return
        if include_llm_inputs:
            yield _llm_input_event(
                stage="router",
                title="前置分析：轮次、路由与身份",
                model=settings.router_model,
                messages=build_router_messages(
                    query=cleaned,
                    history=router_history,
                    turn_prompt=settings.deepseek_turn_prompt,
                ),
            )
            if _cancel_requested(should_cancel):
                return
        fast, query_embeddings = self._coordinate_request(cleaned, router_history)
        if _cancel_requested(should_cancel):
            return
        if fast.should_ignore:
            if _cancel_requested(should_cancel):
                return
            yield {
                "type": "ignored",
                "signal": fast.turn_signal,
                "reason": fast.turn_reason or "未识别到有效问题，继续聆听。",
                "model": fast.intent.model,
            }
            return
        if fast.should_end_session:
            yield from self.stream_farewell_events(
                cleaned,
                signal=fast.turn_signal,
                reason=fast.turn_reason,
                should_cancel=should_cancel,
                include_llm_inputs=include_llm_inputs,
            )
            return
        intent = fast.intent
        identity = _identity_from_fast(fast)
        if not intent.needs_retrieval:
            if _cancel_requested(should_cancel):
                return
            retrieval = {
                "bypassed": True,
                "reason": intent.route,
                "intent": intent.to_dict(),
                "context_top_k": 0,
                "results": [],
                "history_rounds": len(history),
                "router_history_rounds": len(router_history),
            }
            client = self._answer_client()
            yield {
                "type": "meta",
                "query": cleaned,
                "model": client.model,
                "evidence": [],
                "retrieval": retrieval,
                "identity": identity.to_dict() if identity else None,
            }
            if include_llm_inputs:
                yield _llm_input_event(
                    stage="chat_answer",
                    title="直接对话回答",
                    model=client.model,
                    messages=build_chat_answer_messages(
                        query=cleaned,
                        history=history,
                        intent=intent,
                        identity=identity,
                        stream=True,
                    ),
                )
                if _cancel_requested(should_cancel):
                    return
            answer_parts: list[str] = []
            fallback_generated = None
            stream_chat = getattr(client, "stream_chat", None)
            if callable(stream_chat):
                try:
                    for delta in stream_chat(cleaned, history=history, intent=intent, identity=identity):
                        if _cancel_requested(should_cancel):
                            self._append_interrupted_turn(cleaned, "".join(answer_parts))
                            return
                        if not delta:
                            continue
                        answer_parts.append(str(delta))
                        yield {"type": "delta", "text": str(delta)}
                except IncompleteAnswerStreamError:
                    if _cancel_requested(should_cancel):
                        self._append_interrupted_turn(cleaned, "".join(answer_parts))
                        return
                    if include_llm_inputs:
                        yield _llm_input_event(
                            stage="chat_answer_fallback",
                            title="直接对话回答（非流式降级）",
                            model=client.model,
                            messages=build_chat_answer_messages(
                                query=cleaned,
                                history=history,
                                intent=intent,
                                identity=identity,
                            ),
                        )
                        if _cancel_requested(should_cancel):
                            self._append_interrupted_turn(cleaned, "".join(answer_parts))
                            return
                    fallback_generated = client.generate_chat(
                        cleaned,
                        history=history,
                        intent=intent,
                        identity=identity,
                    )
                    if _cancel_requested(should_cancel):
                        self._append_interrupted_turn(cleaned, fallback_generated.answer)
                        return
                    answer_parts = [fallback_generated.answer]
            else:
                generated = client.generate_chat(cleaned, history=history, intent=intent, identity=identity)
                if _cancel_requested(should_cancel):
                    self._append_interrupted_turn(cleaned, generated.answer)
                    return
                answer_parts.append(generated.answer)
                yield {"type": "delta", "text": generated.answer}
            answer_text = "".join(answer_parts).strip() or "我在。"
            answer_text = _remove_citation_markers(answer_text)
            result = AnswerResult(
                query=cleaned,
                answer=answer_text,
                citations=[],
                evidence=[],
                model=client.model,
                confidence=fallback_generated.confidence if fallback_generated else "medium",
                retrieval=retrieval,
                identity=identity.to_dict() if identity else {},
            )
            if _cancel_requested(should_cancel):
                self._append_interrupted_turn(cleaned, answer_text)
                return
            self.conversation_history.append(cleaned, answer_text)
            if _cancel_requested(should_cancel):
                return
            yield {"type": "final", "result": result.to_dict()}
            return
        context_top_k = max(1, min(top_k or self.context_top_k, 4))
        if _cancel_requested(should_cancel):
            return
        results, evidence, retrieval = self._retrieve_context(
            query=cleaned,
            intent=intent,
            context_top_k=context_top_k,
            history_rounds=len(history),
            router_history_rounds=len(router_history),
            query_embeddings=query_embeddings,
        )
        if _cancel_requested(should_cancel):
            return
        client = self._answer_client()
        yield {
            "type": "meta",
            "query": cleaned,
            "model": client.model,
            "evidence": [asdict(item) for item in evidence],
            "retrieval": retrieval,
            "identity": identity.to_dict() if identity else None,
        }
        if include_llm_inputs and evidence:
            yield _llm_input_event(
                stage="rag_answer",
                title="RAG 回答",
                model=client.model,
                messages=build_rag_answer_messages(
                    query=cleaned,
                    evidence=evidence,
                    history=history,
                    identity=identity,
                    stream=True,
                ),
            )
            if _cancel_requested(should_cancel):
                return

        answer_parts: list[str] = []
        fallback_generated = None
        stream_method = getattr(client, "stream_text", None)
        if callable(stream_method):
            try:
                for delta in stream_method(cleaned, evidence, history=history, identity=identity):
                    if _cancel_requested(should_cancel):
                        self._append_interrupted_turn(cleaned, "".join(answer_parts))
                        return
                    if not delta:
                        continue
                    answer_parts.append(str(delta))
                    yield {"type": "delta", "text": str(delta)}
            except IncompleteAnswerStreamError:
                if _cancel_requested(should_cancel):
                    self._append_interrupted_turn(cleaned, "".join(answer_parts))
                    return
                if include_llm_inputs and evidence:
                    yield _llm_input_event(
                        stage="rag_answer_fallback",
                        title="RAG 回答（非流式降级）",
                        model=client.model,
                        messages=build_rag_answer_messages(
                            query=cleaned,
                            evidence=evidence,
                            history=history,
                            identity=identity,
                        ),
                    )
                    if _cancel_requested(should_cancel):
                        self._append_interrupted_turn(cleaned, "".join(answer_parts))
                        return
                fallback_generated = client.generate(
                    cleaned,
                    evidence,
                    history=history,
                    identity=identity,
                )
                if _cancel_requested(should_cancel):
                    self._append_interrupted_turn(cleaned, fallback_generated.answer)
                    return
                answer_parts = [fallback_generated.answer]
        else:
            generated = client.generate(cleaned, evidence, history=history, identity=identity)
            if _cancel_requested(should_cancel):
                self._append_interrupted_turn(cleaned, generated.answer)
                return
            answer_parts.append(generated.answer)
            yield {"type": "delta", "text": generated.answer}

        answer_text = "".join(answer_parts).strip() or "当前资料中没有找到足够证据。"
        citation_ids = _citation_ids_from_text(answer_text)
        if fallback_generated is not None:
            citation_ids.extend(fallback_generated.citation_ids)
        citation_ids = _dedupe_ids(citation_ids, valid_ids={item.id for item in evidence})
        answer_text = _remove_citation_markers(answer_text)
        citations = _ground_citations(citation_ids, evidence)
        confidence = (
            fallback_generated.confidence
            if fallback_generated is not None
            else _stream_confidence(answer_text, citation_count=len(citations))
        )
        result = AnswerResult(
            query=cleaned,
            answer=answer_text,
            citations=citations,
            evidence=evidence,
            model=client.model,
            confidence=confidence,
            retrieval=retrieval,
            identity=identity.to_dict() if identity else {},
        )
        if _cancel_requested(should_cancel):
            self._append_interrupted_turn(cleaned, answer_text)
            return
        self.conversation_history.append(cleaned, answer_text)
        if _cancel_requested(should_cancel):
            return
        yield {"type": "final", "result": result.to_dict()}

    def _append_interrupted_turn(self, query: str, answer: str) -> None:
        """Keep useful streamed output when a turn is cancelled mid-generation."""

        cleaned_answer = _remove_citation_markers(str(answer or "")).strip()
        if cleaned_answer:
            self.conversation_history.append(query, cleaned_answer)

    def stream_farewell_events(
        self,
        query: str,
        *,
        signal: str = TURN_SIGNAL_END_SESSION,
        reason: str = "",
        should_cancel: Callable[[], bool] | None = None,
        include_llm_inputs: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Generate a session-ending utterance without routing or retrieval."""
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Farewell query cannot be empty")
        if _cancel_requested(should_cancel):
            return
        settings = self.settings_store.load()
        self.conversation_history.set_max_rounds(settings.conversation_max_rounds)
        history = self.conversation_history.snapshot(max_rounds=settings.conversation_max_rounds)
        client = self._answer_client()
        retrieval = {
            "bypassed": True,
            "reason": "end_session",
            "turn_signal": signal,
            "turn_reason": reason,
            "context_top_k": 0,
            "results": [],
            "history_rounds": len(history),
        }
        yield {
            "type": "meta",
            "query": cleaned,
            "model": client.model,
            "evidence": [],
            "retrieval": retrieval,
        }
        if include_llm_inputs:
            yield _llm_input_event(
                stage="farewell",
                title="会话结束语",
                model=client.model,
                messages=build_farewell_messages(history),
            )
            if _cancel_requested(should_cancel):
                return
        parts: list[str] = []
        stream_farewell = getattr(client, "stream_farewell", None)
        try:
            if callable(stream_farewell):
                for delta in stream_farewell(history=history):
                    if _cancel_requested(should_cancel):
                        return
                    text = str(delta or "")
                    if text:
                        parts.append(text)
                        yield {"type": "delta", "text": text}
            else:
                if _cancel_requested(should_cancel):
                    return
                parts.append("感谢您的交流，再见！")
                yield {"type": "delta", "text": parts[0]}
        except IncompleteAnswerStreamError:
            if _cancel_requested(should_cancel):
                return
            parts = ["感谢您的交流，再见！"]
            yield {"type": "delta", "text": parts[0]}
        answer = _remove_citation_markers("".join(parts)) or "感谢您的交流，再见！"
        result = AnswerResult(
            query=cleaned,
            answer=answer,
            citations=[],
            evidence=[],
            model=client.model,
            confidence="medium",
            retrieval=retrieval,
        )
        if _cancel_requested(should_cancel):
            return
        self.conversation_history.clear()
        if _cancel_requested(should_cancel):
            return
        yield {"type": "final", "result": result.to_dict()}

    def _collect_farewell_result(
        self,
        query: str,
        *,
        signal: str,
        reason: str,
    ) -> AnswerResult:
        for event in self.stream_farewell_events(query, signal=signal, reason=reason):
            if event.get("type") != "final":
                continue
            result = event.get("result")
            if not isinstance(result, dict):
                continue
            return AnswerResult(
                query=str(result.get("query") or query),
                answer=str(result.get("answer") or "感谢您的交流，再见！"),
                citations=[],
                evidence=[],
                model=str(result.get("model") or self._answer_client().model),
                confidence=str(result.get("confidence") or "medium"),  # type: ignore[arg-type]
                retrieval=dict(result.get("retrieval") or {}),
                identity=dict(result.get("identity") or {}),
            )
        raise RuntimeError("farewell provider did not emit a final result")

    def _retrieve_context(
        self,
        *,
        query: str,
        intent: IntentRouteResult,
        context_top_k: int,
        history_rounds: int,
        router_history_rounds: int,
        query_embeddings: QueryEmbeddingBundle | None = None,
    ) -> tuple[list[SearchResult], list[EvidenceChunk], dict[str, Any]]:
        indexing_service = self._indexing_service()
        search_method = getattr(indexing_service, "search", None)
        if callable(search_method):
            try:
                first_results = search_method(query, top_k=context_top_k, query_embeddings=query_embeddings)
            except TypeError:
                first_results = search_method(query, top_k=context_top_k)
        else:
            raise RuntimeError("Indexing service does not provide search()")
        selected_results = first_results[:context_top_k]
        quality = _assess_retrieval_quality(selected_results)
        evidence = build_evidence_chunks(selected_results, max_items=context_top_k)
        retrieval = {
            "results": [asdict(result) for result in selected_results],
            "context_top_k": context_top_k,
            "intent": intent.to_dict(),
            "retrieval_query": query,
            "history_rounds": history_rounds,
            "router_history_rounds": router_history_rounds,
            "quality": quality.to_dict(),
            "first_pass_quality": quality.to_dict(),
            "second_pass": {"triggered": False, "reason": "disabled", "reasons": []},
            "embedding_errors": query_embeddings.errors if query_embeddings else {},
        }
        return selected_results, evidence, retrieval

    def _indexing_service(self) -> IndexingService:
        if self.indexing_service is None:
            self.indexing_service = IndexingService(settings_store=self.settings_store, index_dir=self.index_dir)
        return self.indexing_service

    def _answer_client(self) -> AnswerClient:
        if self.answer_client is None:
            self.answer_client = DeepSeekAnswerClient.from_settings(self.settings_store.load())
        return self.answer_client

    def _coordinate_request(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> tuple[Any, QueryEmbeddingBundle]:
        """Start routing and both query embeddings at the same time."""
        indexing_service = self._indexing_service()
        bundle = QueryEmbeddingBundle()
        dense_prefetch = getattr(indexing_service, "prefetch_dense_query_embedding", None)
        sparse_prefetch = getattr(indexing_service, "prefetch_sparse_query_embedding", None)
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="rag-request") as executor:
            route_future = executor.submit(self._route_fast, query, history)
            dense_future = executor.submit(dense_prefetch, query) if callable(dense_prefetch) else None
            sparse_future = executor.submit(sparse_prefetch, query) if callable(sparse_prefetch) else None
            try:
                fast = route_future.result()
            except Exception as exc:
                fast = self._route_fast_fallback(query, exc)
            try:
                bundle.dense = dense_future.result() if dense_future is not None else None
            except Exception as exc:
                bundle.dense_error = str(exc)[:300]
            try:
                bundle.sparse = sparse_future.result() if sparse_future is not None else None
            except Exception as exc:
                bundle.sparse_error = str(exc)[:300]
        return fast, bundle

    def _route_fast_fallback(self, query: str, exc: Exception) -> Any:
        from rag_system.generation.intent_router import FastRouteResult

        return FastRouteResult(intent=fallback_intent(
            query,
            error=str(exc)[:300],
            model=self.settings_store.load().router_model,
        ))

    def _route_intent(self, query: str, history: list[ConversationTurn]) -> IntentRouteResult:
        """单独的意图路由（向后兼容）。"""
        try:
            return self._route_fast(query, history).intent
        except Exception as exc:
            return fallback_intent(query, error=str(exc)[:300], model=self.settings_store.load().router_model)

    def _route_fast(self, query: str, history: list[ConversationTurn]):
        """合并路由+身份检测，一次 API 调用。"""
        try:
            router = self._intent_router()
            return router.route_fast(query, history)
        except Exception as exc:
            intent = fallback_intent(query, error=str(exc)[:300], model=self.settings_store.load().router_model)
            # 返回空 FastRouteResult 降级
            from rag_system.generation.intent_router import FastRouteResult
            return FastRouteResult(intent=intent)

    def _intent_router(self):
        if self._intent_router_injected:
            return self.intent_router
        return DeepSeekIntentRouter.from_settings(self.settings_store.load())

    def _identity_detector(self) -> IdentityDetector:
        if self.identity_detector is None:
            base = DeepSeekIdentityDetector.from_settings(self.settings_store.load())
            self.identity_detector = CachedIdentityDetector(base)
        return self.identity_detector

    def _detect_identity(self, query: str, history: list[ConversationTurn]) -> IdentityResult | None:
        """单独的身份检测（向后兼容，CLI/独立场景使用）。"""
        try:
            detector = self._identity_detector()
            return detector.detect(query, history=history)
        except Exception:
            return None


def _ground_citations(citation_ids: list[int], evidence) -> list[Citation]:
    by_id = {item.id: item for item in evidence}
    citations: list[Citation] = []
    for citation_id in citation_ids:
        item = by_id.get(citation_id)
        if item is not None:
            citations.append(citation_from_evidence(item))
    return citations


def _identity_from_fast(fast) -> IdentityResult | None:
    """从合并的 FastRouteResult 提取 IdentityResult。"""
    from rag_system.generation.intent_router import FastRouteResult
    if fast is None or isinstance(fast, FastRouteResult) and fast.persona == "general_public" and fast.identity_confidence < 0.3:
        return None
    return IdentityResult(
        persona=fast.persona,
        label=fast.identity_label,
        confidence=fast.identity_confidence,
        tone_guide=fast.tone_guide,
        reasoning=fast.identity_reasoning,
        model="combined-router",
    )


def _direct_answer_result(
    *,
    query: str,
    answer: str,
    reason: str,
    history_rounds: int = 0,
    router_history_rounds: int = 0,
) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer=answer,
        citations=[],
        evidence=[],
        model="local-answer-router",
        confidence="high",
        retrieval={
            "bypassed": True,
            "reason": reason,
            "context_top_k": 0,
            "results": [],
            "history_rounds": history_rounds,
            "router_history_rounds": router_history_rounds,
        },
        identity={},
    )


def _ignored_turn_result(
    *,
    query: str,
    model: str,
    signal: str,
    reason: str,
    history_rounds: int,
    router_history_rounds: int,
) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer="",
        citations=[],
        evidence=[],
        model=model,
        confidence="high",
        retrieval={
            "bypassed": True,
            "reason": "ignored_turn",
            "turn_signal": signal,
            "turn_reason": reason or "未识别到有效问题，继续聆听。",
            "context_top_k": 0,
            "results": [],
            "history_rounds": history_rounds,
            "router_history_rounds": router_history_rounds,
        },
        identity={},
    )


def _chat_answer_result(
    *,
    query: str,
    answer: str,
    model: str,
    confidence: str,
    intent: IntentRouteResult,
    history_rounds: int,
    router_history_rounds: int,
    identity: IdentityResult | None = None,
) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer=answer,
        citations=[],
        evidence=[],
        model=model,
        confidence=confidence,  # type: ignore[arg-type]
        retrieval={
            "bypassed": True,
            "reason": intent.route,
            "intent": intent.to_dict(),
            "context_top_k": 0,
            "results": [],
            "history_rounds": history_rounds,
            "router_history_rounds": router_history_rounds,
        },
        identity=identity.to_dict() if identity else {},
    )


def _stream_confidence(answer: str, *, citation_count: int) -> str:
    if citation_count:
        return "medium"
    if "没有找到足够证据" in answer or "证据不足" in answer:
        return "low"
    return "low"


def _llm_input_event(
    *,
    stage: str,
    title: str,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "type": "llm_input",
        "stage": stage,
        "title": title,
        "model": model,
        "messages": messages,
    }


def _assess_retrieval_quality(results: list[SearchResult]) -> RetrievalQuality:
    if not results:
        return RetrievalQuality(
            level="low",
            should_retry=False,
            reasons=["no_results"],
            metrics={"result_count": 0},
        )

    top = results[0]
    metadata = top.metadata
    rrf_score = _optional_float(metadata.get("rrf_score")) or float(top.score or 0.0)
    matched_token_ratio = _optional_float(metadata.get("matched_token_ratio")) or 0.0
    matched_token_count = _optional_int(metadata.get("matched_token_count")) or 0
    route_count = _optional_int(metadata.get("route_count")) or _route_count_from_metadata(metadata)
    retrieval_source = str(metadata.get("retrieval_source") or "unknown")
    metrics = {
        "result_count": len(results),
        "top_rrf_score": round(rrf_score, 6),
        "top_matched_token_ratio": round(matched_token_ratio, 6),
        "top_matched_token_count": matched_token_count,
        "top_route_count": route_count,
        "top_retrieval_source": retrieval_source,
        "thresholds": {
            "low_rrf": LOW_RRF_THRESHOLD,
            "min_matched_token_ratio": MIN_MATCHED_TOKEN_RATIO,
        },
    }

    reasons: list[str] = []
    if rrf_score < LOW_RRF_THRESHOLD:
        reasons.append("low_rrf_score")
    if route_count <= 1 and matched_token_ratio < 0.1 and rrf_score < LOW_RRF_THRESHOLD:
        reasons.append("single_route_and_weak_token_match")
    level = "low" if reasons else ("medium" if results else "low")
    return RetrievalQuality(level=level, should_retry=False, reasons=reasons or ["usable_results"], metrics=metrics)


def _route_count_from_metadata(metadata: dict[str, Any]) -> int:
    route_details = metadata.get("route_details")
    if isinstance(route_details, list):
        return len(route_details)
    routes = metadata.get("routes")
    if isinstance(routes, list):
        return len(routes)
    return sum(1 for key in ("vector_rank", "bm25_rank", "sparse_rank") if metadata.get(key))


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
