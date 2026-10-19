from __future__ import annotations

from typing import Any

from rag_system.generation.deepseek_chat import (
    ANSWER_SYSTEM_PROMPT_JSON,
    ANSWER_SYSTEM_PROMPT_STREAM,
    CHAT_SYSTEM_PROMPT_JSON,
    CHAT_SYSTEM_PROMPT_STREAM,
    FAREWELL_SYSTEM_PROMPT_STREAM,
    _answer_prompt,
    _chat_prompt,
    _history_messages,
    _stream_answer_prompt,
    _stream_chat_prompt,
)
from rag_system.generation.identity import _IDENTITY_SYSTEM_PROMPT, _build_identity_history
from rag_system.generation.intent_router import _COMBINED_SYSTEM_PROMPT, _combined_prompt
from rag_system.generation.models import ConversationTurn, EvidenceChunk, IntentRouteResult
from rag_system.settings import AppSettings


CURRENT_USER_MESSAGE = "{{ current_user_message }}"
PREVIOUS_USER_MESSAGE = "{{ conversation_history[].user }}"
PREVIOUS_ASSISTANT_MESSAGE = "{{ conversation_history[].assistant }}"


def build_llm_prompt_catalog(settings: AppSettings) -> dict[str, Any]:
    """Return every chat-completion prompt template without exposing credentials."""

    history = [
        ConversationTurn(
            user=PREVIOUS_USER_MESSAGE,
            assistant=PREVIOUS_ASSISTANT_MESSAGE,
        )
    ]
    evidence = [
        EvidenceChunk(
            id=1,
            search_result_id="{{ search_result_id }}",
            text="{{ retrieved_evidence_text }}",
            source_name="{{ source_name }}",
            chunk_id="{{ chunk_id }}",
            heading_path="{{ heading_path }}",
            page="{{ page_number }}",
            section_index=0,
            chunk_index=0,
        )
    ]
    intent = IntentRouteResult(
        route="rag_retrieve",
        needs_retrieval=True,
        confidence=0.0,
        reason="{{ route_reason }}",
        standalone_query="{{ standalone_query }}",
        rewrite_hints=["{{ rewrite_hint }}"],
        model=settings.router_model,
        method="dashscope",
    )
    history_messages = _history_messages(history)
    identity_history_messages = _build_identity_history(history)
    answer_model = settings.answer_model
    router_model = settings.router_model
    rewrite_model = router_model

    return {
        "prompts": [
            _entry(
                "combined_router",
                "前置分析：轮次、路由与身份",
                router_model,
                "JSON",
                [
                    _message("system", _COMBINED_SYSTEM_PROMPT),
                    _message(
                        "user",
                        _combined_prompt(
                            query=CURRENT_USER_MESSAGE,
                            history=history,
                            turn_prompt=settings.deepseek_turn_prompt,
                        ),
                    ),
                ],
            ),
            _entry(
                "rag_answer_stream",
                "RAG 回答（Web 流式）",
                answer_model,
                "流式文本",
                [
                    _message("system", ANSWER_SYSTEM_PROMPT_STREAM),
                    *_repeatable_history(history_messages),
                    _message(
                        "user",
                        _stream_answer_prompt(query=CURRENT_USER_MESSAGE, evidence=evidence),
                    ),
                ],
            ),
            _entry(
                "rag_answer_json",
                "RAG 回答（JSON）",
                answer_model,
                "JSON",
                [
                    _message("system", ANSWER_SYSTEM_PROMPT_JSON),
                    *_repeatable_history(history_messages),
                    _message(
                        "user",
                        _answer_prompt(query=CURRENT_USER_MESSAGE, evidence=evidence),
                    ),
                ],
            ),
            _entry(
                "chat_answer_stream",
                "直接对话（Web 流式）",
                answer_model,
                "流式文本",
                [
                    _message("system", CHAT_SYSTEM_PROMPT_STREAM),
                    *_repeatable_history(history_messages),
                    _message(
                        "user",
                        _stream_chat_prompt(query=CURRENT_USER_MESSAGE, intent=intent),
                    ),
                ],
            ),
            _entry(
                "chat_answer_json",
                "直接对话（JSON）",
                answer_model,
                "JSON",
                [
                    _message("system", CHAT_SYSTEM_PROMPT_JSON),
                    *_repeatable_history(history_messages),
                    _message(
                        "user",
                        _chat_prompt(query=CURRENT_USER_MESSAGE, intent=intent),
                    ),
                ],
            ),
            _entry(
                "farewell_stream",
                "会话结束语",
                answer_model,
                "流式文本",
                [
                    _message("system", FAREWELL_SYSTEM_PROMPT_STREAM),
                    *_repeatable_history(history_messages),
                    _message("user", "请生成结束语。"),
                ],
            ),
            _entry(
                "identity_compatibility",
                "身份检测（兼容路径）",
                getattr(settings, "identity_model", rewrite_model),
                "JSON",
                [
                    _message("system", _IDENTITY_SYSTEM_PROMPT),
                    *_repeatable_history(identity_history_messages),
                    _message(
                        "user",
                        f"用户提问：\n{CURRENT_USER_MESSAGE}\n\n请判断用户身份。",
                    ),
                ],
            ),
        ]
    }


def _entry(
    prompt_id: str,
    title: str,
    model: str,
    mode: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": prompt_id,
        "title": title,
        "model": model,
        "mode": mode,
        "messages": messages,
    }


def _message(role: str, content: str, *, repeatable: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": role, "content": content}
    if repeatable:
        payload["repeatable"] = True
    return payload


def _repeatable_history(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        _message(message["role"], message["content"], repeatable=True)
        for message in messages
    ]
