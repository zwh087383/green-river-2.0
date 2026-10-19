from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from rag_system.settings import AppSettings


QUERY_REWRITE_SYSTEM_PROMPT = (
    "你是 RAG 检索系统的 Query Rewrite 模块。"
    "只输出 JSON，不要解释。目标是把用户问题改写成更适合向量检索和 BM25 检索的短查询。"
)


class QueryRewriteClient(Protocol):
    model: str

    def rewrite(self, query: str) -> "QueryRewriteResult":
        ...


@dataclass(slots=True)
class QueryRewriteResult:
    original_query: str
    rewritten_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    intent: str = ""
    method: str = "fallback"
    model: str = ""
    error: str | None = None

    @property
    def queries(self) -> list[str]:
        seen: set[str] = set()
        queries: list[str] = []
        for query in [self.original_query, *self.rewritten_queries]:
            cleaned = _clean_query(query)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                queries.append(cleaned)
        return queries

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["queries"] = self.queries
        return payload


class DeepSeekQueryRewriteClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str,
        max_rewrites: int = 5,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_rewrites = max(1, min(max_rewrites, 8))

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "DeepSeekQueryRewriteClient":
        if not settings.deepseek_api_key:
            raise MissingDeepSeekApiKeyError("DeepSeek API Key is not configured")
        return cls(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=getattr(settings, "router_model", settings.answer_model),
        )

    def rewrite(self, query: str) -> QueryRewriteResult:
        cleaned = _clean_query(query)
        if not cleaned:
            return QueryRewriteResult(original_query=query, error="empty_query")
        payload = self._call_deepseek(cleaned)
        rewrites = _normalise_string_list(payload.get("rewrites"), limit=self.max_rewrites)
        keywords = _normalise_string_list(payload.get("keywords"), limit=16)
        intent = str(payload.get("intent") or "").strip()
        rewrites = [item for item in rewrites if item.casefold() != cleaned.casefold()]
        return QueryRewriteResult(
            original_query=cleaned,
            rewritten_queries=rewrites,
            keywords=keywords,
            intent=intent,
            method="deepseek",
            model=self.model,
        )

    def _call_deepseek(self, query: str) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for DeepSeek query rewrite") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0)
        response = client.chat.completions.create(
            model=self.model,
            messages=build_query_rewrite_messages(query, self.max_rewrites),
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content if response.choices else ""
        try:
            payload = json.loads(content or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("DeepSeek query rewrite returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("DeepSeek query rewrite returned non-object JSON")
        return payload


class MissingDeepSeekApiKeyError(RuntimeError):
    pass


def fallback_rewrite(query: str, *, error: str | None = None, model: str = "") -> QueryRewriteResult:
    cleaned = _clean_query(query)
    return QueryRewriteResult(
        original_query=cleaned,
        rewritten_queries=[],
        keywords=[],
        method="fallback",
        model=model,
        error=error,
    )


def _rewrite_prompt(query: str, max_rewrites: int) -> str:
    return f"""
用户原始问题：
{query}

请输出 JSON 对象，字段如下：
- rewrites: 字符串数组，{max_rewrites} 条以内，每条 3 到 16 个词，适合检索，不要写完整回答。
- keywords: 字符串数组，提取模型名、文件名、接口名、页码、数字、中文核心名词、英文缩写。
- intent: 一句话概括用户要找什么。

改写要求：
- 保留专有名词、模型名、路径、文件名，例如 text-embedding-v4、qwen3-vl-embedding、word/media、ROS2。
- 中文问题可以补充英文检索词；英文问题可以补充中文检索词。
- 需要同时照顾语义向量检索和 BM25 关键词检索。
- 不要编造不存在的文件名、页码、API Key 或配置。
- 不要输出 Markdown。

输出示例：
{{"rewrites":["docx 内嵌图片 OCR","word/media image extraction"],"keywords":["docx","内嵌图片","OCR","word/media"],"intent":"查找 Word 内嵌图片解析和入库相关内容"}}
""".strip()


def build_query_rewrite_messages(query: str, max_rewrites: int = 5) -> list[dict[str, str]]:
    """Build the exact messages used for query rewriting."""

    return [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": _rewrite_prompt(query, max_rewrites)},
    ]


def _normalise_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _clean_query(str(item))
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            items.append(cleaned)
        if len(items) >= limit:
            break
    return items


def _clean_query(value: str) -> str:
    return " ".join(str(value or "").strip().split())
