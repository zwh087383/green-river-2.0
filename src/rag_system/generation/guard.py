from __future__ import annotations

import re


ASSISTANT_IDENTITY_ANSWER = (
    "我是小绿，绿色江河环保组织的志愿讲解员。"
    "我致力于用科技解决绿色环保问题，助力绿色江河组织的环保工作。"
)

_IDENTITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^你是(?:谁|什么|干嘛的|做什么的)$",
        r"^你叫什么(?:名字)?$",
        r"^你叫啥$",
        r"^介绍(?:一下)?你自己$",
        r"^自我介绍(?:一下)?$",
        r"^你的(?:身份|名字|角色)是什么$",
        r"^whoareyou\??$",
        r"^whatareyou\??$",
        r"^who\s+are\s+you\??$",
        r"^what\s+are\s+you\??$",
    ]
]

_DOCUMENT_SCOPE_WORDS = {
    "文档",
    "文件",
    "资料",
    "知识库",
    "里面",
    "里",
    "中",
    "这份",
    "这篇",
    "上面",
    "证据",
}


def direct_answer_for_query(query: str) -> tuple[str, str] | None:
    cleaned = _normalize_query(query)
    if not cleaned:
        return None
    if _looks_document_scoped(cleaned):
        return None
    if any(pattern.match(cleaned) for pattern in _IDENTITY_PATTERNS):
        return "assistant_identity", ASSISTANT_IDENTITY_ANSWER
    return None


def _normalize_query(query: str) -> str:
    cleaned = str(query or "").strip()
    cleaned = re.sub(r"[\s，。！？!?、,.；;：:]+", "", cleaned)
    return cleaned


def _looks_document_scoped(query: str) -> bool:
    return any(word in query for word in _DOCUMENT_SCOPE_WORDS)
