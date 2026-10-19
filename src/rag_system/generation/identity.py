"""提问者身份识别 — 用 DeepSeek 判断用户身份，生成语气调整指令。

通过分析提问的用词、语法复杂度、关注点来推断提问者是什么人（小孩、
专家、游客等），然后输出对应的回答风格指南，注入到下游回答生成环节。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_system.generation.models import ConversationTurn
from rag_system.settings import AppSettings


# ── 身份类型定义 ────────────────────────────────────────────────

@dataclass(slots=True)
class IdentityResult:
    """一次身份识别的结果"""

    persona: str  # child | expert | tourist | volunteer_applicant | student | general_public
    label: str  # 中文标签，如"小朋友"、"研究人员"、"游客"、"志愿者申请人"、"学生"、"普通用户"
    confidence: float  # 0.0–1.0
    tone_guide: str  # 注入到 answer prompt 的语气与详略指令
    reasoning: str = ""  # 判断依据，便于调试
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "label": self.label,
            "confidence": self.confidence,
            "tone_guide": self.tone_guide,
            "reasoning": self.reasoning,
            "model": self.model,
        }


# ── 各身份的默认语气指南 ──────────────────────────────────────────

_PERSONA_PROFILES: dict[str, dict[str, str]] = {
    "child": {
        "label": "小朋友",
        "tone_guide": (
           "优先基于用户的提问进行回答，不要说废话，回复尽量精炼，直接且正面的进行回复"
            "提问者是一个小朋友。请用非常简单易懂的语言回答，用生活化的比喻解释概念。"
            "句子要短，每句话不超过 25 个字。不要用专业术语，如果必须用到要跟着解释。"
            "回答控制在 3–5 句话以内，语气亲切温暖。可以在回答末尾加一句鼓励探索的话。"
        ),
    },
    "expert": {
        "label": "研究人员/专业人士",
        "tone_guide": (
            "优先基于用户的提问进行回答，不要说废话，回复尽量精炼，直接且正面的进行回复"
            "提问者是一位研究人员或专业人士。请用精炼、学术化的语言回答，"
            "优先给出具体数据和年份。可以使用专业术语。按逻辑层次组织答案，"
            "先给核心结论再展开细节。如果有数据不确定的，明确标注不确定性。"
        ),
    },
    "tourist": {
        "label": "游客",
        "tone_guide": (
            "优先基于用户的提问进行回答，不要说废话，回复尽量精炼，直接且正面的进行回复"
            "提问者是一位准备去或正在青藏高原的游客。请用热情友好的语气回答，"
            "优先给出实用建议（路线、季节、装备、注意事项）。可以描述一些自然景观的壮美细节。"
            "重点回答怎么做、什么时候去、带什么，不需要过多理论背景。"
        ),
    },
    "volunteer_applicant": {
        "label": "志愿者申请人",
        "tone_guide": (
            "优先基于用户的提问进行回答，不要说废话，回复尽量精炼，直接且正面的进行回复"
            "提问者有意向成为绿色江河志愿者。请用鼓励但实事求是的语气回答，"
            "详细说明申请条件、流程、工作内容和注意事项。强调高原环境的艰苦和需要做好身体准备。"
            "不要美化志愿工作的困难，但可以分享志愿者的成就感。"
        ),
    },
    "student": {
        "label": "学生",
        "tone_guide": (
"优先基于用户的提问进行回答，不要说废话，回复尽量精炼，直接且正面的进行回复"
            "提问者是一位学生。请用清晰的结构化方式回答，先给出核心知识点，再展开说明原因和背景。"
            "可以适当使用加粗标题分层、分点列举。对专业名词做简短解释。"
            "鼓励进一步思考和提问，在末尾可以提一个延伸问题引导深入学习。"
        ),
    },
    "general_public": {
        "label": "普通公众",
        "tone_guide": (
"优先基于用户的提问进行回答，不要说废话，回复尽量精炼，直接且正面的进行回复"
            "提问者是普通公众。请用平实易懂的语言回答，避免学术黑话。"
            "语气自然友好，像朋友聊天一样。"
        ),
    },
}

# 兜底
_FALLBACK_PROFILE = {
    "label": "普通用户",
"优先基于用户的提问进行回答，不要说废话，回复尽量精炼，直接且正面的进行回复"
    "tone_guide": "提问者身份不明确，请用中性友好的语气回答。",
}


# ── 检测协议 ──────────────────────────────────────────────────────

class IdentityDetector(Protocol):
    """身份检测器协议。返回 IdentityResult。"""

    model: str

    def detect(self, query: str, *, history: list[ConversationTurn] | None = None) -> IdentityResult:
        ...


# ── DeepSeek 实现 ──────────────────────────────────────────────────

_IDENTITY_SYSTEM_PROMPT = """你是用户身份分析器。根据用户的提问内容、用词习惯、语气和关注点，判断提问者最可能是什么身份。

▎重要：以下名称是绿色江河环保组织的**项目或行动名称，不是品牌或商品**：
垃圾换食品、带走一袋垃圾、青藏绿色驿站、长江源水生态保护站、
索南达杰自然保护站、斑头雁守护与监测、可可西里生态保护、
长江源冰川监测、绿色江河环境保护促进会

▎可选身份（严格从下列选一个）：
- child：用词简单、提问直接、可能有拼音或错别字、问"是什么""为什么"
- expert：使用专业术语、追问具体数据或方法、引用文献或年份、提问深入
- tourist：询问路线/季节/景点/装备/注意事项/能不能/怎么去/什么时候适合
- volunteer_applicant：询问报名条件/流程/待遇/需要什么技能/工作内容
- student：提问结构清晰像在做作业或报告、问原因/影响/措施、用"请问"
- general_public：闲聊式提问、简单好奇、不涉及专业细节、语气随意

▎特别注意：
- 用户提到"垃圾换食品""绿色驿站"等是问环保项目，不是品牌推广
- 如果用户问题中带同音错字（如"有局"=邮局、"驿站"=驿站），按正确语义判断

只输出 JSON，不要输出 Markdown。JSON 字段：
- persona: 上述六个值之一
- confidence: 0.0–1.0
- reasoning: 一句话说明判断依据"""


class DeepSeekIdentityDetector:
    """用 DeepSeek 判断提问者身份。"""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "DeepSeekIdentityDetector":
        if not settings.deepseek_api_key:
            raise RuntimeError("DeepSeek API Key is not configured")
        return cls(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=getattr(settings, "identity_model", settings.router_model),
        )

    def detect(self, query: str, *, history: list[ConversationTurn] | None = None) -> IdentityResult:
        payload = self._call(query, history or [])
        persona = str(payload.get("persona") or "general_public").strip().lower()
        confidence = _clamp_confidence(float(payload.get("confidence") or 0.5))
        reasoning = str(payload.get("reasoning") or "").strip()

        if persona not in _PERSONA_PROFILES:
            persona = "general_public"

        profile = _PERSONA_PROFILES.get(persona, _FALLBACK_PROFILE)
        return IdentityResult(
            persona=persona,
            label=str(profile["label"]),
            confidence=confidence,
            tone_guide=str(profile["tone_guide"]),
            reasoning=reasoning,
            model=self.model,
            raw=payload,
        )

    def _call(self, query: str, history: list[ConversationTurn]) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for identity detection") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _IDENTITY_SYSTEM_PROMPT},
                *_build_identity_history(history),
                {"role": "user", "content": f"用户提问：\n{query}\n\n请判断用户身份。"},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content if response.choices else ""
        try:
            parsed = json.loads(content or "{}")
        except json.JSONDecodeError:
            return _FALLBACK_PAYLOAD
        if not isinstance(parsed, dict):
            return _FALLBACK_PAYLOAD
        return parsed


# ── 缓存包装器 ─────────────────────────────────────────────────────

class CachedIdentityDetector:
    """对另一检测器做同会话缓存，同一问题（文本 hash）不重复调用 LLM。"""

    def __init__(self, inner: IdentityDetector) -> None:
        self._inner = inner
        self._cache: dict[int, IdentityResult] = {}

    @property
    def model(self) -> str:
        return self._inner.model

    def detect(self, query: str, *, history: list[ConversationTurn] | None = None) -> IdentityResult:
        qhash = hash(query.strip())
        cached = self._cache.get(qhash)
        if cached is not None:
            return cached
        result = self._inner.detect(query, history=history)
        self._cache[qhash] = result
        return result


# ── 辅助函数 ───────────────────────────────────────────────────────

_FALLBACK_PAYLOAD: dict[str, Any] = {
    "persona": "general_public",
    "confidence": 0.3,
    "reasoning": "身份检测模型调用失败，回退为普通用户",
}


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _build_identity_history(history: list[ConversationTurn]) -> list[dict[str, str]]:
    """取最近几轮对话帮助理解上下文。"""
    messages: list[dict[str, str]] = []
    for turn in history[-3:]:
        if turn.user.strip():
            messages.append({"role": "user", "content": turn.user[:200]})
        if turn.assistant.strip():
            messages.append({"role": "assistant", "content": turn.assistant[:120]})
    return messages
