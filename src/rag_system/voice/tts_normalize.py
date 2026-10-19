"""TTS 文本规范化 — 播报前预处理数字和多音字。

规则：
- 公路编号 "318" → "三幺八"（逐位读，"1"读"幺"，不加"百十"）
- 国道 "G109" → "G幺零九"
- 具体年份 "1940年" → "一九四零年"（按位读）
- 普通数量 "近30年" → "近三十年"（按数值读）
- 多音字按上下文替换为 TTS 能稳定读对的同音字
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── 公路/路线编号正则 ────────────────────────────────────────────────

# 匹配需要逐位读的公路编号场景
_HIGHWAY_PATTERN = re.compile(
    r"G\s*(\d{3,4})"                    # G109, G 318
    r"|国道\s*(\d{3,4})"                 # 国道109, 国道 318
    r"|省道\s*(\d{3,4})"                 # 省道213
    r"|(\d{3,4})\s*(?:国道|公路|线)"     # 109国道, 318公路, 109线
    r"|青藏(?:公路|线)\s*(\d{3,4})"     # 青藏公路318
    r"|川藏(?:公路|线)\s*(\d{3,4})"     # 川藏公路318
    r"|G(\d{3,4})(?:国道|公路|线)",      # G318国道
    re.IGNORECASE,
)

# 数字→中文逐位读法
_DIGIT_MAP: dict[str, str] = {
    "0": "零", "1": "幺", "2": "二", "3": "三",
    "4": "四", "5": "五", "6": "六", "7": "七",
    "8": "八", "9": "九",
}

# 年份使用普通的“一”，不用公路/编号读法中的“幺”。
_YEAR_DIGIT_MAP: dict[str, str] = {
    "0": "零", "1": "一", "2": "二", "3": "三",
    "4": "四", "5": "五", "6": "六", "7": "七",
    "8": "八", "9": "九",
}

_YEAR_PATTERN = re.compile(r"(?<!\d)(\d{4})(\s*年)")

# 只转换明确带量词/时间单位的整数，避免把型号、文件名或小数误当数量。
_QUANTITY_PATTERN = re.compile(
    r"(?<![\d.])(\d{1,12})"
    r"(?=\s*(?:个月|小时|分钟|公里|千米|年|月|天|日|米|只|个|次|人|项|座|处|种|条|份|倍|%|％))"
)

_CN_DIGITS = "零一二三四五六七八九"
_SECTION_UNITS = ("", "十", "百", "千")
_LARGE_UNITS = ("", "万", "亿", "兆")


# ── 多音字上下文规则 ─────────────────────────────────────────────

# Qwen Realtime TTS 和 Edge-TTS 的纯文本接口都不能可靠接收 SSML
# phoneme。对项目领域中最重要的多音词，改用同音字只影响送给 TTS 的
# 文本，不改变界面展示或知识库原文。
_PRONUNCIATION_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"藏羚羊"), "葬羚羊", "藏羚羊:zang4"),
)


# ── 引用标记 ──────────────────────────────────────────────────────

_CITATION_PATTERN = re.compile(r"【\d+】|\[\d+\]")

# ── Markdown 格式标记 ─────────────────────────────────────────────

_MARKDOWN_PATTERN = re.compile(r"\*\*(.+?)\*\*|(?<!\*)\*(?!\*)(.+?)\*(?!\*)|__(.+?)__")

# ── 时间/数量区间连接符 ──────────────────────────────────────────
# "11月-次年4月" → "11月至次年4月"，"8-10月" → "8至10月"

_RANGE_DASH_PATTERN = re.compile(
    r"(\d+\s*(?:月|天|日|年|公里|米|小时|分钟|只|个|次|万)?)"
    r"\s*-\s*"
    r"(?=\d|[次去上今明])",
)


# ── 数字读法分类 ─────────────────────────────────────────────────

def _digit_by_digit(digits: str) -> str:
    """逐位读数，如 "318" → "三一八"。"""
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in digits)


def _normalize_highway(match: re.Match) -> str:
    """公路编号替换 — 提取数字替换为逐位读法，保留前缀。"""
    matched = match.group(0)
    digits_only = re.findall(r"\d{3,4}", matched)
    for d in digits_only:
        matched = matched.replace(d, _digit_by_digit(d))
    return matched


def _normalize_year(match: re.Match[str]) -> str:
    """具体年份按位读，如 1940年 → 一九四零年。"""
    digits, suffix = match.groups()
    return "".join(_YEAR_DIGIT_MAP[digit] for digit in digits) + suffix


def _section_to_chinese(value: int) -> str:
    """把 0..9999 的一节整数转成中文数值读法。"""
    output: list[str] = []
    pending_zero = False
    for position in range(3, -1, -1):
        divisor = 10 ** position
        digit = value // divisor % 10
        if digit:
            if pending_zero and output:
                output.append("零")
            output.append(_CN_DIGITS[digit] + _SECTION_UNITS[position])
            pending_zero = False
        elif output and value % divisor:
            pending_zero = True
    return "".join(output)


def _integer_to_chinese(digits: str) -> str:
    """按数值读整数，如 30 → 三十、1001 → 一千零一。"""
    value = int(digits)
    if value == 0:
        return "零"

    sections: list[int] = []
    while value:
        sections.append(value % 10_000)
        value //= 10_000
    if len(sections) > len(_LARGE_UNITS):
        return digits

    output: list[str] = []
    pending_zero = False
    for index in range(len(sections) - 1, -1, -1):
        section = sections[index]
        if section == 0:
            if output:
                pending_zero = True
            continue
        if output and (pending_zero or section < 1000):
            if output[-1] != "零":
                output.append("零")
        output.append(_section_to_chinese(section) + _LARGE_UNITS[index])
        pending_zero = False

    normalized = "".join(output)
    if normalized.startswith("一十"):
        normalized = normalized[1:]
    return normalized


def _normalize_quantity(match: re.Match[str]) -> str:
    return _integer_to_chinese(match.group(1))


# ── 主入口 ────────────────────────────────────────────────────────

@dataclass(slots=True)
class TTSNormalizeResult:
    original: str
    normalized: str
    changes: list[str] = field(default_factory=list)


def normalize_for_tts(text: str) -> TTSNormalizeResult:
    """对 TTS 输入文本做规范化。

    处理:
    1. 公路编号 → 逐位读（G109→G一零九）
    2. 具体年份 → 普通数字按位读（1940年→一九四零年）
    3. 带单位的数量 → 按数值读（近30年→近三十年）
    4. 项目领域多音字 → TTS 发音提示字（藏羚羊→葬羚羊）
    """
    result = _CITATION_PATTERN.sub("", text)
    result = _MARKDOWN_PATTERN.sub(r"\1\2\3", result)
    result = _RANGE_DASH_PATTERN.sub(r"\1至", result)
    changes: list[str] = []
    if result != text:
        changes.append("citations_removed")
    before_highway = result
    result = _HIGHWAY_PATTERN.sub(_normalize_highway, result)
    if result != before_highway:
        changes.append("highway_number_normalized")
    before_year = result
    result = _YEAR_PATTERN.sub(_normalize_year, result)
    if result != before_year:
        changes.append("calendar_year_digit_normalized")
    before_quantity = result
    result = _QUANTITY_PATTERN.sub(_normalize_quantity, result)
    if result != before_quantity:
        changes.append("quantity_number_normalized")
    for pattern, replacement, change_name in _PRONUNCIATION_RULES:
        updated = pattern.sub(replacement, result)
        if updated != result:
            changes.append(change_name)
            result = updated
    return TTSNormalizeResult(original=text, normalized=result.strip(), changes=changes)


def wrap_ssml(text: str, *, voice: str = "", rate: str = "+0%") -> str:
    """用 SSML 包裹文本以精细控制 Edge TTS 发音。"""
    voice_attr = f' name="{voice}"' if voice else ""
    rate_attr = f' rate="{rate}"' if rate else ""
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="zh-CN">'
        f'<voice{voice_attr}>'
        f'<prosody{rate_attr}>'
        f"{text}"
        f"</prosody>"
        f"</voice>"
        f"</speak>"
    )
