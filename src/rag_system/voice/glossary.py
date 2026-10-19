"""绿色江河领域词表 — 被 STT 纠错、TTS 规范化、身份识别共用。

维护项目专属名词、地名、人名和技术术语，减少语音识别同音误判和概念混淆。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# ── 领域专有名词（语音识别优先纠错对象）─────────────────────────────

# 格式: ("常见误识结果", "正确词")
# 按优先级排序 — 越长的短语越先匹配，避免子串误替换
_HOMOPHONE_CORRECTIONS: list[tuple[str, str]] = [
    # 项目/行动名称
    ("垃圾换食品", "垃圾换食品"),
    ("垃圾换十品", "垃圾换食品"),
    ("垃圾换饰品", "垃圾换食品"),
    ("拉机换食品", "垃圾换食品"),
    ("拉基换食品", "垃圾换食品"),
    ("垃圾换食物", "垃圾换食品"),

    ("青藏绿色驿站", "青藏绿色驿站"),
    ("青橙色驿站", "青藏绿色驿站"),
    ("青橙绿色驿站", "青藏绿色驿站"),
    ("青藏绿色医站", "青藏绿色驿站"),
    ("青藏绿色一战", "青藏绿色驿站"),

    ("绿色江河", "绿色江河"),
    ("绿色江和", "绿色江河"),
    ("绿色江合", "绿色江河"),

    ("索南达杰", "索南达杰"),
    ("索兰达杰", "索南达杰"),
    ("索南大街", "索南达杰"),
    ("锁南达杰", "索南达杰"),

    ("斑头雁", "斑头雁"),
    ("搬头雁", "斑头雁"),
    ("半头雁", "斑头雁"),

    ("可可西里", "可可西里"),
    ("壳壳西里", "可可西里"),
    ("可客西里", "可可西里"),

    # 地名
    ("沱沱河", "沱沱河"),
    ("拖拖河", "沱沱河"),
    ("陀陀河", "沱沱河"),

    ("格拉丹东", "格拉丹东"),
    ("格拉丹冬", "格拉丹东"),
    ("格拉丁东", "格拉丹东"),

    ("姜根迪如", "姜根迪如"),
    ("江根迪如", "姜根迪如"),

    ("唐古拉山", "唐古拉山"),
    ("唐三拉山", "唐古拉山"),

    ("格尔木", "格尔木"),
    ("格耳目", "格尔木"),

    ("玉树", "玉树"),
    ("遇数", "玉树"),

    # 常见 STT 误识（通用汉语）
    ("邮局", "邮局"),
    ("有局", "邮局"),
    ("油局", "邮局"),

    ("驿站", "驿站"),
    ("一站", "驿站"),
    ("依站", "驿站"),
    ("益站", "驿站"),

    ("保护站", "保护站"),
    ("宝户站", "保护站"),
    ("保户站", "保护站"),

    ("藏羚羊", "藏羚羊"),
    ("苍蝇羊", "藏羚羊"),
    ("藏羚阳", "藏羚羊"),

    ("野牦牛", "野牦牛"),
    ("野毛牛", "野牦牛"),
    ("夜髦牛", "野牦牛"),

    ("冰川", "冰川"),
    ("冰船", "冰川"),
    ("兵川", "冰川"),

    ("长江源", "长江源"),
    ("长江圆", "长江源"),
    ("长江园", "长江源"),

    # 数字/代码常见误识
    ("G1零9", "G109"),
    ("G10 9", "G109"),
    ("G一零九", "G109"),
    ("g109", "G109"),
    ("记109", "G109"),
    ("G四二零", "G420"),
    ("G三幺八", "G318"),
    ("G318", "G318"),
]

# 整理为按长度降序（长串优先匹配，避免子串误伤）
_HOMOPHONE_CORRECTIONS.sort(key=lambda x: len(x[0]), reverse=True)


# ── 领域词条（注入身份检测 prompt）───────────────────────────────────

_DOMAIN_TERMS: list[str] = [
    # 项目/行动名称
    "垃圾换食品",
    "带走一袋垃圾",
    "青藏绿色驿站",
    "长江源水生态保护站",
    "索南达杰自然保护站",
    "斑头雁守护与监测",
    "可可西里生态保护",
    "长江源冰川监测",
    "志愿者项目",
    "绿色江河环境保护促进会",

    # 地名
    "沱沱河", "格拉丹东", "姜根迪如", "唐古拉山",
    "格尔木", "玉树", "可可西里", "青藏高原",
    "长江源", "黄河源", "澜沧江源",

    # 组织
    "绿色江河", "Green River",
    "中国科学院青藏高原研究所",
    "三江源国家公园管理局",
    "青海省生态环境厅",
    "四川省林业和草原局",
    "世界自然基金会", "WWF",
    "阿拉善SEE基金会",
    "中国绿化基金会",
    "联合国开发计划署", "UNDP",

    # 术语
    "高原反应", "藏羚羊", "野牦牛", "藏野驴",
    "雪豹", "黑颈鹤", "斑头雁", "高原鼠兔",
    "高寒草甸", "高寒湿地", "多年冻土",
    "冰芯", "退缩速率", "年径流量",
    "志愿时长", "保护站", "驿站",
]


# ── 数据结构 ────────────────────────────────────────────────────────

@dataclass(slots=True)
class CorrectionResult:
    original: str
    corrected: str
    corrections: list[dict[str, str]] = field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return len(self.corrections) > 0


# ── API ──────────────────────────────────────────────────────────

def correct_stt_text(
    text: str,
    custom_rules: Sequence[tuple[str, str]] | None = None,
) -> CorrectionResult:
    """对语音识别的中文文本做领域纠错。

    自定义规则覆盖同名内置规则，所有误词按长度降序匹配，并基于
    原始文本一次性替换，避免 ``A -> B``、``B -> C`` 发生连锁纠错。
    """
    replacements = dict(_HOMOPHONE_CORRECTIONS)
    for wrong, correct in custom_rules or ():
        replacements[str(wrong)] = str(correct)
    ordered = sorted(replacements, key=len, reverse=True)
    if not ordered:
        return CorrectionResult(original=text, corrected=text)

    pattern = re.compile("|".join(re.escape(wrong) for wrong in ordered))
    applied: list[dict[str, str]] = []
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        wrong = match.group(0)
        correct = replacements[wrong]
        if wrong != correct and wrong not in seen:
            seen.add(wrong)
            applied.append({"wrong": wrong, "correct": correct})
        return correct

    result = pattern.sub(replace, text)
    return CorrectionResult(original=text, corrected=result, corrections=applied)


def get_domain_glossary() -> list[str]:
    """返回领域词条列表，供身份检测 prompt 注入。"""
    return list(_DOMAIN_TERMS)


def get_domain_glossary_text() -> str:
    """返回领域词条的一句话描述，注入 system prompt。"""
    items = get_domain_glossary()
    return "、".join(items[:30])  # 控制长度
