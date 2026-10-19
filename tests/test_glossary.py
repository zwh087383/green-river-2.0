from __future__ import annotations

from rag_system.voice.glossary import correct_stt_text


def test_custom_asr_correction_overrides_same_builtin_key() -> None:
    result = correct_stt_text(
        "绿色江和是什么组织",
        custom_rules=(("绿色江和", "绿色长河"),),
    )

    assert result.corrected == "绿色长河是什么组织"
    assert result.corrections == [
        {"wrong": "绿色江和", "correct": "绿色长河"}
    ]


def test_asr_correction_prefers_longest_matching_source_text() -> None:
    result = correct_stt_text(
        "介绍一下绿色江河",
        custom_rules=(
            ("江河", "河流"),
            ("绿色江河", "绿色组织"),
        ),
    )

    assert result.corrected == "介绍一下绿色组织"


def test_asr_correction_does_not_cascade_replacement_output() -> None:
    result = correct_stt_text(
        "甲乙",
        custom_rules=(
            ("甲", "乙"),
            ("乙", "丙"),
        ),
    )

    # Each rule applies only to the original input span. The generated first
    # replacement must not be consumed by the second rule.
    assert result.corrected == "乙丙"
