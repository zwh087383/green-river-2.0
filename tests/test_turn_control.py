from __future__ import annotations

import json
import pytest

from rag_system.generation.turn_control import (
    END_SESSION_INSTRUCTIONS, END_SESSION_TOOL, execute_end_session,
)


@pytest.mark.parametrize('arguments', [
    '{}', '[]', 'not json', '{"reason":true}', '{"reason":""}',
    '{"reason":"   "}', '{"reason":"bye","end_session":true}',
    json.dumps({'reason': 'x' * 201}),
])
def test_invalid_tool_arguments_cannot_end_session(arguments):
    with pytest.raises(ValueError):
        execute_end_session('end_conversation', arguments)


def test_only_allowlisted_tool_returns_end_result():
    with pytest.raises(ValueError):
        execute_end_session('delete_history', '{"reason":"bye"}')
    assert execute_end_session('end_conversation', '{"reason":" 明确道别 "}') == {
        'end_session': True, 'reason': '明确道别', 'status': 'finish_after_playback',
    }
    assert END_SESSION_TOOL['function']['name'] == 'end_conversation'


def test_main_prompt_distinguishes_turn_completion_from_session_ending():
    assert '不是判断你的本次回答是否说完' in END_SESSION_INSTRUCTIONS
    assert '后半句的新需求优先' in END_SESSION_INSTRUCTIONS
    assert '再见是什么意思' in END_SESSION_INSTRUCTIONS
    assert '先调用 end_conversation' in END_SESSION_INSTRUCTIONS
    assert '单纯“谢谢”' in END_SESSION_INSTRUCTIONS
