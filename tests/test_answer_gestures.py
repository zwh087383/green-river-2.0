import pytest
from rag_system.answer_gestures import select_answer_gesture


@pytest.mark.parametrize('query,text,tone,purpose', [
    ('介绍长江源', '长江源位于青藏高原。', 'neutral', 'answer_gesture'),
    ('想了解保护工作', '欢迎一起参观我们的展厅。', 'happy', 'answer_invite'),
    ('怎么办', '目前有两种方案可以选择。', 'neutral', 'answer_open_palms'),
    ('怎么办', '暂时无法确定具体原因。', 'neutral', 'answer_open_palms'),
    ('带我参观一下', '', 'neutral', 'answer_invite'),
    ('介绍长江源', '', 'shy', 'answer_gesture'),
    ('比心飞吻', '这是一项生态保护工作。', 'happy', 'answer_gesture'),
    ('那边在哪', '请看远方的山。', 'neutral', 'answer_gesture'),
])
def test_semantic_gesture_uses_existing_actions_without_unlocated_pointing(query,text,tone,purpose):
    assert select_answer_gesture(query,text,tone)==purpose
