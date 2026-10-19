from __future__ import annotations
from rag_system.generation.service import build_direct_messages
from rag_system.settings import AppSettings


def build_llm_prompt_catalog(settings: AppSettings) -> dict:
    return {"prompts": [{
        "id": "direct_audio_reply", "title": "Qwen Audio 直接回复",
        "model": settings.answer_model, "mode": "流式文字与音频",
        "messages": build_direct_messages("{{ current_user_message }}", [], []),
    }]}
