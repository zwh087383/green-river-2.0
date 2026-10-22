"""Choose one existing gesture from the reply prefix available at playback start."""
from __future__ import annotations
import re


def select_answer_gesture(query: str, reply_prefix: str, tone: str = "neutral") -> str:
    text = str(reply_prefix or "").strip()[:1200]
    if not text:
        # Before generated text arrives, use only explicit visitor intent.
        if re.search(r"(?:带我|一起|想去).{0,8}(?:参观|看看|了解)|邀请", str(query)):
            return "answer_invite"
        return "answer_gesture"
    if re.search(r"(?:欢迎|邀请).{0,15}(?:参观|参加|加入|一起)|(?:请|我们).{0,8}一起|跟我来", text):
        return "answer_invite"
    if re.search(r"可以选择|有两种|各有|一方面|另一方面|不确定|暂时无法|不能确定", text):
        return "answer_open_palms"
    if tone in {"confused", "uncertain"}:
        return "answer_open_palms"
    return "answer_gesture"
