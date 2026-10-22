"""Short-lived, single-use microphone turns; audio never goes to disk."""
from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from threading import Lock, Timer

MAX_AUDIO_BYTES = 16000 * 2 * 120


def validate_pcm(pcm: bytes) -> bytes:
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise ValueError("本轮原始音频缺失或格式错误，请重新录音")
    if len(pcm) > MAX_AUDIO_BYTES:
        raise ValueError("原始音频输入最长支持 120 秒，请缩短后重试")
    return pcm


class ReplyAudioStore:
    def __init__(self, *, ttl: float = 120, capacity: int = 16):
        self.ttl, self.capacity = ttl, capacity
        self._lock = Lock()
        self._items: OrderedDict[str, tuple[float, str, bytes]] = OrderedDict()

    def put(self, text: str, pcm: bytes) -> str:
        validate_pcm(pcm)
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._items[token] = (time.monotonic() + self.ttl, text.strip(), pcm)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        timer = Timer(self.ttl, self.discard, args=(token,))
        timer.daemon = True
        timer.start()
        return token

    def discard(self, token: str) -> None:
        with self._lock:
            self._items.pop(token, None)

    def take(self, token: str, text: str) -> bytes:
        with self._lock:
            entry = self._items.pop(token, None)
        if entry is None or entry[0] <= time.monotonic() or entry[1] != text.strip():
            raise ValueError("本轮原始音频已失效或不匹配，请重新录音")
        return entry[2]


reply_audio_store = ReplyAudioStore()
