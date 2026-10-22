"""Bounded PCM counters for diagnostics; never stores audio samples."""
from __future__ import annotations
import math
import struct
import time
from rag_system.voice_diagnostics import trace_voice


class PcmProbe:
    def __init__(self, stage: str, **context):
        self.stage, self.context = stage, context
        self.frames = self.samples = self.squares = self.peak = 0
        self.last = 0.

    def feed(self, pcm: bytes, *, hit: bool = False):
        self.frames += 1
        for (sample,) in struct.iter_unpack('<h', pcm):
            self.samples += 1
            self.squares += sample * sample
            self.peak = max(self.peak, abs(sample))
        now = time.monotonic()
        if hit or now - self.last >= 1:
            trace_voice(self.stage, **self.context, frames=self.frames,
                        audio_ms=self.samples / 16,
                        rms=round(math.sqrt(self.squares / max(1, self.samples)), 2),
                        peak=self.peak, hit=hit)
            self.last = now
            self.samples = self.squares = self.peak = 0
