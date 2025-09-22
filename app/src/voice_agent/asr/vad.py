# src/asr/vad.py
from typing import List, Optional
from collections import deque
import numpy as np

class VADSegmenter:
    """
    Energy-based endpointing for 16k mono int16 PCM frames (10/20/30 ms).
    Warmup calibrates ambient noise; pre-roll keeps a bit of audio before speech start.
    Emits an utterance (bytes) after trailing silence exceeds threshold.
    """
    def __init__(
        self,
        frame_ms: int = 20,
        silence_tail_ms: int = 400,
        sample_rate: int = 16000,
        rms_floor: float = 60.0,    # ambient noise floor (auto-raised during warmup)
        ema_alpha: float = 0.9,
        thr_mult: float = 1.6,      # speech threshold = max(floor, ema) * thr_mult
        warmup_ms: int = 400,       # listen before starting detection
        pre_roll_ms: int = 240,     # prepend this much audio to utterances
        min_utter_ms: int = 300     # drop utterances shorter than this
    ):
        assert frame_ms in (10, 20, 30), "frame_ms must be 10/20/30"
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_tail_frames = max(1, silence_tail_ms // frame_ms)
        self.min_utter_frames = max(1, min_utter_ms // frame_ms)
        self.warmup_frames = max(1, warmup_ms // frame_ms)
        self.pre_frames = max(0, pre_roll_ms // frame_ms)

        self.buffer: List[bytes] = []
        self.prebuf: deque[bytes] = deque(maxlen=self.pre_frames)
        self.trailing_silence = 0
        self.seen_frames = 0
        self.speech_active = False

        self._ema = 0.0
        self._alpha = ema_alpha
        self._floor = rms_floor
        self._mult = thr_mult

    def reset(self):
        self.buffer.clear()
        self.trailing_silence = 0
        self.speech_active = False

    def _rms(self, frame: bytes) -> float:
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(x * x)) + 1e-8)

    def _is_speech(self, frame: bytes) -> bool:
        rms = self._rms(frame)
        self._ema = self._alpha * self._ema + (1 - self._alpha) * rms
        thr = max(self._floor, self._ema) * self._mult
        return rms >= thr

    def push(self, frame: bytes) -> Optional[bytes]:
        self.prebuf.append(frame)
        # Warmup: collect ambient, raise floor if room is louder than default.
        if self.seen_frames < self.warmup_frames:
            self.seen_frames += 1
            self._ema = self._alpha * self._ema + (1 - self._alpha) * self._rms(frame)
            self._floor = max(self._floor, self._ema * 1.1)
            return None

        is_speech = self._is_speech(frame)

        if is_speech:
            if not self.speech_active:
                self.speech_active = True
                # start utterance with pre-roll for smoother onsets
                self.buffer.extend(self.prebuf)
            self.buffer.append(frame)
            self.trailing_silence = 0
            return None

        # not speech
        if self.speech_active and self.buffer:
            self.trailing_silence += 1
            if self.trailing_silence >= self.silence_tail_frames:
                utter = b"".join(self.buffer)
                enough = len(self.buffer) >= self.min_utter_frames
                self.reset()
                return utter if enough else None
        return None

    def flush(self) -> Optional[bytes]:
        if not self.buffer:
            return None
        utter = b"".join(self.buffer)
        ok = len(self.buffer) >= self.min_utter_frames
        self.reset()
        return utter if ok else None
