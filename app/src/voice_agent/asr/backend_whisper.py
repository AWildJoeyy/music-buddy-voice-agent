from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from faster_whisper import WhisperModel

@dataclass
class Segment:
    start: float
    end: float
    text: str

class WhisperBackend:
    """
    faster-whisper wrapper that accepts raw 16k/mono INT16 PCM bytes,
    returns text + segments + lang. Falls back to CPU/int8 if needed.
    """
    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = "en",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.degraded_info: Optional[Tuple[str, str]] = None
        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception:
            # Degrade gracefully
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.degraded_info = ("cpu", "int8")

        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    @staticmethod
    def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return x / 32768.0

    # Backward-compatible text-only API
    def transcribe(self, pcm: bytes) -> str:
        text, _lang, _segs = self.transcribe_with_segments(pcm)
        return text

    # segments + language for richer UI/LLM context
    def transcribe_with_segments(self, pcm: bytes) -> Tuple[str, Optional[str], List[Segment]]:
        waveform = self._pcm16_to_float32(pcm)
        segments, info = self.model.transcribe(
            waveform,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        seg_list: List[Segment] = [Segment(s.start, s.end, s.text) for s in segments]
        text = "".join(s.text for s in seg_list).strip()
        lang = getattr(info, "language", None) if info is not None else None
        return text, lang, seg_list
