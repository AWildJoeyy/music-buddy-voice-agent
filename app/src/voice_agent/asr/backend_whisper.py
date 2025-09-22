from typing import Optional
from faster_whisper import WhisperModel
import numpy as np


class WhisperBackend:
    """
    faster-whisper wrapper that accepts raw 16k/mono INT16 PCM bytes,
    converts to float32 waveform, and transcribes. Also supports file paths.
    """

    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "auto",          # "auto" | "cpu" | "cuda"
        compute_type: str = "auto",    # "int8" on CPU, "float16" on GPU, etc.
        language: Optional[str] = "en",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        # Try requested device; fall back to CPU int8 if CUDA/cuDNN is missing
        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception:
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    @staticmethod
    def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
        """INT16 mono @ 16k → float32 waveform in [-1, 1]."""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return x / 32768.0

    def transcribe(self, pcm: bytes) -> str:
        """
        Transcribe a single utterance (bytes of INT16 PCM @ 16k mono).
        Returns the concatenated text.
        """
        waveform = self._pcm16_to_float32(pcm)
        segments, _info = self.model.transcribe(
            waveform,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return "".join(seg.text for seg in segments).strip()

    def transcribe_file(self, path: str) -> str:
        """
        Transcribe an audio file given by path (wav/mp3/m4a/etc.).
        Uses faster-whisper's internal decoder.
        """
        segments, _info = self.model.transcribe(
            path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return "".join(seg.text for seg in segments).strip()
