# asr/capture.py
import sounddevice as sd
from typing import Optional, Callable

SAMPLE_RATE = 16000
CHANNELS = 1

class MicStream:
    """
    Simple mic capture at 16k/mono. Calls `on_frame(bytes)` every block.
    """
    def __init__(self, frame_ms: int = 30, device: Optional[int] = None,
                 on_status: Optional[Callable[[str], None]] = None):
        assert frame_ms in (10, 20, 30), ""
        self.frame_samples = SAMPLE_RATE * frame_ms // 1000
        self._on_frame = None
        self._on_status = on_status
        self.device = device
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status and self._on_status:
            self._on_status(str(status))
        if self._on_frame:
            # ensure exact frame length for VAD
            pcm = indata[: self.frame_samples].tobytes()
            self._on_frame(pcm)

    def start(self, on_frame: Callable[[bytes], None]):
        self._on_frame = on_frame
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
