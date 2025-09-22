import sounddevice as sd
import numpy as np
from typing import Optional, Callable

SAMPLE_RATE = 16000
CHANNELS = 1

class MicStream:
    """
    Mic capture at 16k/mono. Calls `on_frame(bytes)` every block.
    Supports a software preamp (linear gain) with clipping protection.
    """
    def __init__(
        self,
        frame_ms: int = 30,
        device: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
        preamp_gain: float = 1.0,           
    ):
        assert frame_ms in (10, 20, 30), "frame_ms must be 10/20/30"
        self.frame_samples = SAMPLE_RATE * frame_ms // 1000
        self._on_frame = None
        self._on_status = on_status
        self.device = device
        self.preamp_gain = float(preamp_gain)
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
            callback=self._callback,
        )

    def set_preamp_gain(self, gain: float):
        # clamp to a sensible range: 0.05x .. 20x
        self.preamp_gain = float(max(0.05, min(gain, 20.0)))

    def _callback(self, indata, frames, time_info, status):
        if status and self._on_status:
            self._on_status(str(status))
        if self._on_frame is None:
            return

        block = indata[: self.frame_samples]

        if self.preamp_gain != 1.0:
            arr = np.asarray(block, dtype=np.int32)
            arr = (arr * self.preamp_gain).clip(-32768, 32767).astype(np.int16)
            pcm = arr.tobytes()
        else:
            pcm = block.tobytes()

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
