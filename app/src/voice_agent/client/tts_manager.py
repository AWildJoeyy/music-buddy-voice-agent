from __future__ import annotations
import threading, queue, time, os, sys, traceback
from typing import Optional

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import pythoncom
    import win32com.client  
except Exception:
    pythoncom = None
    win32com = None  # type: ignore

class TTSManager:
    """
    Robust TTS queue with two backends:
      - 'sapi' (raw COM; recommended on Windows)
      - 'pyttsx3' (wraps SAPI; sometimes fragile under load)
    Auto-fallback on failures, engine auto-reinit, thread-safe.
    Exposes speaking Event so callers can duck ASR while TTS is active.
    """

    def __init__(self, rate: int = 190, volume: float = 1.0,
                 voice_id: Optional[str] = None, backend: Optional[str] = None):
        self._q: "queue.Queue[tuple[str, bool]]" = queue.Queue()
        self._stop = threading.Event()
        self._speaking_evt = threading.Event()
        self._last_end_ts = 0.0
        self._fail_count = 0

        self._voice_id = voice_id
        self._rate = int(rate)
        self._volume = float(volume)
        self._backend_pref = (backend or os.getenv("TTS_BACKEND", "sapi")).lower()
        self._backend = None  
        self._engine = None  

        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _init_pyttsx3(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 not installed")
        eng = pyttsx3.init(driverName="sapi5")
        eng.setProperty("rate", self._rate)
        eng.setProperty("volume", self._volume)
        if self._voice_id:
            try:
                eng.setProperty("voice", self._voice_id)
            except Exception:
                pass
        self._backend = "pyttsx3"
        self._engine = eng
        print("[tts] backend: pyttsx3", flush=True)

    def _init_sapi(self):
        if pythoncom is None or win32com is None:
            raise RuntimeError("pywin32 not installed")
        pythoncom.CoInitialize()
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        if self._voice_id:
            try:
                for v in voice.GetVoices():
                    if self._voice_id.lower() in v.GetDescription().lower():
                        voice.Voice = v
                        break
            except Exception:
                pass
        try:
            voice.Rate = max(-10, min(10, int((self._rate - 170) / 10)))
        except Exception:
            pass
        try:
            voice.Volume = int(self._volume * 100)
        except Exception:
            pass
        self._backend = "sapi"
        self._engine = voice
        print("[tts] backend: SAPI (COM)", flush=True)

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return
            order = [self._backend_pref] + ([b for b in ("sapi", "pyttsx3") if b != self._backend_pref])
            last_err = None
            for b in order:
                try:
                    if b == "sapi":
                        self._init_sapi()
                    else:
                        self._init_pyttsx3()
                    return
                except Exception as e:
                    last_err = e
            raise RuntimeError(f"TTS init failed: {last_err}")


    def speak(self, text: str, high_priority: bool = False):
        if not text:
            return
        try:
            item = (text, bool(high_priority))
            if high_priority:
                with self._q.mutex:
                    items = list(self._q.queue)
                    self._q.queue.clear()
                self._q.put_nowait(item)
                for it in items:
                    self._q.put_nowait(it)
            else:
                self._q.put_nowait(item)
        except Exception:
            pass

    def shutdown(self):
        self._stop.set()
        try:
            self._q.put_nowait(("", False))
        except Exception:
            pass

    def is_speaking(self) -> bool:
        return self._speaking_evt.is_set()

    def last_end_time(self) -> float:
        return self._last_end_ts

    def _speak_pyttsx3(self, text: str):
        self._engine.say(text)    
        self._engine.runAndWait() 

    def _speak_sapi(self, text: str):
        self._engine.Speak(text, 0)  
    def _worker(self):
        # Engine is created lazily here
        while not self._stop.is_set():
            try:
                text, _prio = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            if not text:
                self._q.task_done()
                continue

            try:
                self._ensure_engine()
                self._speaking_evt.set()
                t0 = time.time()
                if self._backend == "sapi":
                    self._speak_sapi(text)
                else:
                    self._speak_pyttsx3(text)
                # success
                self._fail_count = 0
            except Exception as e:
                self._fail_count += 1
                print(f"[tts] error ({self._backend}): {e}", file=sys.stderr)
                traceback.print_exc(limit=1)
                # Hard reset engine
                with self._lock:
                    self._engine = None
                if self._fail_count >= 2:
                    self._backend_pref = "sapi" if self._backend == "pyttsx3" else "pyttsx3"
                    print(f"[tts] switching backend to {self._backend_pref}", flush=True)
                    self._fail_count = 0
                try:
                    self._q.put_nowait((text, False))
                except Exception:
                    pass
            finally:
                self._speaking_evt.clear()
                self._last_end_ts = time.time()
                # small cooldown helps some drivers flush
                time.sleep(0.02)
                try:
                    self._q.task_done()
                except Exception:
                    pass
