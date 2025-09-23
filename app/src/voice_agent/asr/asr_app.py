import argparse, json, math, queue, time, uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pynput import keyboard
from rich.console import Console

from .capture import MicStream, SAMPLE_RATE
from .vad import VADSegmenter
from .backend_whisper import WhisperBackend, Segment
from .postprocess import clean_text
from .sinks import ConsoleSink, ClipboardSink, JSONLSink, CompositeSink, Transcript

console = Console()

# ---------- Event Bus ----------

ASR_PARTIAL = "ASR_PARTIAL"      
ASR_ENDPOINTED = "ASR_ENDPOINTED" 
ASR_FINAL = "ASR_FINAL"            
BARGE_IN = "BARGE_IN"            
ASR_STATUS = "ASR_STATUS"          
ASR_DEGRADED = "ASR_DEGRADED"     

class EventBus:
    def __init__(self):
        self._subs: List[Callable[[str, dict], None]] = []
    def subscribe(self, fn: Callable[[str, dict], None]): self._subs.append(fn)
    def emit(self, kind: str, payload: Dict):
        for fn in list(self._subs):
            try: fn(kind, payload)
            except Exception: pass

# ---------- Internal state ----------

@dataclass
class AppState:
    running: bool = True
    listening: bool = True       # F9 toggle on/off
    ptt_active: bool = False     # F10 push-to-talk (hold)
    ptt_buf: List[bytes] = None
    last_speech_active: bool = False
    copy_clip: bool = True

    def __post_init__(self):
        if self.ptt_buf is None: self.ptt_buf = []

# ---------- Utilities ----------

def build_sinks(jsonl_path: Optional[str], copy_clip: bool) -> CompositeSink:
    sinks = [ConsoleSink(printer=lambda s: console.print(f"[bold]{s}[/bold]"))]
    if copy_clip: sinks.append(ClipboardSink())
    if jsonl_path: sinks.append(JSONLSink(jsonl_path))
    return CompositeSink(sinks)

def log_event(kind: str, payload: dict, path: Optional[str]) -> None:
    if not path: return
    rec = {"ts": time.time(), "kind": kind, **payload}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------- Library entrypoint ----------

class ASRApp:
    def __init__(self, args):
        self.args = args
        self.state = AppState(copy_clip=(not args.no_clip))
        self.bus = EventBus()
        self.sinks = build_sinks(args.jsonl, self.state.copy_clip)

        self.backend = WhisperBackend(
            model_name=args.model,
            device=args.device_type,
            compute_type=args.compute_type,
            language=args.lang,
            beam_size=args.beam_size,
            vad_filter=True,
        )

        if self.backend.degraded_info is not None:
            dev, comp = self.backend.degraded_info
            self.bus.emit(ASR_DEGRADED, {"device": dev, "compute": comp})

        self.vad = VADSegmenter(
            frame_ms=args.frame_ms,
            silence_tail_ms=args.silence_tail_ms,
            sample_rate=SAMPLE_RATE,
            rms_floor=args.rms_floor,
            thr_mult=args.thr_mult,
            warmup_ms=args.warmup_ms,
            pre_roll_ms=args.pre_roll_ms,
            min_utter_ms=args.min_utter_ms,
        )  # :contentReference[oaicite:6]{index=6}

        self.preamp_gain = 10.0 ** (args.gain_db / 20.0)
        self.q_utter: queue.Queue[bytes] = queue.Queue()
        self.current_uid: Optional[str] = None
        self.t0_ms: Optional[int] = None

        self.mic = MicStream(
            frame_ms=args.frame_ms,
            device=args.device,
            on_status=lambda m: self.bus.emit(ASR_STATUS, {"level": "audio", "msg": m}),
            preamp_gain=self.preamp_gain,
        )  # :contentReference[oaicite:7]{index=7}

    # ---- public API
    def subscribe(self, fn: Callable[[str, dict], None]): self.bus.subscribe(fn)

    def start(self):
        self._start_hotkeys()
        self.mic.start(self._on_frame)
        self._print_banner()
        self._loop()

    def stop(self):
        try: self.mic.stop()
        except Exception: pass
        try: self.listener.stop()
        except Exception: pass
        console.print("[red]ASR stopped.[/red]")

    # ---- internals
    def _print_gain(self):
        db = 20.0 * math.log10(self.mic.preamp_gain) if self.mic.preamp_gain > 0 else -120.0
        console.print(f"[yellow]Preamp:[/yellow] {db:.1f} dB  (x{self.mic.preamp_gain:.2f})")

    def _on_frame(self, pcm: bytes):
        if self.state.listening:
            # Peek into VAD decision by running push twice:
            utter = self.vad.push(pcm)  
            speech_now = self.vad.speech_active
            if speech_now and not self.state.last_speech_active:
                self.current_uid = self.current_uid or str(uuid.uuid4())
                self.t0_ms = int(time.time() * 1000)
                self.bus.emit(BARGE_IN, {"uid": self.current_uid}) 
            self.state.last_speech_active = speech_now

            if utter:
                # End of utterance
                uid = self.current_uid or str(uuid.uuid4())
                self.bus.emit(ASR_ENDPOINTED, {"uid": uid})
                log_event("asr_start", {"uid": uid, "frames": len(utter)}, self.args.jsonl)
                self.q_utter.put((uid, self.t0_ms or int(time.time() * 1000), utter))
                # reset tracking for next turn
                self.current_uid, self.t0_ms = None, None
        else:
            # not listening, but let user flush current buffer on toggle
            pass

        # Push-to-talk capture
        if self.state.ptt_active:
            self.state.ptt_buf.append(pcm)

    def _start_hotkeys(self):
        def on_press(key):
            if key == keyboard.Key.f10:
                self.state.ptt_active = True

        def on_release(key):
            if key == keyboard.Key.f10:
                self.state.ptt_active = False
                if self.state.ptt_buf:
                    uid = str(uuid.uuid4())
                    self.bus.emit(ASR_ENDPOINTED, {"uid": uid})
                    blob = b"".join(self.state.ptt_buf)
                    self.state.ptt_buf.clear()
                    self.q_utter.put((uid, int(time.time()*1000), blob))
            elif key == keyboard.Key.f9:
                self.state.listening = not self.state.listening
                console.print(f"[cyan]Listening:[/cyan] {self.state.listening}")
                if not self.state.listening:
                    blob = self.vad.flush()
                    if blob:
                        uid = str(uuid.uuid4())
                        self.bus.emit(ASR_ENDPOINTED, {"uid": uid})
                        self.q_utter.put((uid, int(time.time()*1000), blob))
            elif key == keyboard.Key.f5:  # gain up (1.25x)
                self.mic.set_preamp_gain(self.mic.preamp_gain * 1.25)
                self._print_gain()
            elif key == keyboard.Key.f4:  # gain down (÷1.25)
                self.mic.set_preamp_gain(self.mic.preamp_gain / 1.25)
                self._print_gain()
            elif key in (keyboard.Key.f8, keyboard.Key.esc):
                self.state.running = False
                return False

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

    def _print_banner(self):
        console.print("[bold cyan]ASR running[/bold cyan] — F9: toggle listen, F10: hold-to-talk, F4/F5: gain, F8/Esc: quit.")
        if self.state.copy_clip: console.print("Clipboard: [green]ON[/green]")
        if self.args.jsonl: console.print(f"Logging JSONL → {self.args.jsonl}")
        if self.args.gain_db: self._print_gain()

    def _loop(self):
        sinks = self.sinks
        try:
            while self.state.running:
                try:
                    uid, t0_ms, utter = self.q_utter.get(timeout=0.2)
                except queue.Empty:
                    continue

                t_decode0 = time.time()
                text, lang, segs = self.backend.transcribe_with_segments(utter)  # segments + lang  :contentReference[oaicite:9]{index=9}
                t_decode_ms = int((time.time() - t_decode0) * 1000)

                # Final cleaning; partial cleaning would be lighter
                text_clean = clean_text(text, normalize_punct=True, strip_fillers=False)  # :contentReference[oaicite:10]{index=10}

                tN_ms = int(time.time() * 1000)
                payload = {
                    "uid": uid,
                    "text": text_clean,
                    "lang": lang,
                    "segments": [s.__dict__ for s in segs],
                    "t0_ms": t0_ms,
                    "tN_ms": tN_ms,
                    "dur_ms": max(0, tN_ms - t0_ms),
                    "decode_ms": t_decode_ms,
                }
                log_event("asr_done", payload, self.args.jsonl)
                self.bus.emit(ASR_FINAL, payload)

                if text_clean:
                    sinks.write(Transcript(text=text_clean, start_ts=t0_ms/1000.0, end_ts=tN_ms/1000.0))
        except KeyboardInterrupt:
            self.state.running = False

# ---------- CLI ----------

def run_asr(**overrides):
    # Convenience for library callers
    ns = argparse.Namespace(**overrides)
    app = ASRApp(ns)
    return app

def _parse_args():
    p = argparse.ArgumentParser(description="Always-on ASR (faster-whisper + energy VAD)")

    # faster-whisper / backend
    p.add_argument("--model", default="small.en", help="Whisper model (e.g., small.en, base.en)")
    p.add_argument("--device-type", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type", default="auto", help="e.g., int8 on CPU, float16 on GPU")
    p.add_argument("--lang", default="en", help="Language hint (e.g., en)")
    p.add_argument("--beam-size", type=int, default=5)

    # VAD / endpointing
    p.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    p.add_argument("--silence-tail-ms", type=int, default=400)
    p.add_argument("--rms-floor", type=float, default=60.0)
    p.add_argument("--thr-mult", type=float, default=1.6)
    p.add_argument("--warmup-ms", type=int, default=400)
    p.add_argument("--pre-roll-ms", type=int, default=240)
    p.add_argument("--min-utter-ms", type=int, default=300)

    # I/O and UX
    p.add_argument("--device", type=int, default=None, help="Input device index")
    p.add_argument("--jsonl", default=None, help="Path to append transcripts as JSONL")
    p.add_argument("--no-clip", action="store_true", help="Disable copying transcript to clipboard")
    p.add_argument("--gain-db", type=float, default=0.0, help="Input preamp in dB (e.g., 6 = ~2x)")

    return p.parse_args()

def main():
    args = _parse_args()
    app = ASRApp(args)
    app.subscribe(lambda k, p: None)  # no-op; LLM/TTS will subscribe later
    app.start()
    app.stop()

if __name__ == "__main__":
    main()
