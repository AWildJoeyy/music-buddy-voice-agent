import argparse, json, math, queue, time, uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pynput import keyboard
from rich.console import Console

from .capture import MicStream, SAMPLE_RATE
from .backend_whisper import WhisperBackend, Segment
from .postprocess import clean_text
from .sinks import ConsoleSink, ClipboardSink, JSONLSink, CompositeSink, Transcript

console = Console()

ASR_ENDPOINTED = "ASR_ENDPOINTED"
ASR_FINAL = "ASR_FINAL"
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

@dataclass
class AppState:
    running: bool = True
    listening: bool = False  #
    ptt_active: bool = False  # F10 push-to-talk (hold)
    ptt_buf: List[bytes] = None
    copy_clip: bool = True

    def __post_init__(self):
        if self.ptt_buf is None:
            self.ptt_buf = []

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

        self.preamp_gain = 10.0 ** (args.gain_db / 20.0)
        self.q_utter: "queue.Queue[tuple[str,int,bytes]]" = queue.Queue()

        self.mic = MicStream(
            frame_ms=args.frame_ms,
            device=args.device,
            on_status=lambda m: self.bus.emit(ASR_STATUS, {"level": "audio", "msg": m}),
            preamp_gain=self.preamp_gain,
        )

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

    def _print_gain(self):
        db = 20.0 * math.log10(self.mic.preamp_gain) if self.mic.preamp_gain > 0 else -120.0
        console.print(f"[yellow]Preamp:[/yellow] {db:.1f} dB  (x{self.mic.preamp_gain:.2f})")

    def _on_frame(self, pcm: bytes):
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
                # No-op in PTT mode (kept for compatibility / hint)
                self.state.listening = False
                console.print("[cyan]PTT-only mode — hold F10 to talk.[/cyan]")
            elif key == keyboard.Key.f5:  # gain up
                self.mic.set_preamp_gain(self.mic.preamp_gain * 1.25)
                self._print_gain()
            elif key == keyboard.Key.f4:  # gain down
                self.mic.set_preamp_gain(self.mic.preamp_gain / 1.25)
                self._print_gain()
            elif key in (keyboard.Key.f8, keyboard.Key.esc):
                self.state.running = False
                return False

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

    def _print_banner(self):
        console.print("[bold cyan]ASR (PTT-only) running[/bold cyan] — Hold [bold]F10[/bold] to talk. F4/F5: gain, F8/Esc: quit.")
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
                text, lang, segs = self.backend.transcribe_with_segments(utter)
                t_decode_ms = int((time.time() - t_decode0) * 1000)

                text_clean = clean_text(text, normalize_punct=True, strip_fillers=False)

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

def run_asr(**overrides):
    ns = argparse.Namespace(**overrides)
    app = ASRApp(ns)
    return app

def _parse_args():
    p = argparse.ArgumentParser(description="ASR (PTT-only)")
    # backend
    p.add_argument("--model", default="small.en", help="Whisper model (e.g., small.en, base.en)")
    p.add_argument("--device-type", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type", default="auto", help="e.g., int8 on CPU, float16 on GPU")
    p.add_argument("--lang", default="en", help="Language hint (e.g., en)")
    p.add_argument("--beam-size", type=int, default=5)
    # capture / UX
    p.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    p.add_argument("--device", type=int, default=None, help="Input device index")
    p.add_argument("--jsonl", default=None, help="Path to append transcripts as JSONL")
    p.add_argument("--no-clip", action="store_true", help="Disable copying transcript to clipboard")
    p.add_argument("--gain-db", type=float, default=0.0, help="Input preamp in dB (e.g., 6 = ~2x)")
    return p.parse_args()

def main():
    args = _parse_args()
    app = ASRApp(args)
    app.subscribe(lambda k, p: None)  # LLM/TTS will subscribe later
    app.start()
    app.stop()

if __name__ == "__main__":
    main()
