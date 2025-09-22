# src/asr/app.py
import argparse
import json
import queue
import time
from typing import Optional, List

from pynput import keyboard
from rich.console import Console

from .capture import MicStream, SAMPLE_RATE
from .vad import VADSegmenter
from .backend_whisper import WhisperBackend
from .postprocess import clean_text
from .sinks import ConsoleSink, ClipboardSink, JSONLSink, CompositeSink, Transcript

console = Console()


class AppState:
    def __init__(self, copy_clip: bool = True):
        self.running: bool = True
        self.listening: bool = True       # F9 toggle on/off
        self.ptt_active: bool = False     # F10 push-to-talk (hold)
        self.copy_clip: bool = copy_clip
        self.ptt_buf: List[bytes] = []    # frames collected while holding F10


def build_sinks(jsonl_path: Optional[str], copy_clip: bool) -> CompositeSink:
    sinks = [ConsoleSink(printer=lambda s: console.print(f"[bold]{s}[/bold]"))]
    if copy_clip:
        sinks.append(ClipboardSink())
    if jsonl_path:
        sinks.append(JSONLSink(jsonl_path))
    return CompositeSink(sinks)


def log_event(kind: str, payload: dict, path: Optional[str]) -> None:
    """Append a tiny JSON record for debugging / future Notion sync."""
    if not path:
        return
    rec = {"ts": time.time(), "kind": kind, **payload}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Always-on ASR (faster-whisper + energy VAD)")

    # faster-whisper / backend
    p.add_argument("--model", default="small.en", help="Whisper model (e.g., small.en, base.en)")
    p.add_argument("--device-type", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type", default="auto", help="e.g., int8 on CPU, float16 on GPU")
    p.add_argument("--lang", default="en", help="Language hint (e.g., en)")
    p.add_argument("--beam-size", type=int, default=5)

    # VAD / endpointing (pure Python, no native deps)
    p.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    p.add_argument("--silence-tail-ms", type=int, default=400)
    p.add_argument("--rms-floor", type=float, default=60.0)
    p.add_argument("--thr-mult", type=float, default=1.6)
    p.add_argument("--warmup-ms", type=int, default=400)
    p.add_argument("--pre-roll-ms", type=int, default=240)
    p.add_argument("--min-utter-ms", type=int, default=300)

    # I/O and UX
    p.add_argument("--device", type=int, default=None, help="Input device index (see `va devices`)")
    p.add_argument("--jsonl", default=None, help="Path to append transcripts as JSONL")
    p.add_argument("--no-clip", action="store_true", help="Disable copying transcript to clipboard")

    args = p.parse_args()

    state = AppState(copy_clip=(not args.no_clip))
    sinks = build_sinks(args.jsonl, state.copy_clip)

    # Backend (auto-fallback to CPU int8 inside WhisperBackend if CUDA/cuDNN is missing)
    backend = WhisperBackend(
        model_name=args.model,
        device=args.device_type,
        compute_type=args.compute_type,
        language=args.lang,
        beam_size=args.beam_size,
        vad_filter=True,
    )

    # Energy-based VAD (warmup + pre-roll to avoid chopping first words)
    vad = VADSegmenter(
        frame_ms=args.frame_ms,
        silence_tail_ms=args.silence_tail_ms,
        sample_rate=SAMPLE_RATE,
        rms_floor=args.rms_floor,
        thr_mult=args.thr_mult,
        warmup_ms=args.warmup_ms,
        pre_roll_ms=args.pre_roll_ms,
        min_utter_ms=args.min_utter_ms,
    )

    q_utter: queue.Queue[bytes] = queue.Queue()

    def on_status(msg: str):
        console.log(f"[yellow]Audio status[/yellow]: {msg}")

    def on_frame(pcm: bytes):
        # Collect frames during push-to-talk
        if state.ptt_active:
            state.ptt_buf.append(pcm)
            return

        # Normal continuous listening
        if state.listening:
            utter = vad.push(pcm)
            if utter:
                q_utter.put(utter)

    mic = MicStream(frame_ms=args.frame_ms, device=args.device, on_status=on_status)
    mic.start(on_frame)

    # Hotkeys
    def on_press(key):
        if key == keyboard.Key.f10:
            state.ptt_active = True

    def on_release(key):
        if key == keyboard.Key.f10:
            state.ptt_active = False
            if state.ptt_buf:
                q_utter.put(b"".join(state.ptt_buf))
                state.ptt_buf.clear()
        elif key == keyboard.Key.f9:
            state.listening = not state.listening
            console.print(f"[cyan]Listening:[/cyan] {state.listening}")
            if not state.listening:
                utter = vad.flush()
                if utter:
                    q_utter.put(utter)
        elif key in (keyboard.Key.f8, keyboard.Key.esc):
            state.running = False
            return False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    console.print("[bold cyan]ASR running[/bold cyan] — F9: toggle listen, F10: hold-to-talk, F8/Esc: quit.")
    if state.copy_clip:
        console.print("Clipboard: [green]ON[/green]")
    if args.jsonl:
        console.print(f"Logging JSONL → {args.jsonl}")

    try:
        while state.running:
            try:
                utter = q_utter.get(timeout=0.2)
            except queue.Empty:
                continue

            log_event("asr_start", {"frames": len(utter)}, args.jsonl)
            text = backend.transcribe(utter)
            log_event("asr_done", {"text": text}, args.jsonl)

            text = clean_text(text, normalize_punct=True, strip_fillers=False)
            if text:
                sinks.write(Transcript(text=text))
    except KeyboardInterrupt:
        state.running = False
    finally:
        mic.stop()
        listener.stop()
        console.print("[red]ASR stopped.[/red]")


if __name__ == "__main__":
    main()
