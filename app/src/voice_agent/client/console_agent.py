from __future__ import annotations

import atexit
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import gettempdir
from typing import List, Dict, Optional

# ---------- Optional Windows hotkey support ----------
try:
    import msvcrt  # Windows consoles for F9 toggle
except ImportError:
    msvcrt = None

# ---- LLM backend (your file is at src/voice_agent/llm_hf_local.py) ----
try:
    from voice_agent.llm_hf_local import complete as hf_complete, preload as hf_preload
except Exception:
    from voice_agent.llm_hf_local import complete as hf_complete  # type: ignore
    hf_preload = None  # type: ignore

# ---- TTS (persistent) ----
from voice_agent.client.tts_manager import TTSManager

# ---- Analyze code skill ----
from voice_agent.skills.analyze_code import handle_analyze_snippet


BANNER = """\
Console Agent
-------------
F9: Toggle Type-to-Talk mode
Type mode commands:
  /analyze   Paste a code snippet (end with a line: EOF) to get a report
  /quit      Exit

Push-to-talk (ASR) still runs in the background. Press Ctrl+C to exit.
"""


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


class SingleInstance:
    """Prevent multiple console agents via a temp-file lock."""
    def __init__(self, name: str = "voice_agent_console.lock"):
        self.lock_path = Path(gettempdir()) / name
        self.fd: Optional[int] = None
        try:
            self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(self.fd, str(os.getpid()).encode("utf-8"))
        except FileExistsError:
            print("[error] Another console_agent instance is already running.", file=sys.stderr)
            raise SystemExit(1)
        atexit.register(self.release)

    def release(self):
        try:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except Exception:
                    pass
            if self.lock_path.exists():
                try:
                    self.lock_path.unlink()
                except Exception:
                    pass
        except Exception:
            pass


# -------- Markdown → prose (strip all code + links) --------
_INLINE_CODE_RX = re.compile(r"`[^`\n]+`")
_URL_RX = re.compile(r"\bhttps?://\S+")

def _strip_fenced_blocks(md: str) -> str:
    """
    Remove fenced code blocks marked by ``` or ~~~ (with or without a language tag),
    even if the closing fence is at end-of-string without a trailing newline.
    """
    text = md
    fence_rx = re.compile(r"(```|~~~)")
    while True:
        m1 = fence_rx.search(text)
        if not m1:
            break
        fence = m1.group(1)
        start = m1.start()
        m2 = re.compile(re.escape(fence)).search(text, m1.end())
        if not m2:
            # no closing fence -> drop everything from start
            text = text[:start]
            break
        end = m2.end()
        text = text[:start] + text[end:]
    return text

def _strip_indented_blocks(md: str) -> str:
    lines = md.splitlines()
    out: List[str] = []
    in_block = False
    for ln in lines:
        if ln.startswith(("    ", "\t")):
            in_block = True
            continue
        if in_block and (ln.strip() == "" or ln.startswith(("    ", "\t"))):
            continue
        in_block = False
        out.append(ln)
    return "\n".join(out)

def prose_from_markdown(md: str) -> str:
    """
    Return only readable prose:
      - remove fenced blocks (``` / ~~~)
      - remove indented code blocks
      - remove inline code spans
      - remove URLs
    """
    text = _strip_fenced_blocks(md)
    text = _strip_indented_blocks(text)
    text = _INLINE_CODE_RX.sub("", text)
    text = _URL_RX.sub("", text)
    # tidy whitespace
    text = re.sub(r"[ \t]+(\n)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class ConsoleAgent:
    def __init__(self) -> None:
        # ---- Chat / LLM ----
        def _load_system_prompt() -> str:
            sp_file = os.getenv("AGENT_SYSTEM_FILE")
            if sp_file:
                try:
                    with open(sp_file, "r", encoding="utf-8") as f:
                        txt = f.read().strip()
                    if txt:
                        return txt
                except Exception as e:
                    print(f"[warn] Could not read AGENT_SYSTEM_FILE ({sp_file}): {e}")
            sp_inline = os.getenv("AGENT_SYSTEM")
            if sp_inline and sp_inline.strip():
                return sp_inline.strip()
            return (
                "You are a pragmatic coding assistant for a push-to-talk voice agent.\n\n"
                "Formatting:\n"
                "- Always answer in Markdown: short paragraphs/lists first, then code in fenced blocks.\n"
                "- Put all code inside triple backticks with a language tag (```python, ```bash).\n"
                "- Keep prose concise (1–3 sentences per paragraph) and end clearly so TTS can finish.\n\n"
                "Behavior:\n"
                "- Provide complete, minimal working snippets or unified diffs when code is requested.\n"
                "- Mention OS/shell assumptions briefly. If unknown, say how to verify."
            )

        self.system_prompt = _load_system_prompt()
        self.max_history = env_int("AGENT_MAX_TURNS", 8)
        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

        # ---- TTS ----
        tts_rate = env_int("TTS_RATE", 190)
        tts_voice = os.getenv("TTS_VOICE_ID") or None
        tts_backend = os.getenv("TTS_BACKEND", "sapi")
        self.tts = TTSManager(rate=tts_rate, volume=1.0, voice_id=tts_voice, backend=tts_backend)
        print(f"[tts] selected backend: {tts_backend}", flush=True)
        atexit.register(self.tts.shutdown)

        # ---- Runtime ----
        self.user_q: "queue.Queue[str]" = queue.Queue()
        self.asr_proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._asr_started = False
        self._printed_ready = False

        # ASR auto-restart
        self._asr_restarts = 0
        self._asr_restarts_max = 3

        # Accept '> USER: ...' OR bare '> ...' (but not our own LLM prints)
        self._pat_user = re.compile(r"^\s*>\s*USER:\s*(.+?)\s*$")
        self._pat_bare = re.compile(r"^\s*>\s*(?!LLM\s+response:)(.+?)\s*$")

        # Duck ASR while TTS is speaking (avoid echo)
        self._duck_asr = os.getenv("AGENT_IGNORE_ASR_WHILE_TTS", "1").lower() in ("1", "true", "yes")
        self._duck_cooldown_ms = env_int("AGENT_TTS_COOLDOWN_MS", 300)

        # ---- New: Type-to-Talk mode & hotkey queue ----
        self.type_mode: bool = False
        self._hotkey_q: "queue.Queue[str]" = queue.Queue()

    # ---------- Paths & ASR ----------
    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _src_dir(self) -> Path:
        # VOICE_AGENT/src
        return self._project_root() / "src"

    def _asr_cmd(self) -> List[str]:
        """Build ASR command (PTT-only) and launch by MODULE path: voice_agent.asr.asr_app"""
        dev = os.getenv("VA_ASR_DEVICE", "5")
        fw_model = os.getenv("VA_ASR_MODEL", "small")
        dev_type = os.getenv("VA_ASR_DEVICE_TYPE", "cpu")
        compute = os.getenv("VA_ASR_COMPUTE_TYPE", "int8")
        lang = os.getenv("VA_ASR_LANG", "en")
        gain_db = os.getenv("VA_ASR_GAIN_DB", "12")
        frame_ms = os.getenv("VA_ASR_FRAME_MS", "20")
        beam_size = os.getenv("VA_ASR_BEAM_SIZE", "5")

        return [
            sys.executable, "-u", "-m", "voice_agent.asr.asr_app",
            "--device", str(dev),
            "--gain-db", str(gain_db),
            "--device-type", dev_type,
            "--compute-type", compute,
            "--model", fw_model,
            "--lang", lang,
            "--beam-size", str(beam_size),
            "--frame-ms", str(frame_ms),
        ]

    def start_asr(self) -> None:
        """Launch ASR once (guarded)."""
        if self._asr_started:
            return
        self._asr_started = True

        cmd = self._asr_cmd()
        print(f"[info] launching ASR: {' '.join(cmd)}", flush=True)

        env = os.environ.copy()
        src_dir = str(self._src_dir())
        env["PYTHONPATH"] = os.pathsep.join([src_dir, env.get("PYTHONPATH", "")])

        self.asr_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self._project_root()),
            env=env,
        )
        threading.Thread(target=self._asr_reader_loop, daemon=True).start()

    def _asr_reader_loop(self) -> None:
        assert self.asr_proc and self.asr_proc.stdout
        last_ignored_log = 0.0
        try:
            for raw in self.asr_proc.stdout:
                if raw is None:
                    break
                s = raw.rstrip("\r\n")
                if s:
                    print(s, flush=True)

                if self._duck_asr:
                    if self.tts.is_speaking() or (time.time() - self.tts.last_end_time()) * 1000.0 < self._duck_cooldown_ms:
                        if time.time() - last_ignored_log > 1.5:
                            print("[asr] input ignored while TTS speaking/cooldown", flush=True)
                            last_ignored_log = time.time()
                        continue

                m = self._pat_user.match(s) or self._pat_bare.match(s)
                if m:
                    text = m.group(1).strip()
                    if text:
                        self.user_q.put_nowait(text)

                if self._stop.is_set():
                    break
        except Exception as e:
            print(f"[asr] reader error: {e}", file=sys.stderr)
        finally:
            rc = None
            try:
                rc = self.asr_proc.returncode
            except Exception:
                pass
            print(f"[info] ASR process exited (rc={rc}).", flush=True)

            if not self._stop.is_set() and self._asr_restarts < self._asr_restarts_max:
                self._asr_restarts += 1
                backoff = min(1 + self._asr_restarts, 5)
                print(f"[info] restarting ASR in {backoff}s (attempt {self._asr_restarts}/{self._asr_restarts_max})…", flush=True)
                self._asr_started = False
                time.sleep(backoff)
                self.start_asr()
            else:
                time.sleep(0.2)
                self.stop()

    # ---------- LLM ----------
    def _respond(self, user_text: str) -> None:
        """Common handler for both typed and ASR-sourced text."""
        try:
            self.history.append({"role": "user", "content": user_text})
            if len(self.history) > (2 * self.max_history + 1):
                self.history = [self.history[0]] + self.history[-2*self.max_history:]

            reply = hf_complete(self.history) or "(no response)"
            self.history.append({"role": "assistant", "content": reply})

            # Print full Markdown to console
            print(f"> LLM response: {reply}", flush=True)

            # Send prose-only to TTS (strip code + links)
            tts_text = prose_from_markdown(reply)
            if tts_text:
                self.tts.speak(tts_text)
        except Exception as e:
            print(f"[llm] error: {e}", file=sys.stderr)

    def _llm_worker(self) -> None:
        """Background worker for ASR-sourced text."""
        while not self._stop.is_set():
            try:
                user_text = self.user_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._respond(user_text)
            finally:
                self.user_q.task_done()

    # ---------- New: Hotkey watcher & type-mode ----------
    def _hotkey_watcher(self) -> None:
        """Watch F9 and push events into a queue (Windows consoles)."""
        if not msvcrt:
            return
        while not self._stop.is_set():
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ('\x00', '\xe0'):
                        code = msvcrt.getwch()
                        # On many Windows consoles, F9 emits '\x00' then '\x43'
                        if code in ('\x43',):
                            self._hotkey_q.put("F9")
                time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def _toggle_type_mode(self) -> None:
        self.type_mode = not self.type_mode
        state = "ON" if self.type_mode else "OFF"
        print(f"\n[Mode] Type-to-Talk is {state}.", flush=True)
        if self.type_mode:
            print("Tip: Type `/analyze` to paste a code snippet (finish with a line `EOF`).", flush=True)
            # If you want F9 to immediately launch Analyze:
            # return self._run_analyze_flow()

    def _run_analyze_flow(self) -> None:
        """Prompt user to paste code, analyze it, speak TTS-safe summary."""
        print("\nPaste your code below. End input with a single line: EOF", flush=True)
        pasted_lines: List[str] = []
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            if line.strip() == "EOF":
                break
            pasted_lines.append(line)
        pasted = "".join(pasted_lines)

        result = handle_analyze_snippet(
            pasted,
            filename=None,
            goals=["review"],
            mode="quick",
            llm=None  # plug your local LLM callable here if desired
        )

        print("\n=== Analysis Report (Markdown) ===", flush=True)
        print(result["markdown"], flush=True)
        if self.tts:
            print("\n[TTS] Speaking summary (code elided)…", flush=True)
            self.tts.speak(result["tts_text"])
        print("=== End Report ===\n", flush=True)

    # ---------- Main loop ----------
    def run(self) -> None:
        _ = SingleInstance()

        if hf_preload:
            print("[llm] preloading model…", flush=True)
            try:
                hf_preload()
                print("[llm] ready.", flush=True)
            except Exception as e:
                print(f"[llm] preload warning: {e}", file=sys.stderr)

        try:
            signal.signal(signal.SIGINT, lambda *_: self.stop())
        except Exception:
            pass

        print(BANNER, flush=True)

        # Start ASR + LLM worker
        self.start_asr()
        threading.Thread(target=self._llm_worker, daemon=True).start()

        # Start hotkey watcher (non-blocking)
        if msvcrt:
            threading.Thread(target=self._hotkey_watcher, daemon=True).start()
        else:
            print("[hint] F9 hotkey requires Windows console (msvcrt).", flush=True)

        if not self._printed_ready:
            print("[info] Agent ready. Hold your push-to-talk key in the ASR app. Press F9 to type.", flush=True)
            self._printed_ready = True

        try:
            while not self._stop.is_set():
                # Handle hotkeys
                try:
                    hk = self._hotkey_q.get_nowait()
                except queue.Empty:
                    hk = None
                if hk == "F9":
                    self._toggle_type_mode()

                # Typed interaction loop when type mode is ON
                if self.type_mode:
                    try:
                        user = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n[Interrupted input]", flush=True)
                        self.type_mode = False
                        continue

                    if not user:
                        continue
                    if user.lower() in ("/quit", "quit", "exit"):
                        print("Bye.", flush=True)
                        self.stop()
                        break
                    if user.lower().startswith("/analyze") or user.lower().startswith(":analyze"):
                        self._run_analyze_flow()
                        continue

                    # Normal typed user message
                    print(f"[You] {user}", flush=True)
                    self._respond(user)

                # Allow background threads to run
                time.sleep(0.05)
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()

        try:
            while not self.user_q.empty():
                self.user_q.get_nowait()
                self.user_q.task_done()
        except Exception:
            pass

        try:
            if self.asr_proc and self.asr_proc.poll() is None:
                self.asr_proc.terminate()
                try:
                    self.asr_proc.wait(timeout=2.0)
                except Exception:
                    self.asr_proc.kill()
        except Exception:
            pass

        try:
            self.tts.shutdown()
        except Exception:
            pass

        print("[info] Agent stopped.", flush=True)


def main() -> None:
    agent = ConsoleAgent()
    agent.run()


if __name__ == "__main__":
    main()
