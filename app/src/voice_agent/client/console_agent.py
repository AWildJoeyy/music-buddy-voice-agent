# src/voice_agent/client/console_agent.py
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

# ---- LLM backend (supports Transformers GPTQ + AutoGPTQ) ----
try:
    from voice_agent.llm_hf_local import complete as hf_complete, preload as hf_preload
except Exception:
    from voice_agent.llm_hf_local import complete as hf_complete
    hf_preload = None  # type: ignore

# ---- TTS (persistent) ----
from voice_agent.client.tts_manager import TTSManager


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


# ----------------------------- singleton lock ----------------------------- #

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


# ----------------------------- agent runtime ----------------------------- #

class ConsoleAgent:
    def __init__(self) -> None:
        # ---- Chat / LLM ----
        self.system_prompt = os.getenv("AGENT_SYSTEM", "You are a concise, helpful voice assistant.")
        self.max_history = env_int("AGENT_MAX_TURNS", 8)
        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

        # ---- TTS ----
        tts_rate = env_int("TTS_RATE", 190)
        tts_voice = os.getenv("TTS_VOICE_ID") or None
        tts_backend = os.getenv("TTS_BACKEND", "sapi")  # robust default on Windows
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

    # ------------------------- ASR management ------------------------- #

    def _asr_cmd(self) -> List[str]:
        dev = os.getenv("VA_ASR_DEVICE", "5")
        fw_model = os.getenv("VA_ASR_MODEL", "small")
        dev_type = os.getenv("VA_ASR_DEVICE_TYPE", "cpu")
        compute = os.getenv("VA_ASR_COMPUTE_TYPE", "int8")
        lang = os.getenv("VA_ASR_LANG", "en")

        gain_db = os.getenv("VA_ASR_GAIN_DB", "12")
        frame_ms = os.getenv("VA_ASR_FRAME_MS", "10")
        thr_mult = os.getenv("VA_ASR_THR_MULT", "0.8")
        rms_floor = os.getenv("VA_ASR_RMS_FLOOR", "20")
        warmup_ms = os.getenv("VA_ASR_WARMUP_MS", "200")
        pre_roll_ms = os.getenv("VA_ASR_PRE_ROLL_MS", "300")
        silence_tail_ms = os.getenv("VA_ASR_SILENCE_TAIL_MS", "300")
        min_utter_ms = os.getenv("VA_ASR_MIN_UTTER_MS", "100")
        beam_size = os.getenv("VA_ASR_BEAM_SIZE", "5")
        start_listen = os.getenv("VA_ASR_START_LISTEN", "")  # default PTT

        cmd = [
            "va", "asr",
            "--device", str(dev),
            "--gain-db", str(gain_db),
            "--device-type", dev_type,
            "--compute-type", compute,
            "--model", fw_model,
            "--lang", lang,
            "--beam-size", str(beam_size),
            "--frame-ms", str(frame_ms),
            "--thr-mult", str(thr_mult),
            "--rms-floor", str(rms_floor),
            "--warmup-ms", str(warmup_ms),
            "--pre-roll-ms", str(pre_roll_ms),
            "--silence-tail-ms", str(silence_tail_ms),
            "--min-utter-ms", str(min_utter_ms),
        ]
        if start_listen.strip().lower() in ("1", "true", "yes"):
            cmd.append("--start-listening")
        return cmd

    def start_asr(self) -> None:
        """Launch ASR once (guarded)."""
        if self._asr_started:
            return
        self._asr_started = True

        cmd = self._asr_cmd()
        print(f"[info] launching ASR: {' '.join(cmd)}", flush=True)
        self.asr_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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

                # Duck: ignore ASR while TTS is speaking or just finished
                if self._duck_asr:
                    if self.tts.is_speaking() or (time.time() - self.tts.last_end_time()) * 1000.0 < self._duck_cooldown_ms:
                        if time.time() - last_ignored_log > 1.5:
                            print("[asr] input ignored while TTS speaking/cooldown", flush=True)
                            last_ignored_log = time.time()
                        continue

                # Accept '> USER: ...' OR bare '> ...'
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

            # Try to restart ASR if agent still running
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

    # ---------------------------- LLM worker ---------------------------- #

    def _llm_worker(self) -> None:
        while not self._stop.is_set():
            try:
                user_text = self.user_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.history.append({"role": "user", "content": user_text})
                if len(self.history) > (2 * self.max_history + 1):
                    self.history = [self.history[0]] + self.history[-2*self.max_history:]

                reply = hf_complete(self.history)
                if not reply:
                    reply = "(no response)"
                self.history.append({"role": "assistant", "content": reply})

                print(f"> LLM response: {reply}", flush=True)
                self.tts.speak(reply)
            except Exception as e:
                print(f"[llm] error: {e}", file=sys.stderr)
            finally:
                self.user_q.task_done()

    # ---------------------------- Lifecycle ----------------------------- #

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

        self.start_asr()
        threading.Thread(target=self._llm_worker, daemon=True).start()

        if not self._printed_ready:
            print("[info] Agent ready. Speak when ASR shows 'Listening: True'. Press Ctrl+C to exit.", flush=True)
            self._printed_ready = True

        try:
            while not self._stop.is_set():
                time.sleep(0.1)
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


# ------------------------------- Entrypoint ------------------------------- #

def main() -> None:
    agent = ConsoleAgent()
    agent.run()


if __name__ == "__main__":
    main()
