from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ========== JSONL plumbing ==========
def _read_jsonl(stdin, out_q):
    buf = ""
    while True:
        ch = stdin.read(1)
        if not ch:
            break
        if ch == "\n":
            line = buf.strip()
            buf = ""
            if line:
                try:
                    out_q.put(json.loads(line))
                except Exception as e:
                    _send({"type": "stderr", "data": f"bad json line: {e}"})
        else:
            buf += ch

def _send(obj: Dict[str, Any]):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

# ========== LLM (console-agent style) ==========
_llm_complete = None
_llm_preload = None

def _import_llm():
    global _llm_complete, _llm_preload
    if _llm_complete is not None:
        return
    try:
        from voice_agent.llm_hf_local import complete as _cmp, preload as _pre  # type: ignore
        _llm_complete, _llm_preload = _cmp, _pre
    except Exception:
        from voice_agent.llm_hf_local import complete as _cmp  # type: ignore
        _llm_complete, _llm_preload = _cmp, None

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _load_system_prompt() -> str:
    sp_file = os.getenv("AGENT_SYSTEM_FILE")
    if sp_file:
        try:
            txt = Path(sp_file).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception as e:
            _send({"type":"stderr","data":f"AGENT_SYSTEM_FILE read error: {e}"})
    sp_inline = os.getenv("AGENT_SYSTEM")
    if sp_inline and sp_inline.strip():
        return sp_inline.strip()
    return (
        "You are a pragmatic coding assistant for a push-to-talk voice agent.\n\n"
        "Formatting:\n"
        "- Answer in Markdown; concise bullet points first; code in ``` blocks with language tags.\n"
        "- Keep paragraphs short so TTS reads well.\n"
    )

class ChatCore:
    def __init__(self):
        _import_llm()
        self.system_prompt = _load_system_prompt()
        self.max_history = _env_int("AGENT_MAX_TURNS", 8)
        self.history: List[Dict[str,str]] = [{"role":"system","content":self.system_prompt}]
        if _llm_preload:
            try: _llm_preload()
            except Exception as e: _send({"type":"stderr","data":f"LLM preload: {e}"})
        self._stop = False
        self.fallback_delay = 0.02

    def stop(self):
        self._stop = True

    def _yield_tokens(self, text: str) -> Iterable[str]:
        for tok in text.split(" "):
            if self._stop:
                break
            yield tok + " "
            time.sleep(self.fallback_delay)

    def chat(self, chat_id: str, messages: List[Dict[str,str]]):
        self._stop = False
        try:
            for m in messages:
                if m["role"] in ("user","assistant"):
                    self.history.append(m)
            if len(self.history) > (2*self.max_history + 1):
                self.history = [self.history[0]] + self.history[-2*self.max_history:]
            reply = _llm_complete(self.history) if _llm_complete else "(no llm)"
            if not reply:
                reply = "(empty response)"
        except Exception as e:
            reply = f"(error) {e}"
        for tok in self._yield_tokens(reply):
            _send({"type":"chat_chunk","id":chat_id,"text":tok})
        _send({"type":"chat_end","id":chat_id})

chat_core = ChatCore()

# ========== Analyzer ==========
_analyze_fn = None
_analyzer_err = None

def _import_analyzer():
    global _analyze_fn, _analyzer_err
    if _analyze_fn or _analyzer_err: return
    try:
        from voice_agent.skills.analyze_code import handle_analyze_snippet as _h  # type: ignore
        _analyze_fn = ("pasted", _h)
        return
    except Exception:
        pass
    try:
        from voice_agent.skills.analyze_code import analyze_code as _h  # type: ignore
        _analyze_fn = ("paths", _h)
        return
    except Exception:
        pass
    try:
        from voice_agent.skills.analyze_code import analyze_files as _h  # type: ignore
        _analyze_fn = ("paths", _h)
        return
    except Exception as e:
        _analyzer_err = f"analyzer import failed: {e}"

def _read_small_files(paths: List[str], limit: int = 60_000) -> str:
    out, total = [], 0
    for p in paths:
        P = Path(p)
        if P.is_dir():
            for f in P.rglob("*"):
                if f.is_file():
                    s = f.read_text(encoding="utf-8", errors="ignore")
                    if total + len(s) > limit: break
                    out.append(f"\n\n# FILE: {f}\n{s}"); total += len(s)
        elif P.is_file():
            s = P.read_text(encoding="utf-8", errors="ignore")
            if total + len(s) > limit: break
            out.append(f"\n\n# FILE: {P}\n{s}"); total += len(s)
    return "\n".join(out).strip()

def analyze_paths(paths: List[str]) -> str:
    _import_analyzer()
    if not paths: return "No files selected."
    if not _analyze_fn:
        return f"Analyzer not found. ({_analyzer_err or 'no entrypoint'})"
    mode, fn = _analyze_fn
    try:
        if mode == "paths":
            res = fn(paths)
            if isinstance(res, dict) and "markdown" in res: return res["markdown"]
            return str(res)
        pasted = _read_small_files(paths)
        res = fn(pasted, filename="selection", goals=["review"], mode="quick", llm=None)
        if isinstance(res, dict) and "markdown" in res: return res["markdown"]
        return str(res)
    except Exception as e:
        return f"Analyze failed: {e}"

# ========== TTS ==========
_tts = None
_tts_err = None

def _tts_init():
    global _tts, _tts_err
    if _tts or _tts_err: return
    try:
        # your TTSManager is under voice_agent/client/tts_manager.py
        from voice_agent.client.tts_manager import TTSManager  # type: ignore
        _tts = TTSManager()
    except Exception as e:
        _tts_err = f"TTS init failed: {e}"

def tts_to_b64(text: str):
    _tts_init()
    if not _tts:
        return None, None
    for name in ("speak", "synthesize", "synth", "tts"):
        fn = getattr(_tts, name, None)
        if callable(fn):
            break
    else:
        _send({"type":"stderr","data":"TTSManager has no speak/synthesize/synth/tts"})
        return None, None
    try:
        fn(text)  # plays through device
        # Return a tiny silent wav so the UI shows a valid audio element
        silent = (b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
                  b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
                  b"data\x00\x00\x00\x00")
        return "audio/wav", base64.b64encode(silent).decode("ascii")
    except Exception as e:
        _send({"type":"stderr","data":f"TTS error: {e}"})
        return None, None

# ========== ASR (push-to-talk) ==========
_whisper = None
_asr_err = None

def _asr_init():
    global _whisper, _asr_err
    if _whisper or _asr_err: return
    try:
        from voice_agent.asr.backend_whisper import WhisperBackend  # type: ignore
        model = os.getenv("VA_ASR_MODEL", "small.en")
        device = os.getenv("VA_ASR_DEVICE_TYPE", "auto")
        compute = os.getenv("VA_ASR_COMPUTE_TYPE", "auto")
        lang = os.getenv("VA_ASR_LANG", "en")
        beam = _env_int("VA_ASR_BEAM_SIZE", 5)
        vad = os.getenv("VA_ASR_VAD", "1").lower() in ("1","true","yes")
        _whisper = WhisperBackend(model_name=model, device=device, compute_type=compute, language=lang, beam_size=beam, vad_filter=vad)
    except Exception as e:
        _asr_err = f"ASR init failed: {e}"

_asr_fp = None
_asr_tmp: Optional[Path] = None

def asr_start(fmt: str = "webm"):
    global _asr_fp, _asr_tmp
    _asr_init()
    if _asr_fp:
        try: _asr_fp.close()
        except Exception: pass
    tmpdir = Path(tempfile.mkdtemp(prefix="va_asr_"))
    _asr_tmp = tmpdir / f"rec.{fmt}"
    _asr_fp = _asr_tmp.open("ab")
    _send({"type":"asr_ready"})

def asr_chunk(b64: str):
    global _asr_fp
    if not _asr_fp: return
    _asr_fp.write(base64.b64decode(b64))

def _ffmpeg_raw_s16le(src: Path) -> bytes:
    # Convert to raw PCM s16le, 16kHz, mono for WhisperBackend.transcribe(pcm)
    cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore"))
    return p.stdout

def asr_end():
    global _asr_fp, _asr_tmp, _whisper, _asr_err
    if _asr_fp:
        try: _asr_fp.close()
        except Exception: pass
        _asr_fp = None
    if _asr_tmp is None or not _asr_tmp.exists():
        _send({"type":"stderr","data":"ASR: no audio"})
        return
    if _whisper is None:
        _asr_init()
    if _whisper is None:
        _send({"type":"stderr","data": _asr_err or "ASR backend unavailable"})
        return
    try:
        pcm = _ffmpeg_raw_s16le(_asr_tmp)
        text = _whisper.transcribe(pcm)  # bytes -> text
        _send({"type":"asr_final","text": text or ""})
    except Exception as e:
        _send({"type":"stderr","data": f"ASR failed: {e}"})

# ========== Diagnostics ==========
def _diag():
    return {
        "agent": bool(_llm_complete),
        "analyzer": bool(_analyze_fn),
        "tts": bool(_tts),
        "asr": bool(_whisper),
        "cwd": os.getcwd(),
        "py": sys.version.split()[0],
    }

# ========== Dispatcher ==========
def main():
    import queue
    q: "queue.Queue[Any]" = queue.Queue()
    threading.Thread(target=_read_jsonl, args=(sys.stdin, q), daemon=True).start()

    _send({"type":"ready"})
    _send({"type":"diag","status":_diag()})

    while True:
        req = q.get()
        if not isinstance(req, dict):
            continue
        t = req.get("type")

        if t == "chat":
            cid = req.get("id","chat")
            msgs = req.get("messages",[])
            threading.Thread(target=chat_core.chat, args=(cid,msgs), daemon=True).start()

        elif t == "stop":
            chat_core.stop()
            _send({"type":"stopped","id":req.get("id")})

        elif t == "analyze":
            paths = req.get("paths",[])
            md = analyze_paths(paths)
            _send({"type":"analyze_result","id":req.get("id"),"summary":md})

        elif t == "tts":
            text = req.get("text","")
            mime, b64 = tts_to_b64(text)
            if not b64:
                _send({"type":"stderr","data": "TTS not available."})
            else:
                _send({"type":"tts_result","id":req.get("id"),"mime":mime,"audio_b64":b64})

        elif t == "asr_start":
            asr_start(req.get("fmt","webm"))
        elif t == "asr_chunk":
            asr_chunk(req.get("data_b64",""))
        elif t == "asr_end":
            threading.Thread(target=asr_end, daemon=True).start()

        elif t == "diag":
            _send({"type":"diag","status":_diag()})

        elif t == "quit":
            _send({"type":"bye"})
            break

if __name__ == "__main__":
    main()
