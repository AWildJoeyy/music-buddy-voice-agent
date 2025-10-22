from __future__ import annotations
import base64, json, os, sys, threading, time, tempfile, subprocess, sqlite3, random, shutil, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import builtins
def _jslog(s):
    try:
        sys.stdout.write(json.dumps({"type":"stderr","data":str(s)})+"\n"); sys.stdout.flush()
    except: ...
_old_print = builtins.print
def _p(*a, **k): _jslog(" ".join(map(str, a)))
builtins.print = _p

def _send(o: Dict[str, Any]): sys.stdout.write(json.dumps(o, ensure_ascii=False)+"\n"); sys.stdout.flush()
def _dbg(s: str): _send({"type":"stderr","data":f"[player-debug] {s}"})

_THIS = Path(__file__).resolve()
_VOICE_AGENT_DIR = _THIS.parent
_APP_SRC_DIR = _VOICE_AGENT_DIR.parent
for p in (str(_APP_SRC_DIR), str(_VOICE_AGENT_DIR)):
    if p not in sys.path: sys.path.insert(0, p)

PROJ_ROOT = _THIS.parents[3]; APP_ROOT = _THIS.parents[2]; SRC_ROOT = _THIS.parents[1]
def _first_exists(paths):
    for p in paths:
        if p and Path(p).exists(): return Path(p)
    return None

PLAYLISTS = Path(os.getenv("PLAYLISTS_DIR") or (_first_exists([PROJ_ROOT/"playlists", APP_ROOT/"playlists", SRC_ROOT/"playlists"]) or (PROJ_ROOT/"playlists")))
MUSIC_DIR = Path(os.getenv("MUSIC_DIR") or (PROJ_ROOT/"music"))
DB_PATH = Path(os.getenv("MUSIC_DB_PATH") or (PROJ_ROOT/"music.sqlite3"))
PLAYLISTS.mkdir(parents=True, exist_ok=True); MUSIC_DIR.mkdir(parents=True, exist_ok=True)

def _i(name, default):
    v=os.getenv(name)
    try: return int(v) if v is not None else default
    except: return default

SYS = "You are a compact, helpful music/chat assistant. Keep replies brief. Never print internal logs."
def _sys_prompt():
    f=os.getenv("AGENT_SYSTEM_FILE")
    if f:
        try: t=Path(f).read_text(encoding="utf-8").strip()
        except Exception as e: _send({"type":"stderr","data":f"system file error: {e}"}); t=""
        if t: return t
    t=os.getenv("AGENT_SYSTEM")
    return t.strip() if t and t.strip() else SYS

_LLM_NAME="none"
_llm_complete=None; _llm_preload=None
_LLM_SEM = threading.BoundedSemaphore(1)

def _import_llm():
    global _llm_complete,_llm_preload,_LLM_NAME
    if _llm_complete is not None: return
    try:
        from voice_agent.llm_hf_local import complete as C, preload as P
        _llm_complete,_llm_preload=C,P; _LLM_NAME="voice_agent.llm_hf_local"; _dbg(f"LLM={_LLM_NAME}"); return
    except Exception as e: _dbg(f"hf_local import failed: {e}")
    try:
        from app.src.voice_agent.llm_hf_local import complete as C, preload as P
        _llm_complete,_llm_preload=C,P; _LLM_NAME="app.src.voice_agent.llm_hf_local"; _dbg(f"LLM={_LLM_NAME}"); return
    except Exception as e: _dbg(f"alt import failed: {e}")

    import re
    def _fallback_complete(msgs):
        last = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                last = (m.get("content") or "").lower()
                break

        def pick_playlist(s: str) -> str:
            m = re.search(r"play (?:from (?:my|the) )?([\w\- ]+)\s+playlist", s)
            m2 = re.search(r"\bplay\s+([\w\- ]+)\b", s)
            raw = (m.group(1) if m else (m2.group(1) if m2 else "")).strip()
            return re.sub(r"^(my|the)\s+", "", raw, flags=re.I).strip()

        pl = pick_playlist(last)
        if pl: return f"Okay — play_playlist('{pl}')."
        if "pause" in last: return "Pausing now."
        if "resume" in last or "continue" in last: return "Resuming now."
        return "Got it. Say play from my <playlist> playlist (e.g., upbeat, mid, slow)."

class ChatCore:
    def __init__(self):
        _import_llm()
        self.system=_sys_prompt()
        self.hist=[{"role":"system","content":self.system}]
        self.max_turns=_i("AGENT_MAX_TURNS",8)
        try:_llm_preload()
        except Exception as e:_send({"type":"stderr","data":f"LLM preload: {e}"})
        self._stop=False
        self.delay=float(os.getenv("AGENT_TOKEN_DELAY","0.0"))
        self.timeout_s=float(os.getenv("AGENT_COMPLETE_TIMEOUT","8"))  # faster fail

    # rest of ChatCore unchanged (busy semaphore & streaming helpers as in your latest)


    def stop(self): self._stop=True

    def _tok(self,text:str):
        if self.delay<=0:  # fast path
            yield text+" "
            return
        for t in text.split(" "):
            if self._stop: break
            time.sleep(self.delay)
            yield t+" "

    def _complete_with_timeout(self, msgs: List[Dict[str,Any]]) -> str:
        got = _LLM_SEM.acquire(timeout=float(os.getenv("AGENT_BUSY_WAIT","0.05")))
        if not got:
            _dbg("LLM busy — fast fallback")
            return "I’m on it. Meanwhile you can say play_playlist('upbeat') or 'slow'."
        res={"text":None,"err":None}
        def work():
            try: res["text"]=_llm_complete(msgs)
            except Exception as e: res["err"]=e
            finally:
                try:_LLM_SEM.release()
                except: ...
        th=threading.Thread(target=work, daemon=True); th.start()
        th.join(self.timeout_s)
        if th.is_alive():
            _dbg("LLM timeout — using fallback string")
            try:_LLM_SEM.release()
            except: ...
            return "Sorry — my brain is slow right now. Say play_playlist('upbeat') to start some energy."
        if res["err"] is not None:
            _dbg(f"LLM error: {res['err']}")
            return "I hit an error, but I can still manage the player — try play_playlist('upbeat') or 'slow'."
        return res["text"] or "(empty)"

    def chat(self,cid:str,msgs:List[Dict[str,str]]):
        self._stop=False
        try:
            for m in msgs:
                if m["role"] in ("user","assistant"): self.hist.append(m)
            if len(self.hist)>(2*self.max_turns+1):
                self.hist=[self.hist[0]]+self.hist[-2*self.max_turns:]
            out=self._complete_with_timeout(self.hist)
        except Exception as e:
            out=f"(error) {e}"
        for t in self._tok(out): _send({"type":"chat_chunk","id":cid,"text":t})
        _send({"type":"chat_end","id":cid})

chat=ChatCore()

SCHEMA="""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tracks(
 id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL,
 title TEXT, artist TEXT, duration_s REAL, bpm REAL, energy REAL,
 key_tag TEXT, tags_json TEXT, emb BLOB);
"""
def _cx():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    cx=sqlite3.connect(str(DB_PATH)); cx.row_factory=sqlite3.Row; return cx
def _db_init():
    with _cx() as cx: cx.executescript(SCHEMA)

PLAYER_ERR=None
try:
    from voice_agent.skills.music_player import MusicPlayer, list_audio_files
except Exception as e1:
    try:
        from skills.music_player import MusicPlayer, list_audio_files
    except Exception as e2:
        try:
            from music_player import MusicPlayer, list_audio_files
        except Exception as e3:
            MusicPlayer=None; list_audio_files=None; PLAYER_ERR=f"{e1!r} | {e2!r} | {e3!r}"
            _dbg(f"all player imports failed: {PLAYER_ERR}")

def _load_audio_monoral(path: str, target_sr: int = 16000):
    import numpy as np
    try:
        import librosa
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype("float32", copy=False), sr
    except Exception: ...
    p = subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1","-ar",str(target_sr),"-f","s16le","-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode!=0: raise RuntimeError(p.stderr.decode("utf-8","ignore").strip() or "ffmpeg decode failed")
    y = np.frombuffer(p.stdout, dtype=np.int16).astype("float32")/32768.0
    return y, target_sr

_play_started_at = 0.0
_pause_started_at = 0.0
_accum_pause_ms = 0
_last_duration_ms = 0
_last_track: Optional[str] = None
status_thread=None; _stop_tick=False

def _probe_duration_ms(path: str) -> int:
    try:
        from mutagen import File as MutFile
        m = MutFile(path)
        if m is not None and hasattr(m, "info") and getattr(m.info, "length", None):
            return int(m.info.length * 1000)
    except Exception: ...
    try:
        p = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if p.returncode == 0:
            s = (p.stdout or "").strip()
            if s: return int(float(s) * 1000.0)
    except Exception: ...
    return 0

def _current_from_status(st: Dict[str,Any]) -> Optional[str]:
    c = st.get("current") or st.get("current_path") or st.get("track") or st.get("path")
    if not c and hasattr(PLAYER, "current"):
        try: c = PLAYER.current()
        except: ...
    return str(c) if c else None

def _start_status_thread():
    global status_thread,_stop_tick,_last_track,_last_duration_ms
    if status_thread and status_thread.is_alive(): return
    _stop_tick=False
    def run():
        global _last_track,_last_duration_ms
        while not _stop_tick:
            try:
                if not PLAYER or PLAYER is False: time.sleep(1); continue
                st = PLAYER.status() if hasattr(PLAYER,"status") else {}
                cur = _current_from_status(st) or _last_track
                if cur and cur != _last_track:
                    _last_track = cur
                    _last_duration_ms = 0
                    _send({"type":"music_now_playing","current":cur,"name":Path(cur).name})

                dur = st.get("duration_ms") or st.get("duration") or 0
                if not dur and cur:
                    if not _last_duration_ms:
                        _last_duration_ms = _probe_duration_ms(cur)
                    dur = _last_duration_ms
                    if dur: st["duration_ms"] = dur

                pos = st.get("position_ms") or st.get("position") or 0
                if not pos:
                    pos = max(0, int((time.time() - _play_started_at) * 1000) - _accum_pause_ms)
                    if dur: pos = min(pos, int(dur))
                    st["position_ms"] = pos

                if cur: st["current"] = cur
                st["tick"]=True
                _send({"type":"music_status","status":st})
            except Exception as e:
                _dbg(f"status tick error: {e!r}")
            time.sleep(1.0)
    status_thread=threading.Thread(target=run, daemon=True); status_thread.start()

AUDIO={".mp3",".wav",".flac",".m4a"}
def _is_audio(p:Path)->bool: return p.suffix.lower() in AUDIO
def _pl(name:str)->Path: p=PLAYLISTS/name; p.mkdir(parents=True,exist_ok=True); return p
def list_playlists()->List[str]: return sorted([p.name for p in PLAYLISTS.iterdir() if p.is_dir()])
def list_tracks(name:str)->List[str]:
    root=_pl(name); return [str(p.resolve()) for p in sorted(root.iterdir()) if p.is_file() and _is_audio(p)]
def import_to_playlist(name:str, paths:List[str], copy:bool=True)->Dict[str,Any]:
    dst=_pl(name); moved=[]
    for s in paths or []:
        sp=Path(s)
        if not(sp.exists() and sp.is_file() and _is_audio(sp)): continue
        d=dst/sp.name; i=1
        while d.exists(): d=dst/f"{sp.stem} ({i}){sp.suffix}"; i+=1
        (shutil.copy2 if copy else shutil.move)(sp,d); moved.append(str(d.resolve()))
    return {"playlist":name,"count":len(moved),"paths":moved}

def _analyze(path: str) -> Dict[str, Any]:
    bpm_tag=0.0
    try:
        from mutagen import File as MutFile
        m=MutFile(path)
        if m is not None and getattr(m,"tags",None):
            v=None
            for key in ("TBPM","bpm"):
                if key in m.tags: v=m.tags.get(key); break
            if v is not None:
                if hasattr(v,"text"): v=v.text
                if isinstance(v,(list,tuple)): v=v[0]
                try: bpm_tag=float(str(v))
                except: ...
    except Exception: ...
    import numpy as np, librosa
    y, sr = _load_audio_monoral(path, target_sr=16000)
    if y.size:
        peak=float(np.max(np.abs(y))); 
        if peak>0: y=y/peak
    duration_s=float(y.size)/float(sr) if sr else 0.0
    bpm_est=0.0
    try:
        tempo,_=librosa.beat.beat_track(y=y,sr=sr)
        bpm_est=float(tempo)
        if not bpm_est or not np.isfinite(bpm_est):
            bpm_est=float(librosa.beat.tempo(y=y,sr=sr).mean())
    except Exception: bpm_est=0.0
    bpm=float(bpm_tag or bpm_est or 0.0)
    energy=float(np.sqrt(np.mean(y*y))) if y.size else 0.0
    if bpm<100 and energy>0.20: bpm*=2.0
    return {"path":path,"duration_s":duration_s,"bpm":bpm,"energy":energy}

def _bucket(bpm: float, energy: float) -> str:
    if bpm>=150 or energy>=0.23: return "upbeat"
    if bpm<=95 or energy<=0.14: return "slow"
    score=(bpm/120.0)*0.4 + (energy/0.25)*0.6
    if score<0.9: return "slow"
    if score>1.1: return "upbeat"
    return "mid"

def auto_bucket(paths: List[str], copy: bool = True) -> Dict[str, Any]:
    analyzed,moved,errors=[],[],[]
    for p in paths or []:
        try:
            st=_analyze(p)
            b=_bucket(st["bpm"],st["energy"])
            res=import_to_playlist(b,[p],copy=copy)
            st["bucket"]=b; analyzed.append(st); moved+=res["paths"]
        except Exception as e:
            errors.append({"path":p,"error":str(e)}); _send({"type":"stderr","data":traceback.format_exc()})
    return {"analyzed":analyzed,"moved":moved,"errors":errors}

PLAYER=None
_last_play_req = {"name": None, "ts": 0.0}

def _player_init():
    global PLAYER
    _dbg("init called")
    if PLAYER is not None: 
        _dbg(f"init short-circuit: PLAYER={type(PLAYER)}"); 
        return
    if MusicPlayer is None or list_audio_files is None:
        _dbg(f"disabled: MusicPlayer or list_audio_files is None; import error={PLAYER_ERR}")
        PLAYER=False; return
    try:
        PLAYER=MusicPlayer(); _dbg(f"player constructed: {type(PLAYER)}")
    except Exception as e:
        _dbg(f"player init error: {e!r}"); PLAYER=False

def _player_play_playlist(name: str, shuffle: bool = False) -> Dict[str, Any]:
    global _last_play_req,_play_started_at,_accum_pause_ms,_last_duration_ms,_last_track
    now = time.time()
    if _last_play_req["name"] == name and (now - _last_play_req["ts"]) < 1.5:
        _dbg("debounce duplicate play request"); return {"status":"busy"}
    _last_play_req = {"name": name, "ts": now}

    _dbg(f"play_playlist requested: name={name!r} shuffle={shuffle}")
    _player_init()
    if not PLAYER: _dbg("unavailable: PLAYER is falsy"); return {"status":"unavailable"}
    try:
        pdir=_pl(name); _dbg(f"playlist dir resolved: {pdir} exists={pdir.exists()}")
        if list_audio_files is None: _dbg("list_audio_files is None"); return {"status":"error","error":"list_audio_files missing"}
        tracks=list_audio_files(Path(pdir)); _dbg(f"tracks found: {len(tracks)}; sample={tracks[:3]}")
        if not tracks: return {"status":"empty","playlist":name}
        if shuffle: random.shuffle(tracks); _dbg("tracks shuffled")
        PLAYER.load_list(tracks); _dbg("load_list OK")
        PLAYER.play(); _dbg("play() called")
        _play_started_at = time.time(); _accum_pause_ms = 0; _last_duration_ms = 0
        _last_track = str(tracks[0])
        _send({"type":"music_now_playing","current":_last_track,"name":Path(_last_track).name})
        st=PLAYER.status(); st.update({"status":"playing","playlist":name,"count":len(tracks),"current":_last_track})
        _dbg(f"status after play: {st}")
        _start_status_thread()
        return st
    except Exception as e:
        _dbg(f"play_playlist error: {e!r}"); return {"status":"error","error":str(e)}

def _seek_via_fraction(ms:int)->bool:
    try:
        st=PLAYER.status() if hasattr(PLAYER,"status") else {}
        dur=int(st.get("duration_ms") or st.get("duration") or 0)
        if dur>0:
            frac=max(0.0,min(1.0,float(ms)/float(dur)))
            if hasattr(PLAYER,"set_position"): PLAYER.set_position(frac); return True
            if hasattr(PLAYER,"player") and hasattr(PLAYER.player,"set_position"): PLAYER.player.set_position(frac); return True
    except Exception as e: _dbg(f"seek fraction error: {e!r}")
    return False

def _seek_best_effort(ms: int):
    try:
        if hasattr(PLAYER,"seek_ms"): PLAYER.seek_ms(ms); return "seek_ms"
        if hasattr(PLAYER,"seek"):
            try: PLAYER.seek(ms/1000.0)
            except TypeError: PLAYER.seek(int(ms/1000))
            return "seek"
        if hasattr(PLAYER,"set_time"): PLAYER.set_time(ms); return "set_time"
        if hasattr(PLAYER,"player") and hasattr(PLAYER.player,"set_time"):
            PLAYER.player.set_time(ms); return "player.set_time"
        if hasattr(PLAYER,"set_position_ms"): PLAYER.set_position_ms(ms); return "set_position_ms"
        if _seek_via_fraction(ms): return "set_position(frac)"
    except Exception as e: _dbg(f"seek backend error: {e!r}")
    return None

def _player_ctrl(cmd: str) -> Dict[str, Any]:
    global _pause_started_at,_accum_pause_ms,_play_started_at
    _dbg(f"ctrl: {cmd!r}")
    _player_init()
    if not PLAYER: _dbg("ctrl unavailable: PLAYER is falsy"); return {"status":"unavailable"}
    try:
        if cmd=="pause":
            PLAYER.pause(); _pause_started_at = time.time()
        elif cmd=="resume":
            PLAYER.resume()
            if _pause_started_at:
                _accum_pause_ms += int((time.time() - _pause_started_at) * 1000)
                _pause_started_at = 0.0
        elif cmd=="stop":
            global _stop_tick; _stop_tick=True
            PLAYER.stop()
        elif cmd=="next": PLAYER.next()
        elif cmd=="prev": PLAYER.prev()
        elif cmd.startswith("vol:"):
            try: v=int(cmd.split(":",1)[1]); _dbg(f"set_volume {v}"); PLAYER.set_volume(v)
            except Exception as ve: _dbg(f"volume parse/set error: {ve!r}")
        elif cmd.startswith("seek:"):
            try:
                ms=int(cmd.split(":",1)[1]); _dbg(f"seek to {ms} ms")
                method=_seek_best_effort(ms); _dbg(f"seek method used: {method or 'none'}")
                _play_started_at = time.time() - (ms / 1000.0)
            except Exception as se: _dbg(f"seek parse/set error: {se!r}")
        st=PLAYER.status() if hasattr(PLAYER,"status") else {}
        if _last_track: st["current"] = _last_track
        st["status"]="ok"
        _start_status_thread()
        return st
    except Exception as e:
        _dbg(f"ctrl error: {e!r}"); return {"status":"error","error":str(e)}

_tts=None; _tts_err=None
def _tts_init():
    global _tts,_tts_err
    if _tts or _tts_err: return
    try:
        from voice_agent.client.tts_manager import TTSManager
        _tts=TTSManager()
    except Exception:
        try:
            from app.src.voice_agent.client.tts_manager import TTSManager
            _tts=TTSManager()
        except Exception as e: _tts_err=f"TTS init failed: {e}"

def _tts_speak_async(text:str):
    def run():
        try:
            _tts_init()
            if not _tts: return
            fn=None
            for n in ("speak","synthesize","synth","tts"):
                f=getattr(_tts,n,None)
                if callable(f): fn=f; break
            if fn: fn(text)
        except Exception as e:
            _send({"type":"stderr","data":f"TTS error: {e}"})
    threading.Thread(target=run, daemon=True).start()

def tts_b64(text:str):
    # fire-and-forget to keep GIL free
    try: _tts_speak_async(text or "")
    except Exception as e: _send({"type":"stderr","data":f"TTS async error: {e}"})
    silent=(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    return "audio/wav", base64.b64encode(silent).decode("ascii")

_whisper=None; _asr_err=None; _asr_fp=None; _asr_tmp:Optional[Path]=None
def _asr_init():
    global _whisper,_asr_err
    if _whisper or _asr_err: return
    try:
        from voice_agent.asr.backend_whisper import WhisperBackend
        _whisper=WhisperBackend(
            model_name=os.getenv("VA_ASR_MODEL","small.en"),
            device=os.getenv("VA_ASR_DEVICE_TYPE","auto"),
            compute_type=os.getenv("VA_ASR_COMPUTE_TYPE","auto"),
            language=os.getenv("VA_ASR_LANG","en"),
            beam_size=_i("VA_ASR_BEAM_SIZE",5),
            vad_filter=os.getenv("VA_ASR_VAD","1").lower() in ("1","true","yes"),
        )
    except Exception as e: _asr_err=f"ASR init failed: {e}"
def asr_start(fmt="webm"):
    global _asr_fp,_asr_tmp
    _asr_init()
    if _asr_fp:
        try:_asr_fp.close()
        except: ...
    td=Path(tempfile.mkdtemp(prefix="va_asr_"))
    _asr_tmp=td/f"rec.{fmt}"
    _asr_fp=_asr_tmp.open("ab")
    _send({"type":"asr_ready"})
def asr_chunk(b64:str):
    global _asr_fp
    if _asr_fp: _asr_fp.write(base64.b64decode(b64))
def _pcm16(src:Path)->bytes:
    p=subprocess.run(["ffmpeg","-v","error","-i",str(src),"-ac","1","-ar","16000","-f","s16le","-"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode!=0: raise RuntimeError(p.stderr.decode("utf-8","ignore"))
    return p.stdout
def asr_end():
    global _asr_fp,_asr_tmp,_whisper,_asr_err
    if _asr_fp:
        try:_asr_fp.close()
        except: ...
        _asr_fp=None
    if not(_asr_tmp and _asr_tmp.exists()): _send({"type":"stderr","data":"ASR: no audio"}); return
    if _whisper is None: _asr_init()
    if _whisper is None: _send({"type":"stderr","data":_asr_err or "ASR unavailable"}); return
    try:
        pcm=_pcm16(_asr_tmp); 
        text=_whisper.transcribe(pcm) or ""
        _send({"type":"asr_final","text":text})
    except Exception as e: _send({"type":"stderr","data":f"ASR failed: {e}"})

AUDIO={".mp3",".wav",".flac",".m4a"}
def _pl(name:str)->Path: p=PLAYLISTS/name; p.mkdir(parents=True,exist_ok=True); return p

def _diag():
    return {
        "agent": True,
        "llm": _LLM_NAME,
        "music_dir": str(MUSIC_DIR),
        "playlists_dir": str(PLAYLISTS),
        "db": str(DB_PATH),
        "cwd": os.getcwd(),
        "py": sys.version.split()[0],
    }

import queue
def _stdin_reader(q: "queue.Queue[dict]"):
    buf=""
    while True:
        ch = sys.stdin.read(1)
        if not ch: break
        if ch == "\n":
            line = buf.strip(); buf = ""
            if line:
                try: q.put(json.loads(line))
                except Exception as e: _send({"type":"stderr","data":f"bad json: {e}"})
        else: buf += ch

def main():
    _db_init()
    q: "queue.Queue[Any]" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(q,), daemon=True).start()
    _send({"type":"ready"}); _send({"type":"diag","status":_diag()})
    while True:
        req=q.get()
        if not isinstance(req,dict): continue
        t=req.get("type")
        if t=="chat":
            cid=req.get("id","chat"); msgs=req.get("messages",[])
            threading.Thread(target=chat.chat, args=(cid,msgs), daemon=True).start()
        elif t=="stop":
            chat.stop(); _send({"type":"stopped","id":req.get("id")})
        elif t=="tts":
            mime,b64 = tts_b64(req.get("text",""))
            _send({"type":"tts_result","id":req.get("id"),"mime":mime,"audio_b64":b64})
        elif t=="asr_start": asr_start(req.get("fmt","webm"))
        elif t=="asr_chunk": asr_chunk(req.get("data_b64",""))
        elif t=="asr_end": threading.Thread(target=asr_end, daemon=True).start()
        elif t=="music_list_playlists": _send({"type":"music_playlists","items":list_playlists()})
        elif t=="music_list_tracks": _send({"type":"music_tracks","playlist":req.get("playlist"),"items":list_tracks(req.get("playlist",""))})
        elif t=="music_import_to_playlist": _send({"type":"music_imported","result":import_to_playlist(req.get("playlist","unsorted"), req.get("paths") or [], bool(req.get("copy",True)))})
        elif t=="music_auto_bucket": _send({"type":"music_bucketed","result":auto_bucket(req.get("paths") or [], bool(req.get("copy",True)))})
        elif t=="music_play_playlist":
            _dbg(f"IPC music_play_playlist payload={req}")
            res=_player_play_playlist(req.get("playlist",""), bool(req.get("shuffle",False)))
            _send({"type":"music_player","result":res})
        elif t=="music_ctrl":
            _dbg(f"IPC music_ctrl payload={req}")
            res=_player_ctrl(str(req.get("cmd","")))
            _send({"type":"music_status","status":res})
        elif t=="diag": _send({"type":"diag","status":_diag()})
        elif t=="quit": _send({"type":"bye"}); break

if __name__=="__main__":
    main()
