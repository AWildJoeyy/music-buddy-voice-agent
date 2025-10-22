from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# --- VLC locate (Windows) ---
def _ensure_vlc_loaded() -> Path | None:
    if sys.platform != "win32":
        return None
    candidates: List[Path] = []
    env = os.environ.get("VLC_PATH")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path(r"C:\Program Files\VideoLAN\VLC"),
        Path(r"C:\Program Files (x86)\VideoLAN\VLC"),
    ]
    for d in candidates:
        if d and (d / "libvlc.dll").exists():
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return d
    return None

_ensure_vlc_loaded()

import vlc  # noqa: E402

def list_audio_files(dir_path: Path) -> List[str]:
    exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
    out: List[str] = []
    if dir_path and dir_path.exists():
        for p in sorted(dir_path.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(str(p.resolve()))
    return out

class MusicPlayer:
    def __init__(self) -> None:
        self._instance = vlc.Instance()  # env PATH/dll dir prepared above
        self._mlp = self._instance.media_list_player_new()
        self._player = self._instance.media_player_new()
        self._mlp.set_media_player(self._player)
        self._media_list = None
        self._vol = 80
        self._player.audio_set_volume(self._vol)

    def load_list(self, paths: List[str]) -> None:
        ml = self._instance.media_list_new(paths)
        self._media_list = ml
        self._mlp.set_media_list(ml)

    def play(self) -> None:
        self._mlp.play()

    def pause(self) -> None:
        self._mlp.pause()

    def resume(self) -> None:
        self._mlp.play()

    def stop(self) -> None:
        self._mlp.stop()

    def next(self) -> None:
        self._mlp.next()

    def prev(self) -> None:
        self._mlp.previous()

    def set_volume(self, v: int) -> None:
        v = max(0, min(100, int(v)))
        self._vol = v
        self._player.audio_set_volume(v)

    def status(self) -> Dict[str, Any]:
        st = self._player.get_state()
        return {
            "vlc_state": str(st),
            "volume": self._player.audio_get_volume(),
        }
