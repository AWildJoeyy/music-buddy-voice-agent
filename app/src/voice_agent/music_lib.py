from __future__ import annotations
import shutil, os
from pathlib import Path
from typing import List
from .music_db import init, upsert_basic

LIB_ROOT = Path(os.getenv("MUSIC_LIBRARY_DIR", Path(__file__).resolve().parents[1] / "library"))

def _ensure_lib():
    LIB_ROOT.mkdir(parents=True, exist_ok=True)
    return LIB_ROOT

def import_files(paths: List[str]) -> int:
    init()
    dest_root = _ensure_lib()
    count = 0
    for p in paths:
        src = Path(p)
        if not src.exists() or not src.is_file(): continue
        if src.suffix.lower() not in (".mp3", ".wav", ".flac", ".m4a"): continue
        dest = dest_root / src.name
        # avoid overwrite by appending a counter
        i, base, ext = 1, src.stem, src.suffix
        while dest.exists():
            dest = dest_root / f"{base} ({i}){ext}"
            i += 1
        shutil.copy2(src, dest)
        upsert_basic(str(dest.resolve()), dest.stem)
        count += 1
    return count
