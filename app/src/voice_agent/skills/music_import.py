from __future__ import annotations
import os, shutil
from pathlib import Path
from typing import List, Dict, Any
from app.src.db.dal_music import init, upsert_track

LIB_DIR = Path(os.getenv("MUSIC_LIBRARY_DIR", Path(__file__).resolve().parents[3] / "library"))

def import_files(paths: List[str]) -> Dict[str, Any]:
    init()
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    imported = []
    for p in paths or []:
        src = Path(p)
        if not src.is_file(): continue
        if src.suffix.lower() not in {".mp3",".wav",".flac",".m4a"}: continue
        dst = LIB_DIR / src.name
        i = 1
        while dst.exists():
            dst = LIB_DIR / f"{src.stem} ({i}){src.suffix}"
            i += 1
        shutil.copy2(src, dst)
        upsert_track({"path": str(dst.resolve()), "title": dst.stem})
        imported.append(str(dst.resolve()))
    return {"count": len(imported), "paths": imported}
