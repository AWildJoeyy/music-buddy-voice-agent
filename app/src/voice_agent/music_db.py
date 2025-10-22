from __future__ import annotations
import sqlite3, json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DB_PATH = Path(__file__).resolve().parents[1] / "music.sqlite3"

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT UNIQUE NOT NULL,
  title TEXT, artist TEXT,
  duration_s REAL, bpm REAL, energy REAL,
  tags_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_bpm ON tracks(bpm);
CREATE INDEX IF NOT EXISTS idx_tracks_energy ON tracks(energy);
"""

def _cx():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(DB_PATH)
    cx.row_factory = sqlite3.Row
    return cx

def init():
    with _cx() as cx:
        cx.executescript(SCHEMA)

def upsert_basic(path: str, title: str):
    with _cx() as cx:
        cx.execute("""INSERT INTO tracks(path,title) VALUES(?,?)
                      ON CONFLICT(path) DO UPDATE SET title=excluded.title""",
                   (path, title))
