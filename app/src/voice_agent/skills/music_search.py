from __future__ import annotations
from typing import Optional, List, Dict, Any
from app.src.db.dal_music import get_tracks, create_playlist, all_tracks

def search_tracks(bpm_min: Optional[float]=None, bpm_max: Optional[float]=None,
                  energy_min: Optional[float]=None, energy_max: Optional[float]=None,
                  include_tags: Optional[List[str]]=None, exclude_tags: Optional[List[str]]=None,
                  limit: int=25) -> List[Dict[str,Any]]:
    rows = get_tracks(bpm_min, bpm_max, energy_min, energy_max, include_tags, exclude_tags, limit)
    return [dict(r) for r in rows]

def make_playlist(name: str, track_ids: List[str]) -> str:
    return create_playlist(name, track_ids)

def list_all() -> List[Dict[str,Any]]:
    return [dict(r) for r in all_tracks()]
