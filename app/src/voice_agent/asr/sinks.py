from dataclasses import dataclass
from typing import Optional, List, Iterable
from datetime import datetime
import json

@dataclass
class Transcript:
    text: str
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

class Sink:
    def write(self, rec: Transcript) -> None: ...

class ConsoleSink(Sink):
    def __init__(self, printer=print):
        self.printer = printer
    def write(self, rec: Transcript) -> None:
        self.printer(f"> {rec.text}")

class ClipboardSink(Sink):
    def __init__(self):
        try:
            import pyperclip
        except Exception:  # pragma: no cover
            pyperclip = None
        self._pc = pyperclip
    def write(self, rec: Transcript) -> None:
        if self._pc:
            try:
                self._pc.copy(rec.text)
            except Exception:
                pass

class JSONLSink(Sink):
    def __init__(self, path: str):
        self.path = path
    def write(self, rec: Transcript) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            obj = {
                "text": rec.text,
                "start_ts": rec.start_ts,
                "end_ts": rec.end_ts,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

class CompositeSink(Sink):
    def __init__(self, sinks: Iterable[Sink]):
        self.sinks: List[Sink] = list(sinks)
    def write(self, rec: Transcript) -> None:
        for s in self.sinks:
            s.write(rec)
