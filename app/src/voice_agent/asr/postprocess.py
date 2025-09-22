import re

_PUNCT_MAP = {
    " ,": ",",
    " .": ".",
    " !": "!",
    " ?": "?",
    " :": ":",
    " ;": ";",
}

_FILLERS = re.compile(r"\b(um+|uh+|erm+|ah+)\b", re.IGNORECASE)

def clean_text(text: str, normalize_punct: bool = True, strip_fillers: bool = False) -> str:
    s = text.strip()
    if strip_fillers:
        s = _FILTER_FILLERS(s)
    if normalize_punct:
        for k, v in _PUNCT_MAP.items():
            s = s.replace(k, v)
        s = re.sub(r"\s+([,.!?;:])", r"\1", s)
        s = re.sub(r"\s{2,}", " ", s)
    return s

def _FILTER_FILLERS(s: str) -> str:
    return _FILLERS.sub("", s)
