import re
from typing import Optional, List
from ..analyzer.snippet import analyze_snippet, LLMFn
from ..analyzer.render import render_markdown_report, speak_only_text

FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.S)

def extract_code(text: str) -> str:
    """Prefer fenced code blocks; fallback to 'mostly code' heuristic."""
    blocks = FENCE_RE.findall(text)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    lines = text.splitlines()
    if not lines:
        return ""
    codeish = sum(bool(re.match(r"\s{2,}|\t|.*[{}();<>]", ln)) for ln in lines)
    return text if codeish / max(1, len(lines)) > 0.6 else ""

def handle_analyze_snippet(
    pasted: str,
    filename: Optional[str] = None,
    goals: Optional[List[str]] = None,
    mode: str = "quick",
    llm: Optional[LLMFn] = None,
) -> dict:
    """
    Returns:
      - report: JSON dict
      - markdown: formatted report for chat
      - tts_text: code-free summary text for TTS
    """
    code = extract_code(pasted) or pasted.strip()
    report = analyze_snippet(code=code, filename=filename, goals=goals, mode=mode, llm=llm)
    md = render_markdown_report(report)
    tts_text = speak_only_text(md)
    return {"report": report, "markdown": md, "tts_text": tts_text}
