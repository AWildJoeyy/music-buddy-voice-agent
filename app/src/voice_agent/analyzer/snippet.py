from __future__ import annotations
import re
from typing import Callable, List, Dict, Any, Optional

# Optional LLM callable signature: (system, prompt) -> str
LLMFn = Callable[[str, str], str]

LANG_HINTS = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "java": "java", "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "c": "c",
    "cs": "csharp", "rb": "ruby", "php": "php", "go": "go",
    "rs": "rust", "swift": "swift", "kt": "kotlin", "m": "objectivec"
}

def _guess_language(code: str, filename: Optional[str]) -> str:
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        return LANG_HINTS.get(ext, "")
    m = re.search(r"^```(\w+)", code, flags=re.M)
    if m:
        lang = m.group(1).lower()
        return LANG_HINTS.get(lang, lang)
    if re.search(r"^\s*def\s+\w+\(", code, re.M): return "python"
    if re.search(r"^\s*(#include|using\s+namespace)", code, re.M): return "cpp"
    if re.search(r"^\s*function\s+|=>|console\.log", code): return "javascript"
    if re.search(r"^\s*class\s+\w+\s*\{", code) and ";" in code: return "java"
    return "text"

def _split_chunks(code: str, max_chars: int = 6000) -> List[str]:
    if len(code) <= max_chars:
        return [code]
    parts, chunk, size = [], [], 0
    for line in code.splitlines(True):
        if size + len(line) > max_chars and chunk:
            parts.append("".join(chunk))
            chunk, size = [], 0
        chunk.append(line); size += len(line)
    if chunk:
        parts.append("".join(chunk))
    return parts

_SECRET_PATTERNS = [
    r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]",
    r"(?i)aws[_-]secret",
    r"(?i)authorization:\s*Bearer\s+[A-Za-z0-9\._\-]+",
    r"(?i)password\s*[:=]\s*['\"].+?['\"]",
]

def _scan_secrets(code: str) -> List[str]:
    hits = []
    for pat in _SECRET_PATTERNS:
        if re.search(pat, code):
            hits.append(f"Potential secret matched: {pat}")
    return hits

def _static_checks(code: str, language: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    lines = code.splitlines()
    for i, ln in enumerate(lines, 1):
        if "TODO" in ln or "FIXME" in ln:
            issues.append({
                "severity": "low",
                "title": "TODO/FIXME present",
                "detail": ln.strip(),
                "lines": [i]
            })
        if language == "python":
            if re.match(r"\s*print\(", ln) and "if __name__" not in ln:
                issues.append({
                    "severity": "low",
                    "title": "Debug print()",
                    "detail": "Prefer logging over print().",
                    "lines": [i]
                })
        if language in ("javascript", "typescript"):
            if "console.log(" in ln:
                issues.append({
                    "severity": "low",
                    "title": "console.log in code",
                    "detail": "Prefer a logger / remove debug logs.",
                    "lines": [i]
                })
    long_lines = [i for i, ln in enumerate(lines, 1) if len(ln) > 120]
    if long_lines:
        issues.append({
            "severity": "info",
            "title": "Very long lines",
            "detail": "Lines exceed 120 chars; consider wrapping.",
            "lines": long_lines[:20]
        })
    return issues

SYSTEM_PROMPT = "You are a precise, no-nonsense code reviewer. Be concise and actionable. Cite line numbers."
TASK_TEMPLATE = (
    "Analyze the following code snippet.\n"
    "Return JSON with keys:\n"
    "language, detected_frameworks, summary, issues(severity,title,detail,lines), complexity, suggested_tests, security_notes, refactor_patch (unified diff), followups.\n\n"
    "Goals: {goals}\n"
    "Snippet ({filename}):\n"
    "```\n"
    "{code}\n"
    "```\n"
)

def analyze_snippet(
    code: str,
    filename: Optional[str] = None,
    goals: Optional[List[str]] = None,
    mode: str = "quick",
    llm: Optional[LLMFn] = None,
) -> Dict[str, Any]:
    """
    Works offline (static checks) if llm is None.
    If llm is provided, merges LLM insights chunk-by-chunk.
    """
    code = code.strip()
    language = _guess_language(code, filename)
    chunks = _split_chunks(code, 6000)
    secrets = _scan_secrets(code)
    issues = _static_checks(code, language)

    base_summary = f"{language or 'code'} with {len(code.splitlines())} lines, {len(chunks)} chunk(s)."
    security_notes: List[str] = []
    if secrets:
        security_notes.append("Potential secrets found (redact before sharing): " + "; ".join(secrets))

    detected_frameworks: List[str] = []
    if language == "python" and ("fastapi" in code or "from fastapi" in code):
        detected_frameworks.append("fastapi")
    if language in ("javascript", "typescript") and "react" in code.lower():
        detected_frameworks.append("react")

    refactor_patch = ""
    suggested_tests: List[str] = []
    complexity = {
        "function_counts": {
            "def_like": len(re.findall(r"\bdef\b|\bfunction\b|=>", code))
        },
        "hotspots": []
    }

    # Optional LLM enrichment
    if llm:
        llm_summaries: List[str] = []
        llm_issues: List[Dict[str, Any]] = []
        llm_tests: List[str] = []
        llm_sec: List[str] = []
        llm_patch = ""

        for c in chunks:
            prompt = TASK_TEMPLATE.format(
                goals=", ".join(goals or []),
                filename=filename or "(pasted)",
                code=c[:6000]
            )
            out = llm(SYSTEM_PROMPT, prompt)
            try:
                import json
                j = json.loads(out)
            except Exception:
                j = {"summary": out[:800], "issues": [], "suggested_tests": [], "security_notes": [], "refactor_patch": ""}

            if j.get("summary"): llm_summaries.append(j["summary"].strip())
            llm_issues.extend(j.get("issues", []))
            llm_tests.extend(j.get("suggested_tests", []))
            llm_sec.extend(j.get("security_notes", []))
            if not llm_patch and j.get("refactor_patch"): llm_patch = j["refactor_patch"]

        if llm_summaries: base_summary += " " + " ".join(s for s in llm_summaries if s)
        issues.extend(llm_issues)
        suggested_tests.extend(llm_tests)
        security_notes.extend(llm_sec)
        if llm_patch: refactor_patch = llm_patch

    # normalize issues
    def _norm(i: Dict[str, Any]) -> Dict[str, Any]:
        i["severity"] = (i.get("severity") or "info").lower()
        i["lines"] = sorted(set(map(int, (i.get("lines") or []))))
        return i
    issues = [_norm(i) for i in issues]

    return {
        "language": language or "text",
        "detected_frameworks": sorted(set(detected_frameworks)),
        "summary": base_summary.strip(),
        "issues": issues,
        "complexity": complexity,
        "suggested_tests": suggested_tests[:20],
        "security_notes": security_notes[:10],
        "refactor_patch": refactor_patch,
        "followups": [
            "Do you want me to generate unit tests?",
            "Should I create a refactor patch file?",
            "Limit code reading to public, non-secret sections?",
        ],
    }
