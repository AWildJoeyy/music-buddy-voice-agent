import re
from typing import Dict, Any, List

def render_markdown_report(rep: Dict[str, Any]) -> str:
    def bullets(items: List[str]) -> str:
        return "\n".join(f"- {x}" for x in items) if items else "- (none)"

    md = []
    md.append(f"# Code Analysis ({rep.get('language','text')})")
    if rep.get("detected_frameworks"):
        md.append(f"*Frameworks:* {', '.join(rep['detected_frameworks'])}")
    md.append("")
    md.append("## Summary")
    md.append(rep.get("summary","(no summary)"))
    md.append("")
    md.append("## Issues")
    issues = rep.get("issues", [])
    if not issues:
        md.append("- No issues found.")
    else:
        for i in issues[:50]:
            sev = i.get("severity","info").upper()
            title = i.get("title","")
            lines = i.get("lines",[])
            detail = i.get("detail","")
            line_str = f" [lines {', '.join(map(str, lines))}]" if lines else ""
            md.append(f"- **{sev}** {title}{line_str}: {detail}")
    md.append("")
    md.append("## Suggested tests")
    md.append(bullets(rep.get("suggested_tests", [])))
    md.append("")
    if rep.get("security_notes"):
        md.append("## Security notes")
        md.append(bullets(rep["security_notes"]))
        md.append("")
    if rep.get("refactor_patch"):
        md.append("## Refactor patch")
        patch = rep["refactor_patch"].strip()
        if not patch.startswith("```"):
            patch = "```diff\n" + patch + "\n```"
        md.append(patch)
        md.append("")
    md.append("## Follow-ups")
    md.append(bullets(rep.get("followups", [])))
    return "\n".join(md)

def speak_only_text(md: str) -> str:
    md = re.sub(r"```.*?```", "", md, flags=re.S)  # strip fenced code
    md = re.sub(r"`[^`]+`", "", md)                # strip inline code
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
