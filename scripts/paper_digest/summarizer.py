from __future__ import annotations

import re

from scripts.paper_digest.arxiv_client import Paper

_CONTRIBUTION_PATTERNS = [
    r"(?:we\s+)?propose\s+(.{20,220}?)(?:\.|$)",
    r"(?:we\s+)?present\s+(.{20,220}?)(?:\.|$)",
    r"(?:we\s+)?introduce\s+(.{20,220}?)(?:\.|$)",
    r"(?:our\s+)?(?:main\s+)?contributions?\s+(?:are|include)\s+(.{20,280}?)(?:\.|$)",
]


def _first_sentences(text: str, count: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    selected = [s.strip() for s in sentences if s.strip()][:count]
    return " ".join(selected)


def _extract_contribution(abstract: str) -> str | None:
    lowered = abstract.lower()
    for pattern in _CONTRIBUTION_PATTERNS:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            snippet = match.group(1).strip()
            snippet = snippet[0].upper() + snippet[1:] if snippet else snippet
            return snippet.rstrip(".") + "."
    return None


def summarize_paper(paper: Paper) -> str:
    """Build a concise Chinese-friendly core introduction from the abstract."""
    contribution = _extract_contribution(paper.abstract)
    intro = contribution or _first_sentences(paper.abstract, 2)

    metrics_match = re.search(
        r"(sdr|si-sdr|pesq|stoi)[^\n.]{0,80}",
        paper.abstract,
        flags=re.IGNORECASE,
    )
    metric_note = ""
    if metrics_match:
        metric_note = f" 实验指标提及：{metrics_match.group(0).strip()}。"

    stems_match = re.search(
        r"\b(vocals?|accompaniment|instrumental|karaoke|singing voice|music source separation)\b[^.]{0,60}",
        paper.abstract,
        flags=re.IGNORECASE,
    )
    focus_note = ""
    if stems_match:
        focus_note = f" 关注方向：{stems_match.group(0).strip()}。"

    return f"{intro}{focus_note}{metric_note}".strip()
