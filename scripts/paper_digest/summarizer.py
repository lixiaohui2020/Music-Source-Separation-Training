from __future__ import annotations

import re

from scripts.paper_digest.arxiv_client import Paper

_CONTRIBUTION_PATTERNS = [
    r"(?:we\s+)?propose\s+(.{20,220}?)(?:\.|$)",
    r"(?:we\s+)?present\s+(.{20,220}?)(?:\.|$)",
    r"(?:we\s+)?introduce\s+(.{20,220}?)(?:\.|$)",
    r"(?:our\s+)?(?:main\s+)?contributions?\s+(?:are|include)\s+(.{20,280}?)(?:\.|$)",
]

_TOPIC_RULES: list[tuple[str, list[str]]] = [
    ("降噪增强", [r"denois", r"noise suppression", r"speech enhancement", r"audio restoration"]),
    ("去回声", [r"echo cancell", r"dereverb", r"acoustic echo"]),
    ("人声分离", [r"source separation", r"vocal separation", r"stem separation", r"singing voice"]),
    ("语音转录", [r"speech recognition", r"automatic speech recognition", r"\basr\b", r"speech-to-text", r"transcription", r"whisper"]),
    ("语音交互", [r"spoken dialogue", r"voice interaction", r"spoken language", r"speech understanding"]),
    ("语音合成", [r"text-to-speech", r"speech synthesis", r"vocoder", r"voice cloning"]),
    ("音频生成", [r"audio generation", r"sound generation", r"music generation", r"audio lm"]),
    ("说话人", [r"speaker diarization", r"voice conversion", r"speaker verification", r"speaker recognition"]),
    ("音频大模型", [r"audio llm", r"audio language model", r"speech foundation", r"multimodal audio"]),
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


def detect_topics(paper: Paper, max_topics: int = 3) -> list[str]:
    text = f"{paper.title} {paper.abstract}".lower()
    matched: list[str] = []
    for label, patterns in _TOPIC_RULES:
        if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
            matched.append(label)
        if len(matched) >= max_topics:
            break
    return matched or ["AI 音频"]


def summarize_paper(paper: Paper) -> str:
    """Backward-compatible brief summary string."""
    from scripts.paper_digest.paper_analyzer import analyze_paper

    return analyze_paper(paper).brief_summary
