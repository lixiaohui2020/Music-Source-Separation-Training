from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from scripts.paper_digest.arxiv_client import Paper
from scripts.paper_digest.pdf_extractor import (
    ExtractedFigure,
    PdfExtract,
    download_arxiv_pdf,
    extract_from_pdf,
)
from scripts.paper_digest.summarizer import detect_topics

logger = logging.getLogger(__name__)


@dataclass
class PaperAnalysis:
    brief_summary: str
    core_idea: str
    architecture_figures: list[ExtractedFigure] = field(default_factory=list)
    result_figures: list[ExtractedFigure] = field(default_factory=list)
    result_tables: list[str] = field(default_factory=list)
    experiment_highlights: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    # kept for backward compatibility with old email renderer paths
    pipeline_steps: list[str] = field(default_factory=list)
    experiment_results: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _first_sentences(text: str, count: int = 2) -> str:
    return " ".join(_sentences(text)[:count])


def _extract_contribution_preserving_case(text: str) -> str | None:
    patterns = [
        r"((?:We|This paper|This work)\s+(?:propose|present|introduce|develop|propose a novel)[^.]{20,260}\.)",
        r"((?:Our (?:main )?contributions? (?:are|include)|The contributions? of this paper)[^.]{20,300}\.)",
        r"((?:In this (?:paper|work), we)[^.]{20,260}\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def _build_brief_summary(paper: Paper, pdf: PdfExtract, topics: list[str]) -> str:
    """核心介绍：优先用摘要前两句原文，不改写、不降格大小写。"""
    source = pdf.abstract_text or paper.abstract
    summary = _first_sentences(source, 2)
    if not summary:
        summary = paper.title
    topic_note = f"（领域：{' / '.join(topics)}）" if topics else ""
    return f"{summary}{topic_note}"


def _build_core_idea(paper: Paper, pdf: PdfExtract) -> str:
    """核心思路：问题背景 + 作者提出的方法贡献（来自摘要/引言原文，保留原文表述）。"""
    abstract = re.sub(r"\s+", " ", (pdf.abstract_text or paper.abstract)).strip()
    intro = re.sub(r"\s+", " ", pdf.intro_text).strip()
    method = re.sub(r"\s+", " ", pdf.method_text).strip()
    source = " ".join(part for part in [abstract, intro, method] if part)
    if not source:
        return "PDF 解析不足，请查看原文。"

    contribution = _extract_contribution_preserving_case(source)

    # Prefer contrastive sentences that look like research motivation
    problem = None
    for pattern in [
        r"((?:However|Nevertheless|Despite this|Despite these)[^.]{40,260}\.)",
        r"((?:Existing (?:methods|approaches|models|works)[^.]{30,240}\.))",
        r"((?:A (?:key|major|critical) challenge[^.]{20,220}\.))",
    ]:
        match = re.search(pattern, abstract + " " + intro, flags=re.IGNORECASE)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip()
            # Filter incomplete fragments
            if len(candidate) >= 45 and not candidate.lower().startswith("while "):
                problem = candidate
                break

    parts: list[str] = []
    if problem:
        parts.append(f"【动机/问题】{problem}")

    if contribution:
        parts.append(f"【方法贡献】{contribution}")
    else:
        for sentence in _sentences(abstract):
            if re.search(r"propose|present|introduce|develop|framework|benchmark", sentence, re.I):
                parts.append(f"【方法贡献】{sentence}")
                break
        if not any(p.startswith("【方法贡献】") for p in parts):
            parts.append(f"【方法贡献】{_first_sentences(abstract, 2)}")

    if method:
        method_snip = _first_sentences(method, 1)
        if method_snip and method_snip not in " ".join(parts):
            parts.append(f"【方法细节】{method_snip}")

    return "\n".join(parts)


def _build_experiment_highlights(pdf: PdfExtract, paper: Paper) -> list[str]:
    highlights: list[str] = []
    source = pdf.result_text or paper.abstract

    for sentence in _sentences(source):
        if re.search(
            r"\d+\.?\d*\s*(?:%|dB|db)|SDR|PESQ|STOI|WER|MOS|outperform|state[- ]of[- ]the[- ]art|ablation",
            sentence,
            re.I,
        ):
            highlights.append(sentence)
        if len(highlights) >= 5:
            break

    if pdf.result_tables:
        for table in pdf.result_tables[:2]:
            first_line = table.split("\n", 1)[0].strip()
            if first_line and first_line not in highlights:
                highlights.append(first_line)

    return highlights[:6]


def analyze_paper(paper: Paper, cache_dir: Path | None = None) -> PaperAnalysis:
    topics = detect_topics(paper)
    cache = cache_dir or Path("data/paper_digest/pdf_cache")

    pdf = PdfExtract()
    pdf_path = download_arxiv_pdf(paper.arxiv_id, cache)
    if pdf_path:
        logger.info("解析 PDF: %s", pdf_path.name)
        pdf = extract_from_pdf(pdf_path)
    else:
        logger.warning("未获取到 PDF，回退到摘要分析: %s", paper.arxiv_id)

    brief = _build_brief_summary(paper, pdf, topics)
    core_idea = _build_core_idea(paper, pdf)
    highlights = _build_experiment_highlights(pdf, paper)

    return PaperAnalysis(
        brief_summary=brief,
        core_idea=core_idea,
        architecture_figures=pdf.architecture_figures,
        result_figures=pdf.result_figures,
        result_tables=pdf.result_tables,
        experiment_highlights=highlights,
        topics=topics,
        experiment_results=highlights,
    )
