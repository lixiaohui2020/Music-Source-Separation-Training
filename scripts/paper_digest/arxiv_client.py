from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    pdf_url: str
    arxiv_url: str
    categories: list[str] = field(default_factory=list)
    comment: str = ""

    @property
    def stable_id(self) -> str:
        return self.arxiv_id.split("v")[0]


def _parse_arxiv_datetime(value: str) -> datetime:
    # Example: 2024-01-15T12:34:56Z
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_arxiv_id(entry: ET.Element) -> str:
    raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    match = re.search(r"arxiv\.org/abs/(.+)$", raw_id)
    if match:
        return match.group(1)
    return raw_id.rsplit("/", 1)[-1]


def _parse_entry(entry: ET.Element) -> Paper:
    arxiv_id = _extract_arxiv_id(entry)
    title = _clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
    abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
    published = _parse_arxiv_datetime(entry.findtext("atom:published", namespaces=ATOM_NS))
    updated = _parse_arxiv_datetime(entry.findtext("atom:updated", namespaces=ATOM_NS))
    authors = [
        _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    categories = [cat.get("term", "") for cat in entry.findall("atom:category", ATOM_NS)]
    comment = _clean_text(entry.findtext("arxiv:comment", default="", namespaces=ATOM_NS))
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        published=published,
        updated=updated,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        categories=categories,
        comment=comment,
    )


def build_query(search_queries: list[str], categories: list[str]) -> str:
    query_parts = [f"({q})" for q in search_queries]
    combined = " OR ".join(query_parts)
    if categories:
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        combined = f"({combined}) AND ({cat_query})"
    return combined


def fetch_recent_papers(
    search_queries: list[str],
    categories: list[str],
    lookback_days: int = 7,
    max_results: int = 50,
) -> list[Paper]:
    query = build_query(search_queries, categories)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API, params=params, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        paper = _parse_entry(entry)
        if paper.published >= cutoff:
            papers.append(paper)

    # arXiv asks for 3-second delay between requests
    time.sleep(3)
    return papers


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in sorted(papers, key=lambda p: p.published, reverse=True):
        key = paper.stable_id
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique
