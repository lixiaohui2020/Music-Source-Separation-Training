from __future__ import annotations

import re
from urllib.parse import urlparse

import requests

from scripts.paper_digest.arxiv_client import Paper

_GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/[\w\-.]+/[\w\-.]+/?",
    flags=re.IGNORECASE,
)


def _normalize_github_url(url: str) -> str:
    parsed = urlparse(url.rstrip("/"))
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        return f"https://github.com/{parts[0]}/{parts[1]}"
    return url.rstrip("/")


def extract_github_links_from_text(*texts: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _GITHUB_URL_RE.findall(text):
            normalized = _normalize_github_url(match)
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                found.append(normalized)
    return found


def search_github_by_title(title: str, timeout: int = 15) -> list[str]:
    """Best-effort GitHub search using the paper title."""
    query = re.sub(r"[^\w\s\-]", " ", title)
    query = re.sub(r"\s+", " ", query).strip()
    if len(query) < 8:
        return []

    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": f"{query} in:name,description", "sort": "stars", "per_page": 3},
            headers={"Accept": "application/vnd.github+json"},
            timeout=timeout,
        )
        if response.status_code != 200:
            return []
        items = response.json().get("items", [])
        links = []
        for item in items[:2]:
            html_url = item.get("html_url")
            if html_url:
                links.append(html_url)
        return links
    except requests.RequestException:
        return []


def find_github_links(paper: Paper, enable_search: bool = True) -> list[str]:
    links = extract_github_links_from_text(paper.abstract, paper.comment)
    if not links and enable_search:
        links = search_github_by_title(paper.title)
    return links
