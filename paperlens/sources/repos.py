"""Find code repositories for a paper.

Two evidence levels, kept separate in the UI:
  * 'paper'  - a GitHub URL written in the paper itself (abstract, arXiv comment, PDF text). High confidence.
  * 'search' - a GitHub repository whose README/description mentions the paper's arXiv ID. Lower confidence.
"""
from __future__ import annotations

import re

import httpx

from .. import config
from .cache import Cache, NullCache

GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I)
_NOT_REPOS = {"orgs", "topics", "features", "about", "sponsors", "settings", "marketplace"}


def extract_github_urls(text: str | None) -> list[str]:
    if not text:
        return []
    # PDFs break URLs across lines: "github.com/foo/ \nbar" -> join obvious continuations
    text = re.sub(r"(github\.com/[\w.-]+/?)\s*\n\s*([\w.-]+)", r"\1\2", text)
    out = []
    for owner, repo in GITHUB_RE.findall(text):
        repo = re.sub(r"(\.git|[.,;:)\]]+)$", "", repo)
        if owner.lower() in _NOT_REPOS or not repo:
            continue
        url = f"https://github.com/{owner}/{repo}"
        if url.lower() not in {u.lower() for u in out}:
            out.append(url)
    return out


class GitHubClient:
    def __init__(self, token: str | None = config.GITHUB_TOKEN, cache: Cache | None = None):
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "paperlens"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(headers=headers, timeout=20.0)
        self.cache = cache or NullCache()

    def _get(self, url: str, params=None):
        key = f"gh:{url}:{params}"
        hit = self.cache.get(key, max_age=3 * 24 * 3600)
        if hit is not None:
            return hit
        r = self.http.get(url, params=params)
        if r.status_code in (403, 404, 422, 429):
            return None
        r.raise_for_status()
        data = r.json()
        self.cache.set(key, data)
        return data

    @staticmethod
    def _summ(d: dict, source: str) -> dict:
        return {
            "url": d.get("html_url"),
            "name": d.get("full_name"),
            "description": d.get("description"),
            "stars": d.get("stargazers_count"),
            "language": d.get("language"),
            "updated": d.get("pushed_at"),
            "source": source,
        }

    def repo_info(self, url: str, source: str = "paper") -> dict:
        m = GITHUB_RE.match(url)
        if not m:
            return {"url": url, "source": source}
        data = self._get(f"https://api.github.com/repos/{m.group(1)}/{m.group(2)}")
        return self._summ(data, source) if data else {"url": url, "name": f"{m.group(1)}/{m.group(2)}",
                                                       "source": source}

    def search_by_arxiv(self, arxiv_id: str, n: int = 3) -> list[dict]:
        data = self._get("https://api.github.com/search/repositories",
                         params={"q": f'"{arxiv_id}" in:readme,description', "sort": "stars", "per_page": n})
        return [self._summ(d, "search") for d in (data or {}).get("items", [])[:n]]
