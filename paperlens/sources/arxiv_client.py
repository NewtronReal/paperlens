"""arXiv export API (Atom feed). Used for the author's 'comment' field, which often holds code links,
and to batch-look-up referenced arXiv papers when backtracking to their repositories."""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import httpx

from .cache import Cache, NullCache

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

ARXIV_ID_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?![\d])")
OLD_ARXIV_ID_RE = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")


def normalize_arxiv_id(text: str) -> str | None:
    m = ARXIV_ID_RE.search(text) or OLD_ARXIV_ID_RE.search(text)
    return m.group(1) if m else None


def _parse_entry(e: ET.Element) -> dict:
    def t(tag):
        el = e.find(tag, NS)
        return " ".join(el.text.split()) if el is not None and el.text else None

    links = {l.get("title") or l.get("rel"): l.get("href") for l in e.findall("a:link", NS)}
    raw_id = t("a:id") or ""
    return {
        "arxiv_id": normalize_arxiv_id(raw_id.split("/abs/")[-1]),
        "title": t("a:title"),
        "abstract": t("a:summary"),
        "published": t("a:published"),
        "authors": [a.findtext("a:name", namespaces=NS) for a in e.findall("a:author", NS)],
        "comment": t("arxiv:comment"),
        "categories": [c.get("term") for c in e.findall("a:category", NS)],
        "pdf_url": links.get("pdf"),
        "abs_url": links.get("alternate"),
    }


class ArxivClient:
    def __init__(self, cache: Cache | None = None, timeout: float = 30.0):
        self.http = httpx.Client(timeout=timeout, follow_redirects=True)
        self.cache = cache or NullCache()
        self._last = 0.0

    def _get(self, params: dict) -> list[dict]:
        key = "arxiv:" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        hit = self.cache.get(key, max_age=30 * 24 * 3600)
        if hit is not None:
            return hit
        wait = 3.0 - (time.time() - self._last)  # arXiv asks for <= 1 request / 3 s
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        r = self.http.get(API, params=params)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        entries = [_parse_entry(e) for e in root.findall("a:entry", NS)]
        entries = [e for e in entries if e.get("title") and e["title"] != "Error"]
        self.cache.set(key, entries)
        return entries

    def by_ids(self, ids: list[str]) -> dict[str, dict]:
        ids = [i for i in dict.fromkeys(ids) if i]
        out = {}
        for i in range(0, len(ids), 50):
            for e in self._get({"id_list": ",".join(ids[i:i + 50]), "max_results": 50}):
                if e["arxiv_id"]:
                    out[e["arxiv_id"]] = e
        return out

    def by_id(self, arxiv_id: str) -> dict | None:
        return self.by_ids([arxiv_id]).get(arxiv_id)

    def search_title(self, title: str, n: int = 5) -> list[dict]:
        q = 'ti:"' + re.sub(r"[^\w\s-]", " ", title).strip() + '"'
        return self._get({"search_query": q, "max_results": n})
