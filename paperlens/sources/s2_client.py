"""Semantic Scholar Graph API client (metadata, references/citations with contexts, SPECTER2 embeddings).

Docs: https://api.semanticscholar.org/api-docs/graph
Without an API key the shared pool is heavily rate-limited; request a free key and set S2_API_KEY.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Iterable

import httpx

from .. import config
from .cache import Cache, NullCache

log = logging.getLogger(__name__)

BASE = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,publicationDate,authors,citationCount,"
    "influentialCitationCount,referenceCount,fieldsOfStudy,tldr,openAccessPdf,url"
)
# Fields for the nested paper inside a reference/citation edge
EDGE_PAPER_FIELDS = "paperId,externalIds,title,abstract,year,venue,authors,citationCount"
EDGE_FIELDS = "contexts,intents,isInfluential," + EDGE_PAPER_FIELDS


class S2Error(RuntimeError):
    pass


class S2Client:
    """Synchronous client with throttling, retry on 429/5xx and an on-disk cache."""

    def __init__(self, api_key: str | None = config.S2_API_KEY, cache: Cache | None = None,
                 min_interval: float | None = None, timeout: float = 30.0):
        self.api_key = api_key
        headers = {"x-api-key": api_key} if api_key else {}
        self.http = httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)
        self.cache = cache if cache is not None else Cache(config.CACHE_DB)
        self.min_interval = min_interval if min_interval is not None else (1.05 if api_key else 1.5)
        self._last = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ low level
    def _throttle(self):
        with self._lock:
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def _request(self, method: str, path: str, *, params=None, json=None, cache_key=None,
                 max_age=7 * 24 * 3600) -> Any:
        if cache_key:
            hit = self.cache.get(cache_key, max_age=max_age)
            if hit is not None:
                return hit
        delay = 2.0
        for attempt in range(7):
            self._throttle()
            try:
                r = self.http.request(method, BASE + path, params=params, json=json)
            except httpx.HTTPError as e:
                log.warning("S2 network error %s (attempt %d)", e, attempt)
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 404:
                raise S2Error(f"Not found on Semantic Scholar: {path}")
            if r.status_code in (429, 500, 502, 503, 504):
                log.info("S2 %s, backing off %.1fs", r.status_code, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code >= 400:
                raise S2Error(f"S2 error {r.status_code}: {r.text[:200]}")
            data = r.json()
            if cache_key:
                self.cache.set(cache_key, data)
            return data
        raise S2Error("Semantic Scholar rate limit: retries exhausted (set S2_API_KEY)")

    # ------------------------------------------------------------------ public API
    def paper(self, paper_id: str, fields: str = PAPER_FIELDS) -> dict:
        return self._request("GET", f"/paper/{paper_id}", params={"fields": fields},
                             cache_key=f"paper:{paper_id}:{fields}")

    def search_title(self, title: str) -> dict | None:
        try:
            data = self._request("GET", "/paper/search/match",
                                 params={"query": title, "fields": PAPER_FIELDS},
                                 cache_key=f"match:{title.lower()}")
        except S2Error:
            return None
        items = data.get("data") or []
        return items[0] if items else None

    def search(self, query: str, limit: int = 10) -> list[dict]:
        data = self._request("GET", "/paper/search",
                             params={"query": query, "limit": limit, "fields": EDGE_PAPER_FIELDS},
                             cache_key=f"search:{query.lower()}:{limit}")
        return data.get("data") or []

    def _edges(self, paper_id: str, kind: str, max_items: int) -> list[dict]:
        """kind = 'references' (cited papers) or 'citations' (citing papers)."""
        inner = "citedPaper" if kind == "references" else "citingPaper"
        out, offset, page = [], 0, min(500, max_items)
        while offset < max_items:
            data = self._request(
                "GET", f"/paper/{paper_id}/{kind}",
                params={"fields": EDGE_FIELDS, "offset": offset, "limit": page},
                cache_key=f"{kind}:{paper_id}:{offset}:{page}",
            )
            for item in data.get("data") or []:
                p = item.get(inner) or {}
                if not p.get("paperId"):
                    continue  # unresolved reference
                out.append({
                    "paper": p,
                    "contexts": item.get("contexts") or [],
                    "s2_intents": item.get("intents") or [],
                    "isInfluential": bool(item.get("isInfluential")),
                })
            nxt = data.get("next")
            if nxt is None or not data.get("data"):
                break
            offset = nxt
        return out[:max_items]

    def references(self, paper_id: str, max_items: int = 500) -> list[dict]:
        return self._edges(paper_id, "references", max_items)

    def citations(self, paper_id: str, max_items: int = 400) -> list[dict]:
        return self._edges(paper_id, "citations", max_items)

    def batch(self, ids: Iterable[str], fields: str) -> list[dict | None]:
        ids = list(ids)
        out: list[dict | None] = []
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            key = f"batch:{fields}:{hashlib.sha1('|'.join(chunk).encode()).hexdigest()}"
            data = self._request("POST", "/paper/batch", params={"fields": fields},
                                 json={"ids": chunk}, cache_key=key)
            out.extend(data)
        return out

    def embeddings(self, ids: Iterable[str]) -> dict[str, list[float]]:
        """SPECTER2 vectors (768-d) keyed by paperId; missing papers are omitted."""
        ids = [i for i in dict.fromkeys(ids) if i]
        res = {}
        for pid, item in zip(ids, self.batch(ids, "paperId,embedding.specter_v2")):
            emb = (item or {}).get("embedding") or {}
            if emb.get("vector"):
                res[pid] = emb["vector"]
        return res
