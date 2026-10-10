"""End-to-end analysis of one paper.

resolve -> metadata -> references & citations (+contexts) -> [Model 1] citation intent for every context
-> local citation graph + SPECTER2 node features -> [Model 2] influence score for every edge
-> lineage (recursive most-influential ancestors) -> idea evolution clusters -> code repositories -> summary
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

import numpy as np

from .. import config
from ..models.influence.features import edge_features, node_features
from ..models.influence.scorer import load_influence_scorer
from ..models.intent.classifier import load_intent_classifier
from ..sources.cache import Cache
from ..sources.repos import extract_github_urls
from ..sources.s2_client import S2Error
from .evolution import build_evolution
from .resolve import to_identifier
from .summary import extractive_summary, llm_summary

log = logging.getLogger(__name__)
LABELS = config.INTENT_LABELS


def _authors(p: dict, n: int = 6) -> list[str]:
    return [a.get("name") for a in (p.get("authors") or [])[:n] if a.get("name")]


def _surname(p: dict) -> str | None:
    a = _authors(p, 1)
    return a[0].split()[-1] if a else None


class Analyzer:
    def __init__(self, s2=None, arxiv=None, github=None, demo: bool | None = None):
        self.demo = config.DEMO if demo is None else demo
        if self.demo:
            from ..sources.demo import DemoGitHub, DemoS2Client
            self.s2, self.arxiv, self.github = s2 or DemoS2Client(), None, github or DemoGitHub()
        else:
            from ..sources.arxiv_client import ArxivClient
            from ..sources.repos import GitHubClient
            from ..sources.s2_client import S2Client
            cache = Cache(config.CACHE_DB)
            self.s2 = s2 or S2Client(cache=cache)
            self.arxiv = arxiv if arxiv is not None else ArxivClient(cache=cache)
            self.github = github or GitHubClient(cache=cache)
        self.intent = load_intent_classifier()
        self.influence = load_influence_scorer()
        self._emb: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ helpers
    def _embeddings(self, ids) -> dict[str, list[float]]:
        missing = [i for i in dict.fromkeys(ids) if i and i not in self._emb]
        if missing:
            try:
                self._emb.update(self.s2.embeddings(missing))
            except S2Error as e:
                log.warning("embedding fetch failed: %s", e)
        return self._emb

    def _classify_edges(self, edges: list[dict]) -> None:
        """Model 1: attach per-context and per-edge intent to each edge in place."""
        flat = [(k, c) for k, e in enumerate(edges) for c in e["contexts"]]
        probs = self.intent.predict_proba([c for _, c in flat]) if flat else np.zeros((0, 3))
        per = [[] for _ in edges]
        for (k, c), p in zip(flat, probs):
            per[k].append((c, p))
        for e, items in zip(edges, per):
            e["_P"] = np.stack([p for _, p in items]) if items else None
            if items:
                mean = e["_P"].mean(0)
                e["intent"] = LABELS[int(mean.argmax())]
                e["intent_probs"] = {l: round(float(v), 3) for l, v in zip(LABELS, mean)}
                e["context_view"] = [{"text": c, "intent": LABELS[int(p.argmax())],
                                      "confidence": round(float(p.max()), 3)} for c, p in items[:8]]
            else:
                e["intent"], e["intent_probs"], e["context_view"] = "unknown", None, []

    def _score_star(self, center: dict, edges: list[dict], direction: str) -> None:
        """Model 2 on a star graph around `center`.
        direction='out': center cites each edge paper (references); 'in': edge papers cite center (citations)."""
        if not edges:
            return
        papers = [center] + [e["paper"] for e in edges]
        emb = self._embeddings([p["paperId"] for p in papers])
        x = np.stack([node_features(emb.get(p["paperId"]), p.get("citationCount"), p.get("year")) for p in papers])
        others = np.arange(1, len(papers))
        src, dst = (np.zeros_like(others), others) if direction == "out" else (others, np.zeros_like(others))
        ea = np.stack([edge_features(e.get("_P"), len(e["contexts"]),
                                     (center if direction == "out" else e["paper"]).get("year"),
                                     (e["paper"] if direction == "out" else center).get("year")) for e in edges])
        ei = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], 1)
        s = self.influence.score(x, ei, src, dst, ea)
        for e, v in zip(edges, s):
            e["influence"] = round(float(v), 4)

    def _fetch_edges(self, pid: str, kind: str, n: int) -> list[dict]:
        try:
            return (self.s2.references if kind == "references" else self.s2.citations)(pid, n)
        except S2Error as e:
            log.warning("%s fetch failed for %s: %s", kind, pid, e)
            return []

    @staticmethod
    def _why(e: dict) -> str:
        n = len(e["contexts"])
        bits = [f"mentioned {n}×" if n else "no citation sentence available"]
        if e.get("intent") not in (None, "unknown"):
            bits.append({"method": "used as method/tool", "background": "cited as background",
                         "result": "compared against / results"}[e["intent"]])
        if e.get("isInfluential"):
            bits.append("S2 marks it influential")
        return " · ".join(bits)

    def _edge_view(self, e: dict, role: str) -> dict:
        p = e["paper"]
        return {
            "paperId": p["paperId"], "title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"),
            "authors": _authors(p, 4), "citationCount": p.get("citationCount") or 0,
            "arxiv": (p.get("externalIds") or {}).get("ArXiv"), "role": role,
            "intent": e.get("intent"), "intent_probs": e.get("intent_probs"),
            "influence": e.get("influence"), "s2_influential": e.get("isInfluential"),
            "n_contexts": len(e["contexts"]), "contexts": e.get("context_view", []),
            "context_source": e.get("context_source", "s2"), "why": self._why(e),
        }

    # ------------------------------------------------------------------ main entry
    def analyze(self, query: str | None = None, pdf_bytes: bytes | None = None,
                progress: Callable[[str, float], None] | None = None) -> dict:
        prog = progress or (lambda msg, frac: None)
        warnings: list[str] = []
        if self.demo:
            warnings.append("DEMO MODE: synthetic papers and citations, not real data.")
        if "heuristic" in self.intent.name:
            warnings.append("Citation-intent model not trained yet: using a keyword heuristic.")
        if "heuristic" in self.influence.name:
            warnings.append("Influence model not trained yet: using a transparent heuristic.")

        # 1. resolve -------------------------------------------------------------------------
        prog("Resolving paper", 0.03)
        pdf = None
        if pdf_bytes:
            from ..sources.pdf_parse import parse_pdf
            pdf = parse_pdf(pdf_bytes)
        paper = self._resolve(query, pdf)
        pid = paper["paperId"]
        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv") or (pdf.arxiv_id if pdf else None)

        # 2. edges -----------------------------------------------------------------------------
        prog("Fetching references", 0.12)
        refs = self._fetch_edges(pid, "references", config.MAX_REFERENCES)
        prog("Fetching citing papers", 0.25)
        cits = self._fetch_edges(pid, "citations", config.MAX_CITATIONS)
        if pdf:
            n_added = 0
            for e in refs:
                if not e["contexts"]:
                    ctx = pdf.contexts_for(e["paper"].get("title"), _surname(e["paper"]), e["paper"].get("year"))
                    if ctx:
                        e["contexts"], e["context_source"] = ctx, "pdf"
                        n_added += 1
            if n_added:
                warnings.append(f"Recovered citation sentences for {n_added} references from the uploaded PDF.")
        if not refs:
            warnings.append("No references available from Semantic Scholar for this paper.")

        # 3. Model 1 ---------------------------------------------------------------------------
        prog("Classifying citation intent (Model 1)", 0.4)
        self._classify_edges(refs)
        self._classify_edges(cits)

        # 4. Model 2 ---------------------------------------------------------------------------
        prog("Scoring citation influence (Model 2)", 0.52)
        self._score_star(paper, refs, "out")
        self._score_star(paper, cits, "in")
        refs.sort(key=lambda e: -(e.get("influence") or 0))
        cits.sort(key=lambda e: -((e.get("influence") or 0) + 0.02 * np.log1p(e["paper"].get("citationCount") or 0)))

        # 5. lineage ---------------------------------------------------------------------------
        prog("Tracing lineage", 0.62)
        lineage = self._lineage(paper, refs)

        # 6. idea evolution --------------------------------------------------------------------
        prog("Mapping idea evolution", 0.78)
        evo_papers = [{**paper, "role": "focus", "influence": None}]
        seen = {pid}
        for e, role in [(e, "ancestor") for e in refs] + [(e, "descendant") for e in cits[:200]]:
            q = e["paper"]
            if q["paperId"] not in seen:
                seen.add(q["paperId"])
                evo_papers.append({**q, "role": role, "influence": e.get("influence")})
        for n in lineage["nodes"]:
            if n["paperId"] not in seen and n.get("_paper"):
                seen.add(n["paperId"])
                evo_papers.append({**n["_paper"], "role": "ancestor", "influence": n.get("influence")})
        evolution = build_evolution(evo_papers, self._embeddings([p["paperId"] for p in evo_papers]))

        # 7. repositories ----------------------------------------------------------------------
        prog("Backtracking code repositories", 0.88)
        repos = self._repos(paper, arxiv_id, refs, pdf)

        # 8. summary ---------------------------------------------------------------------------
        prog("Writing the one-glance summary", 0.95)
        ref_views = [self._edge_view(e, "reference") for e in refs]
        cit_views = [self._edge_view(e, "citation") for e in cits]
        summary = llm_summary(paper, ref_views, pdf.body if pdf else None) or extractive_summary(paper)

        for n in lineage["nodes"]:
            n.pop("_paper", None)
        result = {
            "paper": self._paper_view(paper, arxiv_id, pdf),
            "summary": summary,
            "stats": self._stats(paper, ref_views, cit_views),
            "models": {"intent": self.intent.name, "influence": self.influence.name,
                       "influence_threshold": getattr(self.influence, "threshold", 0.5)},
            "references": ref_views,
            "citations": cit_views,
            "graph": self._graph(paper, ref_views, cit_views[:120]),
            "lineage": lineage,
            "evolution": evolution,
            "repos": repos,
            "warnings": warnings,
            "demo": self.demo,
        }
        prog("Done", 1.0)
        return result

    # ------------------------------------------------------------------ pieces
    def _resolve(self, query, pdf) -> dict:
        tried = []
        if query:
            ident, kind = to_identifier(query)
            if ident:
                tried.append(ident)
                try:
                    return self.s2.paper(ident)
                except S2Error:
                    pass
            hit = self.s2.search_title(query)
            if hit:
                return hit
        if pdf:
            for ident in ([f"arXiv:{pdf.arxiv_id}"] if pdf.arxiv_id else []) + ([f"DOI:{pdf.doi}"] if pdf.doi else []):
                tried.append(ident)
                try:
                    return self.s2.paper(ident)
                except S2Error:
                    pass
            if pdf.title:
                hit = self.s2.search_title(pdf.title)
                if hit:
                    return hit
        if self.demo:
            return self.s2.paper(self.s2.w.default_focus())
        raise S2Error(f"Could not find this paper on Semantic Scholar (tried {tried or 'title search'}).")

    def _lineage(self, paper: dict, refs: list[dict]) -> dict:
        """Ancestry tree: at each level keep the `branch` most influential references (by Model 2)."""
        depth, branch = config.LINEAGE_DEPTH, config.LINEAGE_BRANCH
        nodes = [{"paperId": paper["paperId"], "title": paper.get("title"), "year": paper.get("year"),
                  "level": 0, "influence": None, "intent": None, "citationCount": paper.get("citationCount") or 0}]
        edges, seen = [], {paper["paperId"]}
        frontier = [(paper, refs)]
        for level in range(1, depth + 1):
            nxt = []
            for parent, parent_refs in frontier:
                picked = [e for e in parent_refs if e["paper"]["paperId"] not in seen][:branch]
                for e in picked:
                    q = e["paper"]
                    seen.add(q["paperId"])
                    nodes.append({"paperId": q["paperId"], "title": q.get("title"), "year": q.get("year"),
                                  "level": level, "influence": e.get("influence"), "intent": e.get("intent"),
                                  "citationCount": q.get("citationCount") or 0, "_paper": q})
                    edges.append({"source": q["paperId"], "target": parent["paperId"],
                                  "influence": e.get("influence"), "intent": e.get("intent")})
                    if level < depth:
                        sub = self._fetch_edges(q["paperId"], "references", 300)
                        self._classify_edges(sub)
                        self._score_star(q, sub, "out")
                        sub.sort(key=lambda x: -(x.get("influence") or 0))
                        nxt.append((q, sub))
            frontier = nxt
        return {"nodes": nodes, "edges": edges, "depth": depth, "branch": branch}

    def _repos(self, paper, arxiv_id, refs, pdf) -> dict:
        if self.demo:
            from ..sources.demo import demo_repos_for
            return {"paper": demo_repos_for(paper),
                    "references": [{"paperId": e["paper"]["paperId"], "title": e["paper"].get("title"),
                                    "influence": e.get("influence"), "repos": demo_repos_for(e["paper"])}
                                   for e in refs[:8] if demo_repos_for(e["paper"])]}
        found: list[str] = []
        texts = [paper.get("abstract")]
        meta = {}
        top = [e for e in refs[:10] if (e["paper"].get("externalIds") or {}).get("ArXiv")]
        if self.arxiv:
            try:
                meta = self.arxiv.by_ids([arxiv_id] + [e["paper"]["externalIds"]["ArXiv"] for e in top]
                                         if arxiv_id else [e["paper"]["externalIds"]["ArXiv"] for e in top])
            except Exception as e:  # noqa: BLE001
                log.warning("arXiv lookup failed: %s", e)
        if arxiv_id and arxiv_id in meta:
            texts += [meta[arxiv_id].get("comment"), meta[arxiv_id].get("abstract")]
        if pdf:
            found += pdf.github
        for t in texts:
            found += extract_github_urls(t)
        found = list(dict.fromkeys(found))[:5]
        own = [self.github.repo_info(u, "paper") for u in found]
        if not own and arxiv_id:
            own = self.github.search_by_arxiv(arxiv_id)
        ref_repos, searches = [], 0
        for e in top[:8]:
            aid = e["paper"]["externalIds"]["ArXiv"]
            m = meta.get(aid, {})
            urls = extract_github_urls(m.get("comment")) + extract_github_urls(m.get("abstract")) \
                + extract_github_urls(e["paper"].get("abstract"))
            rs = [self.github.repo_info(u, "paper") for u in list(dict.fromkeys(urls))[:2]]
            if not rs and searches < 3:   # GitHub search is rate-limited: only for the top few
                rs, searches = self.github.search_by_arxiv(aid, 2), searches + 1
            if rs:
                ref_repos.append({"paperId": e["paper"]["paperId"], "title": e["paper"].get("title"),
                                  "influence": e.get("influence"), "repos": rs})
        return {"paper": own, "references": ref_repos}

    @staticmethod
    def _paper_view(p, arxiv_id, pdf) -> dict:
        return {
            "paperId": p["paperId"], "title": p.get("title") or (pdf.title if pdf else None),
            "authors": _authors(p, 12), "n_authors": len(p.get("authors") or []),
            "year": p.get("year"), "venue": p.get("venue"), "date": p.get("publicationDate"),
            "abstract": p.get("abstract"), "fields": p.get("fieldsOfStudy") or [],
            "citationCount": p.get("citationCount"), "influentialCitationCount": p.get("influentialCitationCount"),
            "referenceCount": p.get("referenceCount"), "arxiv": arxiv_id,
            "doi": (p.get("externalIds") or {}).get("DOI") or (pdf.doi if pdf else None),
            "url": p.get("url"), "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
            "pdf_pages": pdf.n_pages if pdf else None,
        }

    @staticmethod
    def _stats(p, refs, cits) -> dict:
        intents = Counter(r["intent"] for r in refs)
        years = [r["year"] for r in refs if r.get("year")]
        infl = [r for r in refs if r.get("influence") is not None]
        return {
            "n_references": len(refs), "n_citations_analyzed": len(cits),
            "intent_counts": {k: intents.get(k, 0) for k in LABELS + ["unknown"]},
            "n_core_references": sum(r["influence"] >= 0.5 for r in infl),
            "median_reference_year": float(np.median(years)) if years else None,
            "median_reference_age": float(np.median([(p.get("year") or y) - y for y in years])) if years else None,
            "top_reference_venues": Counter(r["venue"] for r in refs if r.get("venue")).most_common(5),
        }

    @staticmethod
    def _graph(p, refs, cits) -> dict:
        nodes = [{"id": p["paperId"], "label": p.get("title"), "type": "focus", "year": p.get("year"),
                  "citationCount": p.get("citationCount") or 0}]
        edges = []
        for r in refs:
            nodes.append({"id": r["paperId"], "label": r["title"], "type": "reference", "year": r["year"],
                          "intent": r["intent"], "influence": r["influence"], "citationCount": r["citationCount"]})
            edges.append({"source": p["paperId"], "target": r["paperId"], "intent": r["intent"],
                          "influence": r["influence"]})
        known = {n["id"] for n in nodes}
        for c in cits:
            if c["paperId"] in known:
                continue
            nodes.append({"id": c["paperId"], "label": c["title"], "type": "citation", "year": c["year"],
                          "intent": c["intent"], "influence": c["influence"], "citationCount": c["citationCount"]})
            edges.append({"source": c["paperId"], "target": p["paperId"], "intent": c["intent"],
                          "influence": c["influence"]})
        return {"nodes": nodes, "edges": edges}

