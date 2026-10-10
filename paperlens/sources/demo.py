"""Deterministic synthetic citation world.

Used for (1) the offline demo mode (PAPERLENS_DEMO=1), (2) tests, and (3) smoke-testing the training
pipelines without network access. It mimics the Semantic Scholar client interface. Nothing here is real
data; the UI shows a DEMO banner whenever this world is used.

Generative story (so models have real signal to find):
  * papers belong to latent topics; SPECTER-like embeddings = topic centroid + year drift + noise
  * papers cite earlier papers, preferring the same topic and highly cited work
  * each citation edge is latently 'core' or 'peripheral'; core edges are mentioned more often,
    are more often cited for their method, and are usually labelled influential
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

TOPICS = [
    ("attention", ["attention", "transformer", "self-attention", "encoder-decoder", "sequence transduction",
                   "positional encoding", "machine translation"]),
    ("graph", ["graph neural network", "message passing", "node embedding", "graph convolution",
               "link prediction", "relational reasoning"]),
    ("contrastive", ["contrastive learning", "visual representation", "data augmentation",
                     "self-supervised pretraining", "image encoder", "momentum encoder"]),
    ("ssm", ["state space model", "long-range sequence", "selective scan", "linear recurrence",
             "structured kernel", "long context"]),
    ("rag", ["retrieval augmentation", "dense retrieval", "passage ranking", "knowledge-intensive generation",
             "retriever", "open-domain question answering"]),
]
ADJ = ["Efficient", "Scalable", "Robust", "Simple", "Deep", "Hierarchical", "Sparse", "Adaptive",
       "Unified", "Generalized", "Rethinking", "Towards Better", "Learning"]
SURNAMES = ["Chen", "Wang", "Smith", "Kumar", "Nair", "Garcia", "Mueller", "Tanaka", "Kim", "Ivanova",
            "Rossi", "Okafor", "Menon", "Dubois", "Silva", "Cohen", "Park", "Li", "Zhang", "Joseph"]
TEMPLATES = {
    "background": [
        "Prior work has studied {kw} extensively [CIT].",
        "{Kw} has become a standard tool in recent years [CIT].",
        "Several approaches address {kw} in different settings [CIT].",
        "A broad survey of {kw} is given in [CIT].",
    ],
    "method": [
        "We follow the {kw} formulation of [CIT].",
        "Our encoder is based on the {kw} architecture introduced in [CIT].",
        "We use the {kw} implementation from [CIT] with default hyperparameters.",
        "Following [CIT], we adopt {kw} as the backbone.",
    ],
    "result": [
        "Our model outperforms the {kw} baseline of [CIT].",
        "These results are consistent with the findings on {kw} reported in [CIT].",
        "Compared to [CIT], our approach improves {kw} accuracy.",
    ],
}


def _pid(i: int) -> str:
    return f"demo{i:04d}"


class DemoWorld:
    def __init__(self, n_papers: int = 420, dim: int = 768, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        n_t = len(TOPICS)
        years = np.sort(rng.integers(2012, 2026, size=n_papers))
        # later years lean toward newer topics (topic drift)
        topic = np.empty(n_papers, dtype=int)
        for i, y in enumerate(years):
            w = np.array([1.5, 1.2, 1.0, 0.2 + 0.25 * max(0, y - 2019), 0.2 + 0.2 * max(0, y - 2018)])
            topic[i] = rng.choice(n_t, p=w / w.sum())
        centroids = rng.normal(0, 1, (n_t, dim))
        drift = rng.normal(0, 1, (n_t, dim))
        emb = centroids[topic] + 0.35 * drift[topic] * ((years - 2018) / 7.0)[:, None] \
            + rng.normal(0, 0.9, (n_papers, dim))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        self.papers: list[dict] = []
        for i in range(n_papers):
            t = topic[i]
            kws = TOPICS[t][1]
            k1, k2 = rng.choice(len(kws), 2, replace=False)
            title = f"{rng.choice(ADJ)} {kws[k1].title()} for {kws[k2].title()}"
            authors = [{"name": f"{rng.choice(['A.', 'R.', 'S.', 'M.', 'J.', 'K.'])} {rng.choice(SURNAMES)}"}
                       for _ in range(rng.integers(2, 6))]
            abstract = (f"{kws[k1].capitalize()} remains a central problem. We propose a method that combines "
                        f"{kws[k1]} with {kws[k2]}. Experiments on standard benchmarks show that our approach "
                        f"improves over strong baselines while being simpler to train. "
                        f"We release code and pretrained models.")
            self.papers.append({
                "paperId": _pid(i), "title": title, "abstract": abstract, "year": int(years[i]),
                "venue": rng.choice(["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "CVPR", "arXiv"]),
                "authors": authors, "externalIds": {"DEMO": str(i)},
                "fieldsOfStudy": ["Computer Science"], "url": None, "openAccessPdf": None,
                "tldr": {"text": f"Combines {kws[k1]} with {kws[k2]} and beats strong baselines."},
                "_topic": int(t), "_kws": [kws[k1], kws[k2]],
            })
        self.emb = emb

        # citation edges
        self.refs: dict[int, list[dict]] = {i: [] for i in range(n_papers)}
        self.cits: dict[int, list[dict]] = {i: [] for i in range(n_papers)}
        indeg = np.zeros(n_papers)
        for i in range(n_papers):
            earlier = np.where(years < years[i])[0]
            if len(earlier) == 0:
                continue
            k = int(min(len(earlier), rng.integers(12, 40)))
            same = (topic[earlier] == topic[i]).astype(float)
            w = (1 + 3.0 * same) * (1 + indeg[earlier]) ** 0.8
            chosen = rng.choice(earlier, size=k, replace=False, p=w / w.sum())
            for j in chosen:
                sim = float(emb[i] @ emb[j])
                p_core = 1 / (1 + math.exp(-(-4.6 + 2.0 * (topic[i] == topic[j]) + 4.0 * sim
                                              + 0.25 * math.log1p(indeg[j]))))
                core = rng.random() < p_core
                n_ctx = 1 + rng.poisson(2.2 if core else 0.4)
                if rng.random() < 0.12:
                    n_ctx = 0  # some edges have no recoverable context, as in real S2 data
                contexts, intents = [], set()
                for _ in range(n_ctx):
                    probs = [0.25, 0.6, 0.15] if core else [0.72, 0.08, 0.20]
                    intent = ["background", "method", "result"][rng.choice(3, p=probs)]
                    kw = rng.choice(self.papers[j]["_kws"])
                    tpl = rng.choice(TEMPLATES[intent])
                    contexts.append(tpl.format(kw=kw, Kw=kw.capitalize()))
                    intents.add("methodology" if intent == "method" else intent)
                infl = core if rng.random() > 0.1 else not core
                edge = {"contexts": contexts, "s2_intents": sorted(intents), "isInfluential": bool(infl)}
                self.refs[i].append({**edge, "_j": int(j)})
                self.cits[int(j)].append({**edge, "_j": i})
                indeg[j] += 1
        for i, p in enumerate(self.papers):
            p["citationCount"] = int(indeg[i] * 7 + rng.integers(0, 15))
            p["influentialCitationCount"] = sum(e["isInfluential"] for e in self.cits[i])
            p["referenceCount"] = len(self.refs[i])
        self._by_id = {p["paperId"]: i for i, p in enumerate(self.papers)}

    # ------------------------------------------------------------------ helpers
    def _public(self, i: int) -> dict:
        return {k: v for k, v in self.papers[i].items() if not k.startswith("_")}

    def _idx(self, paper_id: str) -> int:
        pid = paper_id.split(":", 1)[-1] if paper_id.startswith(("DEMO:", "CorpusId:")) else paper_id
        if pid in self._by_id:
            return self._by_id[pid]
        raise KeyError(paper_id)

    def default_focus(self) -> str:
        best = max(range(len(self.papers)),
                   key=lambda i: (2019 <= self.papers[i]["year"] <= 2022) * min(len(self.refs[i]), 1.5 * len(self.cits[i])))
        return _pid(best)


class DemoS2Client:
    """Drop-in replacement for S2Client backed by DemoWorld."""

    is_demo = True

    def __init__(self, world: DemoWorld | None = None):
        self.w = world or DemoWorld()

    def paper(self, paper_id: str, fields=None) -> dict:
        try:
            return self.w._public(self.w._idx(paper_id))
        except KeyError:
            from .s2_client import S2Error
            raise S2Error(f"Not found in demo world: {paper_id}")

    def search_title(self, title: str) -> dict | None:
        res = self.search(title, 1)
        return self.paper(res[0]["paperId"]) if res else None

    def search(self, query: str, limit: int = 10) -> list[dict]:
        q = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for i, p in enumerate(self.w.papers):
            s = len(q & set(re.findall(r"[a-z]+", p["title"].lower())))
            if s:
                scored.append((s, p["citationCount"], i))
        scored.sort(reverse=True)
        return [self.w._public(i) for _, _, i in scored[:limit]]

    def _edges(self, paper_id, kind, max_items):
        i = self.w._idx(paper_id)
        src = self.w.refs[i] if kind == "references" else self.w.cits[i]
        return [{"paper": self.w._public(e["_j"]), **{k: v for k, v in e.items() if k != "_j"}}
                for e in src[:max_items]]

    def references(self, paper_id: str, max_items: int = 500):
        return self._edges(paper_id, "references", max_items)

    def citations(self, paper_id: str, max_items: int = 400):
        return self._edges(paper_id, "citations", max_items)

    def embeddings(self, ids):
        out = {}
        for pid in ids:
            if pid in self.w._by_id:
                out[pid] = self.w.emb[self.w._by_id[pid]].tolist()
        return out


class DemoGitHub:
    def repo_info(self, url, source="paper"):
        return {"url": url, "name": url.split("github.com/")[-1], "stars": None, "source": source}

    def search_by_arxiv(self, arxiv_id, n=3):
        return []


def demo_repos_for(paper: dict) -> list[dict]:
    """Fake but clearly-labelled repos so the Code panel has something to render in demo mode."""
    h = int(hashlib.md5(paper["paperId"].encode()).hexdigest(), 16)
    if h % 3:
        return []
    slug = re.sub(r"[^a-z0-9]+", "-", paper["title"].lower()).strip("-")[:40]
    return [{"url": None, "name": f"demo-lab/{slug}", "description": "Synthetic demo repository",
             "stars": h % 4000, "language": "Python", "source": "demo"}]
