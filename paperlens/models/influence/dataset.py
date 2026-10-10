"""Turn collected nodes/edges into tensors for the influence GNN.

Split is *by citing paper*: all reference edges of one citing paper land in the same split. A random
edge split would leak (the model would see sibling edges of the same paper during training).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .features import EDGE_DIM, NODE_DIM, edge_features, node_features


def _read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def featurize_edges(edges: list[dict], intent_clf, years: dict[str, int | None]) -> np.ndarray:
    flat, owner = [], []
    for k, e in enumerate(edges):
        for c in e["contexts"]:
            flat.append(c)
            owner.append(k)
    probs = intent_clf.predict_proba(flat) if flat else np.zeros((0, 3))
    per_edge: list[list[np.ndarray]] = [[] for _ in edges]
    for k, p in zip(owner, probs):
        per_edge[k].append(p)
    out = np.zeros((len(edges), EDGE_DIM), dtype=np.float32)
    for k, e in enumerate(edges):
        P = np.stack(per_edge[k]) if per_edge[k] else None
        out[k] = edge_features(P, len(e["contexts"]), years.get(e["src"]), years.get(e["dst"]))
    return out


def load_dataset(data_dir: str | Path, intent_clf, seed: int = 0, split=(0.7, 0.15, 0.15)) -> dict:
    data_dir = Path(data_dir)
    nodes = _read_jsonl(data_dir / "nodes.jsonl")
    edges = _read_jsonl(data_dir / "edges.jsonl")
    id2i = {n["paperId"]: i for i, n in enumerate(nodes)}
    edges = [e for e in edges if e["src"] in id2i and e["dst"] in id2i]

    emb = {}
    if (data_dir / "embeddings.npz").exists():
        z = np.load(data_dir / "embeddings.npz", allow_pickle=False)
        emb = dict(zip(z["ids"].tolist(), z["vectors"]))
    X = np.stack([node_features(emb.get(n["paperId"]), n.get("citationCount"), n.get("year")) for n in nodes]) \
        if nodes else np.zeros((0, NODE_DIM), np.float32)

    slug = re.sub(r"[^a-z0-9]+", "_", intent_clf.name.lower()).strip("_")
    cache = data_dir / f"edge_feats__{slug}.npy"
    if cache.exists() and len(np.load(cache)) == len(edges):
        E = np.load(cache)
    else:
        print(f"featurizing {len(edges)} edges with intent model '{intent_clf.name}' ...")
        E = featurize_edges(edges, intent_clf, {n["paperId"]: n.get("year") for n in nodes})
        np.save(cache, E)

    src = np.array([id2i[e["src"]] for e in edges], dtype=np.int64)
    dst = np.array([id2i[e["dst"]] for e in edges], dtype=np.int64)
    y = np.array([e["label"] for e in edges], dtype=np.float32)

    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(src))
    n_tr, n_va = int(split[0] * len(uniq)), int(split[1] * len(uniq))
    part = {int(s): 0 for s in uniq[:n_tr]} | {int(s): 1 for s in uniq[n_tr:n_tr + n_va]} | \
           {int(s): 2 for s in uniq[n_tr + n_va:]}
    which = np.array([part[int(s)] for s in src])

    return {
        "x": X, "src": src, "dst": dst, "eattr": E, "y": y,
        "train": which == 0, "val": which == 1, "test": which == 2,
        "dst_citations": np.array([nodes[d].get("citationCount") or 0 for d in dst], dtype=np.float32),
        "n_contexts": np.array([len(e["contexts"]) for e in edges], dtype=np.float32),
        "intent_model": intent_clf.name, "n_nodes": len(nodes),
    }
