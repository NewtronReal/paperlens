"""Idea-evolution map: cluster the focus paper's ancestors and descendants in embedding space, order the
clusters in time and label each with class-based TF-IDF keywords (c-TF-IDF, as in BERTopic)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

EXTRA_STOP = {"paper", "propose", "proposed", "method", "methods", "approach", "results", "model", "models",
              "based", "using", "show", "new", "work", "task", "tasks", "performance", "state", "art",
              "demonstrate", "novel", "learning", "existing", "experiments", "achieve", "achieves", "release",
              "code", "simpler", "strong", "baselines", "remains", "central", "problem", "standard",
              "benchmarks", "improves", "train", "pretrained", "combines"}


def _doc(p: dict) -> str:
    return f"{p.get('title') or ''}. {p.get('abstract') or ''}"


def build_evolution(papers: list[dict], embeddings: dict[str, list[float]], max_k: int = 6, seed: int = 0) -> dict:
    """papers: dicts with paperId, title, abstract, year, citationCount, role, influence."""
    papers = [p for p in papers if p.get("year")]
    if len(papers) < 6:
        return {"clusters": [], "nodes": [], "storyline": []}
    ids = [p["paperId"] for p in papers]
    have = [i for i in ids if i in embeddings]
    if len(have) >= 0.7 * len(ids):
        dim = len(embeddings[have[0]])
        X = np.array([embeddings.get(i, np.zeros(dim)) for i in ids], dtype=np.float32)
        source = "SPECTER2 embeddings"
    else:  # fall back to text when embeddings are missing
        tf = TfidfVectorizer(stop_words="english", min_df=1, max_features=5000).fit_transform([_doc(p) for p in papers])
        X = TruncatedSVD(n_components=min(48, tf.shape[1] - 1, len(papers) - 1), random_state=seed).fit_transform(tf)
        source = "TF-IDF + SVD (embeddings unavailable)"
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    k = int(min(max_k, max(2, round(np.sqrt(len(papers) / 4)))))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)

    # c-TF-IDF: one 'document' per cluster
    cluster_docs = [" ".join(_doc(p) for p, l in zip(papers, labels) if l == c) for c in range(k)]
    stop = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    M = vec.fit_transform(cluster_docs)
    vocab = np.array(vec.get_feature_names_out())

    clusters = []
    for c in range(k):
        members = [p for p, l in zip(papers, labels) if l == c]
        row = M[c].toarray().ravel()
        terms = []
        for t in vocab[np.argsort(-row)]:
            if not any(t in s or s in t for s in terms):
                terms.append(t)
            if len(terms) == 3:
                break
        years = [p["year"] for p in members]
        key = max(members, key=lambda p: (p.get("influence") or 0) * 3 + np.log1p(p.get("citationCount") or 0))
        clusters.append({
            "id": c, "keywords": terms, "label": " · ".join(terms), "size": len(members),
            "year_min": int(min(years)), "year_max": int(max(years)), "year_median": float(np.median(years)),
            "n_ancestors": sum(p["role"] == "ancestor" for p in members),
            "n_descendants": sum(p["role"] == "descendant" for p in members),
            "has_focus": any(p["role"] == "focus" for p in members),
            "key_paper": {"paperId": key["paperId"], "title": key.get("title"), "year": key.get("year")},
            "per_year": {int(y): years.count(y) for y in sorted(set(years))},
        })
    order = sorted(range(k), key=lambda c: clusters[c]["year_median"])
    remap = {old: new for new, old in enumerate(order)}
    clusters = [dict(clusters[o], id=remap[o]) for o in order]
    nodes = [{"paperId": p["paperId"], "title": p.get("title"), "year": p["year"], "cluster": remap[int(l)],
              "role": p["role"], "citationCount": p.get("citationCount") or 0, "influence": p.get("influence")}
             for p, l in zip(papers, labels)]
    storyline = [{"span": f"{c['year_min']}–{c['year_max']}", "label": c["label"], "focus": c["has_focus"],
                  "key_paper": c["key_paper"]["title"]} for c in clusters]
    return {"clusters": clusters, "nodes": nodes, "storyline": storyline, "source": source}
