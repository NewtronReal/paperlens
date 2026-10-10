"""Feature construction shared by training (dataset.py) and inference (scorer.py).
Keeping this in one place guarantees train/serve consistency."""
from __future__ import annotations

import math

import numpy as np

EMB_DIM = 768
NODE_EXTRA = 3            # log citation count, year, has-embedding flag
NODE_DIM = EMB_DIM + NODE_EXTRA
EDGE_DIM = 9
INTENT_COLS = slice(0, 6)  # columns produced by the citation-intent model (zeroed in the -intent ablation)
EDGE_FEATURE_NAMES = [
    "mean_p_background", "mean_p_method", "mean_p_result",
    "max_p_background", "max_p_method", "max_p_result",
    "log_n_contexts", "has_context", "year_gap",
]


def node_features(emb: list[float] | np.ndarray | None, citation_count: int | None, year: int | None) -> np.ndarray:
    v = np.zeros(NODE_DIM, dtype=np.float32)
    if emb is not None and len(emb) == EMB_DIM:
        e = np.asarray(emb, dtype=np.float32)
        v[:EMB_DIM] = e / (np.linalg.norm(e) + 1e-8)
        v[EMB_DIM + 2] = 1.0
    v[EMB_DIM] = math.log1p(max(citation_count or 0, 0)) / 10.0
    v[EMB_DIM + 1] = ((year or 2015) - 2010) / 15.0
    return v


def edge_features(intent_probs: np.ndarray | None, n_contexts: int,
                  year_src: int | None, year_dst: int | None) -> np.ndarray:
    """intent_probs: (n_contexts, 3) probabilities from the intent model for this edge's contexts."""
    v = np.zeros(EDGE_DIM, dtype=np.float32)
    if intent_probs is not None and len(intent_probs):
        v[0:3] = intent_probs.mean(0)
        v[3:6] = intent_probs.max(0)
    v[6] = math.log1p(n_contexts) / 3.0
    v[7] = float(n_contexts > 0)
    if year_src and year_dst:
        v[8] = min(max(year_src - year_dst, 0), 30) / 30.0
    return v
