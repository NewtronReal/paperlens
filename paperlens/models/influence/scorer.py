"""Inference: score citation edges of a local graph built around the paper being analysed."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ... import config
from .features import INTENT_COLS, NODE_DIM


class HeuristicInfluence:
    """Used only until a GNN is trained. Transparent formula, labelled as such in the UI."""
    name = "heuristic (untrained): mentions + method-intent"
    threshold = 0.5

    def score(self, x, edge_index, src, dst, eattr) -> np.ndarray:
        e = np.asarray(eattr)
        log_n = e[:, 6] * 3.0  # = log1p(#mentions)
        z = -2.0 + 1.2 * log_n + 2.0 * e[:, 1] + 0.8 * e[:, 4]
        return 1 / (1 + np.exp(-z))


class LearnedInfluence:
    def __init__(self, path: Path):
        import torch
        from .model import InfluenceModel
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.model = InfluenceModel(**ck["cfg"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.threshold = ck.get("threshold", 0.5)
        kind = "GraphSAGE" if ck["cfg"].get("use_graph") else "MLP (no message passing)"
        self.name = f"{kind} influence model '{ck.get('variant', '?')}' (intent feats: {ck.get('intent_model', '?')})"
        self.uses_graph = bool(ck["cfg"].get("use_graph"))
        self.use_intent, self.use_emb = ck.get("use_intent", True), ck.get("use_emb", True)

    def score(self, x, edge_index, src, dst, eattr) -> np.ndarray:
        t = self.torch
        x, eattr = np.array(x, dtype=np.float32), np.array(eattr, dtype=np.float32)
        if not self.use_intent:   # replicate the ablation the checkpoint was trained with
            eattr[:, INTENT_COLS] = 0
        if not self.use_emb:
            x[:, :NODE_DIM - 3] = 0
        with t.no_grad():
            logit = self.model(t.as_tensor(x, dtype=t.float32), t.as_tensor(edge_index, dtype=t.long),
                               t.as_tensor(src, dtype=t.long), t.as_tensor(dst, dtype=t.long),
                               t.as_tensor(eattr, dtype=t.float32))
        return t.sigmoid(logit).numpy()


@lru_cache(maxsize=1)
def load_influence_scorer(ckpt_dir: str | None = None):
    p = Path(ckpt_dir or config.CKPT_DIR) / "influence_model.pt"
    if p.exists():
        try:
            return LearnedInfluence(p)
        except Exception as e:  # noqa: BLE001
            print(f"[influence] failed to load model ({e}); using heuristic")
    return HeuristicInfluence()
