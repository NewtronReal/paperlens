"""Evaluate influence models on the ambiguous edges only (1-3 citation sentences).

In the collected data every edge with 4+ mentions is labelled influential and every edge with no citation
sentence is not, so the full-test metrics are dominated by a mention-count threshold. This script retrains
the chosen variants and scores them on test edges with 1-3 mentions, where the label is genuinely uncertain.

    python -m paperlens.models.influence.eval_subset

Writes results/<tag>.{md,json}.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ... import config
from ..intent.classifier import load_intent_classifier
from .dataset import load_dataset
from .features import INTENT_COLS
from .train import run_variant, to_t

VARIANTS = ["mlp-no-emb", "mlp-no-emb-no-intent", "gnn-no-emb", "gnn-no-emb-no-intent"]


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR / "influence"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="influence_ambiguous")
    args = ap.parse_args(argv)
    hp = SimpleNamespace(epochs=1500, patience=100, hidden=256, layers=2, dropout=0.3, drop_edge=0.3, lr=2e-3, wd=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_dataset(args.data, load_intent_classifier())
    sub = d["test"] & (d["n_contexts"] >= 1) & (d["n_contexts"] <= 3)
    y = d["y"][sub]
    res = {"n_edges": int(sub.sum()), "pos_rate": float(y.mean()), "models": {}}
    print(f"ambiguous test edges: {res['n_edges']}  positive rate {res['pos_rate']:.3f}")

    res["models"]["#mentions"] = metrics(y, d["n_contexts"][sub])
    for name, cols in (("logreg", slice(None)), ("logreg-no-intent", None)):
        X = d["eattr"].copy()
        if cols is None:
            X[:, INTENT_COLS] = 0
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[d["train"]], d["y"][d["train"]])
        res["models"][name] = metrics(y, lr.predict_proba(X[sub])[:, 1])

    x, ei, src, dst, ea, _ = to_t(d, device)
    for v in VARIANTS:
        runs = []
        for s in range(args.seeds):
            _, model = run_variant(d, v, s, hp, device)
            eav, xv = ea.clone(), x.clone()
            if "no-intent" in v:
                eav[:, INTENT_COLS] = 0
            if "no-emb" in v:
                xv[:, :-3] = 0
            with torch.no_grad():
                p = torch.sigmoid(model(xv, ei, src, dst, eav)).cpu().numpy()
            runs.append(metrics(y, p[sub]))
        res["models"][v] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                            for k in ("auroc", "auprc")}
        print(v, res["models"][v])

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    lines = [f"# Ambiguous edges (1-3 mentions), test split: {res['n_edges']} edges, "
             f"positive rate {res['pos_rate']:.3f}", "", "| Model | AUROC | AUPRC |", "|---|---|---|"]
    for k, m in res["models"].items():
        if isinstance(m["auroc"], list):
            lines.append(f"| {k} | {m['auroc'][0]:.4f} ± {m['auroc'][1]:.4f} | {m['auprc'][0]:.4f} ± {m['auprc'][1]:.4f} |")
        else:
            lines.append(f"| {k} | {m['auroc']:.4f} | {m['auprc']:.4f} |")
    (config.RESULTS_DIR / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
