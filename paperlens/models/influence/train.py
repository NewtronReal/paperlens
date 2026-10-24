"""Train / evaluate the citation-influence model with baselines and ablations.

    python -m paperlens.models.influence.train --data data/influence              # full ablation table
    python -m paperlens.models.influence.train --data data/influence_demo --seeds 1 --epochs 60   # smoke test

Writes results/<tag>_results.{md,json} and checkpoints/influence_model.pt: the single run with the best
validation AUPRC across all variants (model selection on val only; the test split is never used to choose).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ... import config
from ..intent.classifier import load_intent_classifier
from .dataset import load_dataset
from .features import EDGE_FEATURE_NAMES, INTENT_COLS, NODE_DIM
from .model import InfluenceModel

VARIANTS = {
    #                use_graph use_edge  intent  node_emb
    "gnn":           (True,    True,     True,   True),
    "gnn-no-intent": (True,    True,     False,  True),
    "gnn-no-emb":    (True,    True,     True,   False),
    "gnn-no-emb-no-intent": (True, True, False,  False),
    "mlp":           (False,   True,     True,   True),
    "mlp-no-intent": (False,   True,     False,  True),
    "mlp-no-emb":    (False,   True,     True,   False),
    "mlp-no-emb-no-intent": (False, True, False, False),
}


def scores(y, p, thr=None):
    out = {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}
    if thr is not None:
        out["f1"] = float(f1_score(y, p >= thr))
    return out


def best_threshold(y, p):
    grid = np.quantile(p, np.linspace(0.05, 0.95, 91))
    f1s = [f1_score(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def to_t(d, device):
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    x = t(d["x"])
    src, dst = t(d["src"], torch.long), t(d["dst"], torch.long)
    ei = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1)  # undirected message passing
    return x, ei, src, dst, t(d["eattr"]), t(d["y"])


def run_variant(d, name, seed, args, device):
    use_graph, use_edge, use_intent, use_emb = VARIANTS[name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, ei, src, dst, ea, y = to_t(d, device)
    if not use_intent:
        ea = ea.clone()
        ea[:, INTENT_COLS] = 0
    if not use_emb:
        x = x.clone()
        x[:, :NODE_DIM - 3] = 0
    m = {k: torch.as_tensor(d[k], device=device) for k in ("train", "val", "test")}
    model = InfluenceModel(x.shape[1], ea.shape[1], args.hidden, args.layers, args.dropout,
                           use_graph, use_edge).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    pos = y[m["train"]].mean().clamp(1e-3, 1 - 1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        e_mp = ei
        if use_graph and args.drop_edge > 0:  # DropEdge: robustness to the sparse local graphs seen at inference
            keep = torch.rand(ei.shape[1], device=device) > args.drop_edge
            e_mp = ei[:, keep]
        logit = model(x, e_mp, src[m["train"]], dst[m["train"]], ea[m["train"]])
        loss = loss_fn(logit, y[m["train"]])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(x, ei, src[m["val"]], dst[m["val"]], ea[m["val"]])).cpu().numpy()
        ap = average_precision_score(d["y"][d["val"]], pv)
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x, ei, src, dst, ea)).cpu().numpy()
    thr = best_threshold(d["y"][d["val"]], p[d["val"]])
    res = {"val": scores(d["y"][d["val"]], p[d["val"]], thr), "test": scores(d["y"][d["test"]], p[d["test"]], thr),
           "threshold": thr, "epochs": ep + 1}
    return res, model


def classic_baselines(d):
    out = {}
    te, va, tr = d["test"], d["val"], d["train"]
    for name, s in (("heuristic: #mentions", d["n_contexts"]), ("heuristic: cited-paper citations",
                                                                 np.log1p(d["dst_citations"]))):
        thr = best_threshold(d["y"][va], s[va])
        out[name] = {"val": scores(d["y"][va], s[va], thr), "test": scores(d["y"][te], s[te], thr)}
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(d["eattr"][tr], d["y"][tr])
    p = lr.predict_proba(d["eattr"])[:, 1]
    thr = best_threshold(d["y"][va], p[va])
    out["logreg: edge features"] = {"val": scores(d["y"][va], p[va], thr), "test": scores(d["y"][te], p[te], thr),
                                    "coef": dict(zip(EDGE_FEATURE_NAMES, lr.coef_[0].round(3).tolist()))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR / "influence"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--drop-edge", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tag", default="influence")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = load_intent_classifier()
    d = load_dataset(args.data, clf)
    n = {k: int(d[k].sum()) for k in ("train", "val", "test")}
    print(f"nodes {d['n_nodes']}  labelled edges {len(d['y'])} {n}  positive rate {d['y'].mean():.3f}  "
          f"intent model: {d['intent_model']}")

    results = {"meta": {"n_nodes": d["n_nodes"], "n_edges": int(len(d["y"])), "splits": n,
                        "pos_rate": float(d["y"].mean()), "intent_model": d["intent_model"], "args": vars(args)},
               "baselines": classic_baselines(d), "variants": {}}
    best = (-1.0, None, None, None)
    for v in args.variants:
        runs = []
        for s in range(args.seeds):
            t = time.time()
            res, model = run_variant(d, v, s, args, device)
            runs.append(res)
            print(f"{v:<14} seed {s}: test AUROC {res['test']['auroc']:.4f}  AUPRC {res['test']['auprc']:.4f}  "
                  f"F1 {res['test']['f1']:.4f}  ({res['epochs']} ep, {time.time() - t:.0f}s)")
            if res["val"]["auprc"] > best[0]:
                best = (res["val"]["auprc"], model, res["threshold"], v)
        agg = {}
        for split in ("val", "test"):
            for k in ("auroc", "auprc", "f1"):
                vals = [r[split][k] for r in runs]
                agg[f"{split}_{k}"] = [float(np.mean(vals)), float(np.std(vals))]
        results["variants"][v] = {"runs": runs, "agg": agg}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"{args.tag}_results.json").write_text(json.dumps(results, indent=2))
    lines = [f"# Citation-influence results ({args.tag})", "",
             f"Nodes {d['n_nodes']}, labelled edges {len(d['y'])}, positive rate {d['y'].mean():.3f}, "
             f"split by citing paper {n}. Intent features from: {d['intent_model']}.", "",
             "| Model | Test AUROC | Test AUPRC | Test F1 |", "|---|---|---|---|"]
    for name, r in results["baselines"].items():
        t = r["test"]
        lines.append(f"| {name} | {t['auroc']:.4f} | {t['auprc']:.4f} | {t['f1']:.4f} |")
    for name, r in results["variants"].items():
        a = r["agg"]
        cell = lambda k: f"{a[k][0]:.4f} ± {a[k][1]:.4f}"  # noqa: E731
        lines.append(f"| **{name}** | {cell('test_auroc')} | {cell('test_auprc')} | {cell('test_f1')} |")
    (config.RESULTS_DIR / f"{args.tag}_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if best[1] is not None:
        config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best[1].state_dict(), "cfg": best[1].cfg, "threshold": best[2],
                    "variant": best[3], "use_intent": VARIANTS[best[3]][2], "use_emb": VARIANTS[best[3]][3],
                    "val_auprc": best[0], "intent_model": d["intent_model"]},
                   config.CKPT_DIR / "influence_model.pt")
        print(f"saved {config.CKPT_DIR / 'influence_model.pt'}  (variant '{best[3]}', val AUPRC {best[0]:.4f})")


if __name__ == "__main__":
    main()
