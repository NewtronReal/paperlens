"""Train/serve gap check (DESIGN.md §4): evaluate the deployed influence checkpoint on held-out star graphs.

Training and train.py's evaluation pass messages over the dense graph of all collected edges. The app instead
scores one paper at a time on a star graph (the paper + its references only). This script scores every
test-split citing paper on its own star graph, exactly as pipeline.py does, and compares with the dense-graph
score of the same checkpoint on the same edges.

    python -m paperlens.models.influence.eval_star --data data/influence

Writes results/<tag>.{md,json}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ... import config
from ..intent.classifier import load_intent_classifier
from .dataset import load_dataset
from .scorer import LearnedInfluence
from .train import best_threshold, scores


def star_scores(scorer, d, edge_ids: np.ndarray) -> np.ndarray:
    """Score each citing paper's edges on a star graph built from that paper and its references only."""
    out = np.zeros(len(edge_ids), dtype=np.float32)
    srcs = d["src"][edge_ids]
    for s in np.unique(srcs):
        pos = np.where(srcs == s)[0]
        idx = edge_ids[pos]
        nodes = np.concatenate([[s], d["dst"][idx]])
        others = np.arange(1, len(nodes))
        lsrc, ldst = np.zeros_like(others), others
        ei = np.concatenate([np.stack([lsrc, ldst]), np.stack([ldst, lsrc])], 1)
        out[pos] = scorer.score(d["x"][nodes], ei, lsrc, ldst, d["eattr"][idx])
    return out


def dense_scores(scorer, d, edge_ids: np.ndarray) -> np.ndarray:
    """Score the same edges with message passing over the full collected graph (as train.py evaluates)."""
    ei = np.concatenate([np.stack([d["src"], d["dst"]]), np.stack([d["dst"], d["src"]])], 1)
    return scorer.score(d["x"], ei, d["src"][edge_ids], d["dst"][edge_ids], d["eattr"][edge_ids])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR / "influence"))
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "influence_model.pt"))
    ap.add_argument("--tag", default="influence_star_eval")
    args = ap.parse_args(argv)

    scorer = LearnedInfluence(args.ckpt)
    d = load_dataset(args.data, load_intent_classifier())
    val, test = np.where(d["val"])[0], np.where(d["test"])[0]
    y = d["y"]

    res = {"model": scorer.name, "n_test_edges": int(len(test)), "n_test_papers": int(len(np.unique(d["src"][test])))}
    for mode, fn in (("dense", dense_scores), ("star", star_scores)):
        pv, pt = fn(scorer, d, val), fn(scorer, d, test)
        thr = best_threshold(y[val], pv)  # threshold picked on val in the same graph mode
        res[mode] = {"test": scores(y[test], pt, thr), "threshold": thr,
                     "test_at_ckpt_threshold": scores(y[test], pt, scorer.threshold)}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    lines = [f"# Star-graph evaluation ({args.tag})", "",
             f"Checkpoint: {scorer.name}. Test split: {res['n_test_edges']} edges from "
             f"{res['n_test_papers']} citing papers.", "",
             "| Graph at inference | Test AUROC | Test AUPRC | Test F1 (val thr) | Test F1 (ckpt thr) |",
             "|---|---|---|---|---|"]
    for mode in ("dense", "star"):
        t, c = res[mode]["test"], res[mode]["test_at_ckpt_threshold"]
        lines.append(f"| {mode} | {t['auroc']:.4f} | {t['auprc']:.4f} | {t['f1']:.4f} | {c['f1']:.4f} |")
    (config.RESULTS_DIR / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
