"""Baseline citation-intent classifier: TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression.

    python -m paperlens.models.intent.train_baseline [--data-dir data/scicite]
"""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from ... import config
from .data import clean_context, load_scicite
from .metrics import evaluate, save_report


def build_pipeline(C: float = 4.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("tfidf", feats), ("clf", clf)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--C", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    args = ap.parse_args(argv)

    data = load_scicite(args.data_dir)
    (Xtr, ytr, _), (Xdv, ydv, _), (Xte, yte, _) = data["train"], data["dev"], data["test"]
    Xtr, Xdv, Xte = ([clean_context(t) for t in X] for X in (Xtr, Xdv, Xte))
    print(f"train {len(Xtr)}  dev {len(Xdv)}  test {len(Xte)}")

    best, best_f1 = None, -1.0
    for C in args.C:  # model selection on dev only
        t = time.time()
        pipe = build_pipeline(C).fit(Xtr, ytr)
        f1 = evaluate(ydv, pipe.predict(Xdv))["macro_f1"]
        print(f"C={C:<5} dev macro-F1 {f1:.4f}  ({time.time() - t:.1f}s)")
        if f1 > best_f1:
            best, best_f1, best_C = pipe, f1, C

    metrics = {"model": "tfidf-logreg", "C": best_C,
               "dev": evaluate(ydv, best.predict(Xdv)), "test": evaluate(yte, best.predict(Xte))}
    print(f"TEST macro-F1 {metrics['test']['macro_f1']:.4f}  acc {metrics['test']['accuracy']:.4f}")
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, config.CKPT_DIR / "intent_tfidf.joblib")
    save_report("intent_tfidf", metrics, config.RESULTS_DIR)


if __name__ == "__main__":
    main()
