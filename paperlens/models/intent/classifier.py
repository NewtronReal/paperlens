"""Inference wrapper for the citation-intent model.

Loads, in order of preference:
  checkpoints/intent_scibert/   (fine-tuned SciBERT, train_scibert.py)
  checkpoints/intent_tfidf.joblib (TF-IDF + logistic regression baseline, train_baseline.py)
  a keyword heuristic (only so the app runs before anything is trained; flagged in the UI)
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ... import config
from .data import LABELS, clean_context

_METHOD = r"\b(we (use|adopt|follow|employ|build on|apply|extend)|following|based on|implementation|" \
          r"initiali[sz]ed|pre-?trained|backbone|as in|we borrow|architecture (of|from|introduced))\b"
_RESULT = r"\b(outperform\w*|compared (to|with)|consistent with|in contrast|similar to|improv\w+ (over|on)|" \
          r"agree\w* with|baseline|than|surpass\w*|state-of-the-art results)\b"


class HeuristicIntent:
    name = "keyword-heuristic (untrained)"

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 3))
        for i, t in enumerate(texts):
            t = t.lower()
            s = np.array([1.0, 2.0 * len(re.findall(_METHOD, t)), 2.0 * len(re.findall(_RESULT, t))])
            e = np.exp(s - s.max())
            out[i] = e / e.sum()
        return out


class TfidfIntent:
    def __init__(self, path: Path):
        import joblib
        self.pipe = joblib.load(path)
        self.name = "tfidf-logreg (SciCite)"

    def predict_proba(self, texts):
        return self.pipe.predict_proba([clean_context(t) for t in texts])


class SciBertIntent:
    def __init__(self, path: Path, device: str | None = None, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.bs = batch_size
        self.name = "SciBERT fine-tuned (SciCite)"

    def predict_proba(self, texts):
        if not texts:
            return np.zeros((0, 3))
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.bs):
                batch = [clean_context(t) for t in texts[i:i + self.bs]]
                enc = self.tok(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
                outs.append(self.torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(outs)


@lru_cache(maxsize=1)
def load_intent_classifier(ckpt_dir: str | None = None):
    ckpt = Path(ckpt_dir) if ckpt_dir else config.CKPT_DIR
    if (ckpt / "intent_scibert" / "config.json").exists():
        try:
            return SciBertIntent(ckpt / "intent_scibert")
        except Exception as e:  # noqa: BLE001 - torch missing etc.
            print(f"[intent] SciBERT checkpoint present but failed to load: {e}")
    if (ckpt / "intent_tfidf.joblib").exists():
        return TfidfIntent(ckpt / "intent_tfidf.joblib")
    return HeuristicIntent()


def label_of(probs: np.ndarray) -> str:
    return LABELS[int(np.argmax(probs))]
