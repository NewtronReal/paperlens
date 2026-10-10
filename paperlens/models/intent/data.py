"""SciCite loading (Cohan et al., NAACL 2019: 'Structural Scaffolds for Citation Intent Classification').

~11k citation sentences labelled background / method / result.
Sources tried in order:
  1. a local directory with train.jsonl / dev.jsonl / test.jsonl from the original release
     (scripts/download_scicite.sh fetches it)
  2. the Hugging Face hub copy  `allenai/scicite` (parquet conversion branch as fallback)
"""
from __future__ import annotations

import json
from pathlib import Path

from ... import config

LABELS = config.INTENT_LABELS
_ALIASES = {"methodology": "method", "uses": "method", "comparison": "result", "result_comparison": "result"}


def _norm(label) -> int:
    lab = str(label).strip().lower()
    lab = _ALIASES.get(lab, lab)
    return LABELS.index(lab)


def _read_jsonl(path: Path) -> tuple[list[str], list[int], list[str]]:
    texts, labels, sections = [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["string"].strip())
            labels.append(_norm(d["label"]))
            sections.append(str(d.get("sectionName") or ""))
    return texts, labels, sections


def load_scicite(local_dir: str | Path | None = None) -> dict[str, tuple[list[str], list[int], list[str]]]:
    local_dir = Path(local_dir) if local_dir else config.DATA_DIR / "scicite"
    if (local_dir / "train.jsonl").exists():
        return {s: _read_jsonl(local_dir / f"{s}.jsonl") for s in ("train", "dev", "test")}

    from datasets import load_dataset  # lazy: heavy import

    last = None
    for kw in ({}, {"revision": "refs/convert/parquet"}):
        try:
            ds = load_dataset("allenai/scicite", **kw)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise RuntimeError(
            f"Could not load SciCite from the hub ({last}). Run scripts/download_scicite.sh "
            f"and pass --data-dir data/scicite"
        )
    names = ds["train"].features["label"].names if hasattr(ds["train"].features["label"], "names") else None

    def conv(split):
        d = ds[split]
        labs = [_norm(names[l] if names else l) for l in d["label"]]
        secs = d["sectionName"] if "sectionName" in d.column_names else [""] * len(d)
        return [t.strip() for t in d["string"]], labs, [str(s or "") for s in secs]

    return {"train": conv("train"), "dev": conv("validation"), "test": conv("test")}


def clean_context(text: str) -> str:
    """Normalise citation markers so train (SciCite) and inference (S2/PDF) text look alike."""
    import re
    t = re.sub(r"\[[\d,\s–\-]+\]", "[CIT]", text)
    t = re.sub(r"\(([A-Z][A-Za-z\-]+( et al\.?)?( and [A-Z][A-Za-z\-]+)?,? \d{4}[a-z]?(; ?)?)+\)", "[CIT]", t)
    return re.sub(r"\s+", " ", t).strip()
