"""One-glance summary.

Default is extractive (no extra model): S2's TL;DR + abstract sentences routed by cue phrases.
If LLM_BASE_URL/LLM_MODEL are set, an OpenAI-compatible chat endpoint (e.g. a local Ollama model)
writes a structured summary instead. The LLM is a convenience layer, not one of the project's trained models.
"""
from __future__ import annotations

import json
import re

import httpx

from .. import config

_APPROACH = re.compile(r"\b(we (propose|introduce|present|develop|design)|our (method|approach|model|framework))\b", re.I)
_RESULTS = re.compile(r"\b(outperform|achiev|improv|state-of-the-art|results (show|demonstrate)|\d+(\.\d+)?\s?%)", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text or "") if len(s.strip()) > 20]


def extractive_summary(paper: dict) -> dict:
    sents = _sentences(paper.get("abstract") or "")
    tldr = (paper.get("tldr") or {}).get("text") if isinstance(paper.get("tldr"), dict) else None
    approach = [s for s in sents if _APPROACH.search(s)][:2]
    results = [s for s in sents if _RESULTS.search(s) and s not in approach][:2]
    problem = [s for s in sents[:2] if s not in approach and s not in results][:1]
    return {
        "source": "extractive (S2 TL;DR + abstract cue phrases)",
        "tldr": tldr or (sents[0] if sents else None),
        "problem": " ".join(problem) or None,
        "approach": " ".join(approach) or None,
        "results": " ".join(results) or None,
        "limitations": None,
    }


def llm_summary(paper: dict, top_refs: list[dict], body: str | None = None) -> dict | None:
    if not (config.LLM_BASE_URL and config.LLM_MODEL):
        return None
    refs = "\n".join(f"- {r['title']} ({r.get('year')}), cited for {r['intent']}" for r in top_refs[:6])
    prompt = (
        "Summarise this research paper for a researcher who wants to grasp it at a glance. "
        "Return ONLY a JSON object with keys: tldr (1 sentence), problem, approach, results, limitations, "
        "read_if (who should read it). Each value at most 2 sentences. Do not invent numbers.\n\n"
        f"Title: {paper.get('title')}\nAbstract: {paper.get('abstract')}\n"
        f"Most influential references (from our citation model):\n{refs}\n"
        + (f"\nOpening of the full text:\n{body[:4000]}" if body else "")
    )
    try:
        r = httpx.post(f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                       headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                       json={"model": config.LLM_MODEL, "temperature": 0.2,
                             "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        data["source"] = f"LLM ({config.LLM_MODEL})"
        return data
    except Exception as e:  # noqa: BLE001
        print("[summary] LLM failed, falling back to extractive:", e)
        return None
