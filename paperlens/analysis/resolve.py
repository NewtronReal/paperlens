"""Turn whatever the user typed (arXiv id/URL, DOI, S2 URL, title) into a Semantic Scholar paper id."""
from __future__ import annotations

import re

from ..sources.arxiv_client import ARXIV_ID_RE, OLD_ARXIV_ID_RE

S2_HEX = re.compile(r"\b([0-9a-f]{40})\b")
DOI = re.compile(r"(10\.\d{4,9}/\S+)")


def to_identifier(query: str) -> tuple[str | None, str]:
    """Returns (identifier or None, kind). None means 'treat as a title search'."""
    q = query.strip()
    if not q:
        return None, "empty"
    if re.fullmatch(r"demo\d{4}", q):
        return q, "demo"
    if "arxiv" in q.lower() or ARXIV_ID_RE.fullmatch(re.sub(r"v\d+$", "", q)):
        m = ARXIV_ID_RE.search(q) or OLD_ARXIV_ID_RE.search(q)
        if m:
            return f"arXiv:{m.group(1)}", "arxiv"
    m = S2_HEX.search(q)
    if m and ("semanticscholar" in q or len(q) == 40):
        return m.group(1), "s2"
    m = DOI.search(q)
    if m:
        return f"DOI:{m.group(1).rstrip('.')}", "doi"
    if q.lower().startswith("corpusid:"):
        return q, "corpus"
    return None, "title"
