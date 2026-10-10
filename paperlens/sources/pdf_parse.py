"""Lightweight PDF parsing with PyMuPDF.

Extracts: probable title, the paper's own arXiv ID / DOI, GitHub links, body sentences, the numbered
reference list, and citation contexts (sentences that cite a given reference). Contexts feed the
citation-intent model when Semantic Scholar has none for an edge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .arxiv_client import ARXIV_ID_RE
from .repos import extract_github_urls

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;)])")
REF_HEADING_RE = re.compile(r"\n\s*(?:\d+\.?\s*)?(References|REFERENCES|Bibliography|BIBLIOGRAPHY|Literature Cited)\s*\n")
NUM_ENTRY_RE = re.compile(r"\n\s*\[(\d{1,3})\]\s+")
NUM_CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*)\]")
STOP = set("a an the of for and in on with to via from by is are we our at as using towards toward".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


@dataclass
class ParsedPDF:
    title: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    github: list[str] = field(default_factory=list)
    body: str = ""
    sentences: list[str] = field(default_factory=list)
    numbered_refs: dict[int, str] = field(default_factory=dict)
    n_pages: int = 0

    def contexts_for(self, title: str | None, first_author: str | None, year: int | None,
                     max_ctx: int = 6) -> list[str]:
        """Sentences citing the reference identified by (title, first author surname, year)."""
        found: list[str] = []
        # 1) numeric style: match the reference entry by title overlap, then find "[n]" markers
        if title and self.numbered_refs:
            tt = _tokens(title)
            best, best_score = None, 0.0
            for n, entry in self.numbered_refs.items():
                if not tt:
                    break
                score = len(tt & _tokens(entry)) / len(tt)
                if score > best_score:
                    best, best_score = n, score
            if best is not None and best_score >= 0.7:
                for s in self.sentences:
                    for grp in NUM_CITE_RE.findall(s):
                        if best in _expand(grp):
                            found.append(s)
                            break
        # 2) author-year style: "Vaswani et al., 2017" / "Vaswani et al. (2017)"
        if not found and first_author and year:
            pat = re.compile(rf"\b{re.escape(first_author)}\b.{{0,40}}?\b{year}\b")
            found = [s for s in self.sentences if pat.search(s)]
        return found[:max_ctx]


def _expand(group: str) -> set[int]:
    nums: set[int] = set()
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[–\-]", part):
            a, b = re.split(r"\s*[–\-]\s*", part)[:2]
            if a.isdigit() and b.isdigit() and int(b) - int(a) < 50:
                nums.update(range(int(a), int(b) + 1))
        elif part.strip().isdigit():
            nums.add(int(part))
    return nums


def _guess_title(page) -> str | None:
    best_size, spans = 0.0, []
    d = page.get_text("dict")
    height = page.rect.height
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for sp in line.get("spans", []):
                txt = sp.get("text", "").strip()
                if len(txt) < 3 or sp["bbox"][1] > height * 0.45 or "arXiv" in txt:
                    continue
                spans.append((sp["size"], sp["bbox"][1], txt))
                best_size = max(best_size, sp["size"])
    parts = [t for s, y, t in sorted(spans, key=lambda x: x[1]) if abs(s - best_size) < 0.6]
    title = " ".join(parts).strip()
    return title if 10 <= len(title) <= 300 else None


def _clean_body(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)        # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text)


def _split_sentences(text: str) -> list[str]:
    # protect common abbreviations before splitting
    t = re.sub(r"\b(et al|e\.g|i\.e|Fig|Eq|Sec|vs|cf)\.", lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", t)
    return [p.replace("<DOT>", ".").strip() for p in parts if 25 <= len(p) <= 800]


def parse_pdf(data: bytes) -> ParsedPDF:
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    pages = [p.get_text() for p in doc]
    full = "\n".join(pages)
    out = ParsedPDF(n_pages=len(pages))
    if not pages:
        return out
    out.title = _guess_title(doc[0])
    first = pages[0]
    m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", first) or ARXIV_ID_RE.search(first[:3000])
    out.arxiv_id = m.group(1) if m else None
    m = DOI_RE.search(first)
    out.doi = m.group(1) if m else None

    heads = list(REF_HEADING_RE.finditer(full))
    split_at = heads[-1].start() if heads else len(full)
    body, refs = full[:split_at], full[split_at:]
    out.github = extract_github_urls(body)
    out.body = _clean_body(body)
    out.sentences = _split_sentences(out.body)

    if refs:
        pieces = NUM_ENTRY_RE.split("\n" + refs)
        # pieces = [preamble, n1, text1, n2, text2, ...]
        for i in range(1, len(pieces) - 1, 2):
            out.numbered_refs[int(pieces[i])] = _clean_body(pieces[i + 1])[:500]
    return out
