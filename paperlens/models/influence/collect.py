"""Build the citation-influence dataset from Semantic Scholar (or the synthetic demo world).

For every 'citing' paper in a snowball sample we store all its references with
  contexts (citation sentences), S2's isInfluential flag (the label), and both endpoints' metadata,
then batch-fetch SPECTER2 embeddings for every node.

    python -m paperlens.models.influence.collect --seeds data/seeds.txt --per-seed-citers 8
    python -m paperlens.models.influence.collect --demo          # offline synthetic dataset

Output: data/influence/{nodes.jsonl, edges.jsonl, embeddings.npz}

Label note: isInfluential comes from Valenzuela et al. (AAAI-W 2015) style classifier run by S2 over full
text. It is *distant* supervision: we learn to reproduce a full-text signal from contexts + graph +
embeddings. State this clearly in the report.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ... import config

DEFAULT_SEEDS = [  # a spread of well-known arXiv papers across NLP / CV / ML (edit freely)
    "1706.03762", "1810.04805", "2005.14165", "1907.11692", "1910.10683", "2203.02155", "2302.13971",
    "2005.11401", "2104.08821", "2201.11903", "2106.09685", "2312.00752", "2111.00396", "2009.14794",
    "1512.03385", "2010.11929", "2002.05709", "1911.05722", "2103.00020", "2112.10752", "2006.11239",
    "2304.02643", "1505.04597", "1506.01497", "2103.14030", "1609.02907", "1710.10903", "1706.02216",
    "1810.00826", "2006.04768", "1301.3781", "1409.0473", "1508.04025", "1502.03167", "1412.6980",
    "1312.6114", "1406.2661", "1707.06347", "2210.03629", "2305.18290", "1803.05457", "1606.05250",
    "2004.05150", "2001.08361", "2204.02311", "2309.17453", "2307.09288", "2103.03206", "1908.10084",
    "2004.04906", "2007.01282", "2212.10560", "2310.06825", "2105.05233", "2208.01618", "1905.11946",
]


def _row(p: dict) -> dict:
    return {
        "paperId": p["paperId"], "title": p.get("title"), "abstract": p.get("abstract"),
        "year": p.get("year"), "citationCount": p.get("citationCount"),
        "arxiv": (p.get("externalIds") or {}).get("ArXiv"),
    }


def collect(client, seeds: list[str], per_seed_citers: int, max_refs: int, out_dir: Path, seed: int = 0):
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes: dict[str, dict] = {}
    citing: list[str] = []
    for s in seeds:
        try:
            p = client.paper(s if ":" in s or s.startswith("demo") else f"arXiv:{s}")
        except Exception as e:  # noqa: BLE001
            print(f"skip seed {s}: {e}")
            continue
        nodes[p["paperId"]] = _row(p)
        citing.append(p["paperId"])
        # snowball: a sample of papers citing the seed become additional citing papers.
        try:
            cits = [c for c in client.citations(p["paperId"], max_items=200) if c["contexts"]]
        except Exception as e:  # noqa: BLE001
            print(f"  citations failed for {s}: {e}")
            cits = []
        for c in rng.sample(cits, min(per_seed_citers, len(cits))):
            nodes[c["paper"]["paperId"]] = _row(c["paper"])
            citing.append(c["paper"]["paperId"])
    citing = list(dict.fromkeys(citing))
    print(f"{len(citing)} citing papers")

    n_edges = 0
    with open(out_dir / "edges.jsonl", "w") as f:
        for k, pid in enumerate(citing):
            try:
                refs = client.references(pid, max_items=max_refs)
            except Exception as e:  # noqa: BLE001
                print(f"  refs failed {pid}: {e}")
                continue
            for r in refs:
                nodes.setdefault(r["paper"]["paperId"], _row(r["paper"]))
                f.write(json.dumps({"src": pid, "dst": r["paper"]["paperId"], "contexts": r["contexts"],
                                    "s2_intents": r["s2_intents"], "label": int(r["isInfluential"])}) + "\n")
                n_edges += 1
            if k % 20 == 0:
                print(f"  [{k}/{len(citing)}] edges so far {n_edges}")

    with open(out_dir / "nodes.jsonl", "w") as f:
        for row in nodes.values():
            f.write(json.dumps(row) + "\n")
    ids = list(nodes)
    embs = client.embeddings(ids)
    have = [i for i in ids if i in embs]
    np.savez_compressed(out_dir / "embeddings.npz", ids=np.array(have),
                        vectors=np.array([embs[i] for i in have], dtype=np.float32))
    print(f"nodes {len(nodes)}  edges {n_edges}  embeddings {len(have)}  -> {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="text file with one arXiv id / S2 id per line")
    ap.add_argument("--per-seed-citers", type=int, default=8)
    ap.add_argument("--max-refs", type=int, default=300)
    ap.add_argument("--out", default=str(config.DATA_DIR / "influence"))
    ap.add_argument("--demo", action="store_true", help="use the synthetic world (no network)")
    args = ap.parse_args(argv)

    if args.demo:
        from ...sources.demo import DemoS2Client
        client = DemoS2Client()
        seeds = [p["paperId"] for p in client.w.papers if p["year"] >= 2016][::3]
        out = Path(args.out + "_demo") if args.out.endswith("influence") else Path(args.out)
    else:
        from ...sources.s2_client import S2Client
        client = S2Client()
        seeds = [l.strip() for l in open(args.seeds)] if args.seeds else DEFAULT_SEEDS
        seeds = [s for s in seeds if s and not s.startswith("#")]
        out = Path(args.out)
    collect(client, seeds, args.per_seed_citers, args.max_refs, out)


if __name__ == "__main__":
    main()
