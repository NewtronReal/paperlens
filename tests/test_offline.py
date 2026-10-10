"""Offline tests: everything runs against the synthetic demo world, no network needed.

    pytest -q
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    from paperlens import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CKPT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "CACHE_DB", tmp_path / "data" / "cache.sqlite")
    from paperlens.models.influence.scorer import load_influence_scorer
    from paperlens.models.intent.classifier import load_intent_classifier
    load_intent_classifier.cache_clear()
    load_influence_scorer.cache_clear()
    yield
    load_intent_classifier.cache_clear()
    load_influence_scorer.cache_clear()


def test_resolve_identifiers():
    from paperlens.analysis.resolve import to_identifier
    assert to_identifier("https://arxiv.org/abs/1706.03762v7") == ("arXiv:1706.03762", "arxiv")
    assert to_identifier("1706.03762") == ("arXiv:1706.03762", "arxiv")
    assert to_identifier("arXiv:2312.00752")[0] == "arXiv:2312.00752"
    assert to_identifier("10.18653/v1/N19-1423")[1] == "doi"
    assert to_identifier("Attention is all you need") == (None, "title")


def test_github_extraction():
    from paperlens.sources.repos import extract_github_urls
    t = "Code: https://github.com/state-spaces/mamba. See also github.com/foo (no) and https://github.com/huggingface/peft.git)"
    assert extract_github_urls(t) == ["https://github.com/state-spaces/mamba", "https://github.com/huggingface/peft"]


def test_pdf_contexts_numeric_and_author_year():
    from paperlens.sources.pdf_parse import ParsedPDF, _expand
    assert _expand("3, 5-7") == {3, 5, 6, 7}
    p = ParsedPDF(sentences=["We build on the Transformer architecture [3, 5].", "Unrelated sentence about data [9].",
                             "Unlike Devlin et al. (2019), we do not pretrain."],
                  numbered_refs={3: "A. Vaswani et al. Attention is all you need. NeurIPS 2017.",
                                 9: "Some dataset paper."})
    assert p.contexts_for("Attention Is All You Need", "Vaswani", 2017) == ["We build on the Transformer architecture [3, 5]."]
    assert p.contexts_for("BERT: Pre-training of deep bidirectional transformers", "Devlin", 2019) == \
        ["Unlike Devlin et al. (2019), we do not pretrain."]


def test_heuristic_intent_shapes():
    from paperlens.models.intent.classifier import HeuristicIntent
    P = HeuristicIntent().predict_proba(["We follow [CIT].", "Our model outperforms [CIT].", "X is popular [CIT]."])
    assert P.shape == (3, 3) and np.allclose(P.sum(1), 1)
    assert P.argmax(1).tolist() == [1, 2, 0]


def test_demo_analysis_end_to_end():
    from paperlens.analysis.pipeline import Analyzer
    stages = []
    res = Analyzer(demo=True).analyze("", progress=lambda m, f: stages.append(m))
    json.dumps(res)  # must be serialisable
    assert stages[-1] == "Done"
    assert res["references"] and res["citations"]
    infl = [r["influence"] for r in res["references"]]
    assert infl == sorted(infl, reverse=True), "references sorted by predicted influence"
    assert all(r["intent"] in ("background", "method", "result", "unknown") for r in res["references"])
    assert res["lineage"]["nodes"][0]["level"] == 0 and len(res["lineage"]["edges"]) > 0
    assert res["evolution"]["clusters"] and res["evolution"]["storyline"]
    assert any("DEMO" in w for w in res["warnings"])


def test_api_job_flow(monkeypatch):
    from paperlens import config
    monkeypatch.setattr(config, "DEMO", True)
    from fastapi.testclient import TestClient
    import paperlens.app as appmod
    appmod._analyzer = None
    c = TestClient(appmod.app)
    assert c.get("/api/health").json()["demo"] is True
    assert len(c.get("/api/examples").json()) >= 1
    job = c.post("/api/analyze", data={"query": ""}).json()["job_id"]
    for _ in range(100):
        j = c.get(f"/api/jobs/{job}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    assert j["result"]["paper"]["title"]
    assert c.get("/").status_code == 200
    assert c.post("/api/analyze", data={"query": "x"}, files={"file": ("a.pdf", b"not a pdf")}).status_code == 400


@pytest.mark.skipif(not __import__("importlib").util.find_spec("torch_geometric"), reason="torch_geometric missing")
def test_influence_training_smoke(tmp_path):
    from paperlens import config
    from paperlens.models.influence import collect, train
    from paperlens.models.influence.scorer import LearnedInfluence, load_influence_scorer
    from paperlens.sources.demo import DemoS2Client, DemoWorld
    client = DemoS2Client(DemoWorld(n_papers=120, seed=3))
    seeds = [p["paperId"] for p in client.w.papers if p["year"] >= 2018][::2]
    collect.collect(client, seeds, per_seed_citers=2, max_refs=50, out_dir=tmp_path / "inf")
    train.main(["--data", str(tmp_path / "inf"), "--variants", "gnn", "mlp", "--seeds", "1",
                "--epochs", "15", "--patience", "5", "--hidden", "32"])
    res = json.loads((config.RESULTS_DIR / "influence_results.json").read_text())
    assert {"gnn", "mlp"} <= set(res["variants"])
    load_influence_scorer.cache_clear()
    scorer = load_influence_scorer()
    assert isinstance(scorer, LearnedInfluence)


def _mock_s2():
    """S2Client wired to an httpx.MockTransport that mimics the real Graph API response shapes."""
    import httpx
    from paperlens.sources.cache import NullCache
    from paperlens.sources.s2_client import S2Client

    def paper(pid, year, title, arxiv=None):
        return {"paperId": pid, "title": title, "year": year, "citationCount": 10, "abstract": f"{title}. We propose X.",
                "externalIds": {"ArXiv": arxiv} if arxiv else {}, "authors": [{"name": "A. Author"}]}

    focus = {**paper("p0", 2021, "Focus Paper", "2101.00001"), "tldr": {"text": "A focus paper."},
             "influentialCitationCount": 1, "referenceCount": 2, "url": "https://www.semanticscholar.org/paper/p0"}
    refs = {"offset": 0, "data": [
        {"contexts": ["We follow the setup of [1]."], "intents": ["methodology"], "isInfluential": True,
         "citedPaper": paper("r1", 2017, "Reference One", "1706.03762")},
        {"contexts": [], "intents": [], "isInfluential": False, "citedPaper": paper("r2", 2015, "Reference Two")},
        {"contexts": [], "intents": [], "isInfluential": False, "citedPaper": {"paperId": None, "title": "unresolved"}},
    ]}
    cits = {"offset": 0, "data": [{"contexts": ["Unlike [4], we ..."], "intents": ["result"], "isInfluential": False,
                                   "citingPaper": paper("c1", 2023, "Citing One")}]}

    def handler(req: httpx.Request):
        path = req.url.path
        if req.method == "POST" and path.endswith("/paper/batch"):
            ids = json.loads(req.content)["ids"]
            return httpx.Response(200, json=[{"paperId": i, "embedding": {"model": "specter_v2", "vector": [0.01] * 768}}
                                             if i != "r2" else None for i in ids])
        if path.endswith("/references"):
            return httpx.Response(200, json=refs)
        if path.endswith("/citations"):
            return httpx.Response(200, json=cits)
        if path.endswith("/paper/arXiv:2101.00001") or path.endswith("/paper/p0"):
            return httpx.Response(200, json=focus)
        if path.endswith(("/paper/r1", "/paper/r2")):
            return httpx.Response(200, json=paper(path.rsplit("/", 1)[-1], 2016, "Ref"))
        return httpx.Response(404, json={"error": "not found"})

    c = S2Client(api_key="test", cache=NullCache(), min_interval=0)
    c.http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.semanticscholar.org")
    return c


def test_s2_client_shapes():
    c = _mock_s2()
    refs = c.references("p0")
    assert [r["paper"]["paperId"] for r in refs] == ["r1", "r2"]       # unresolved reference dropped
    assert refs[0]["contexts"] and refs[0]["isInfluential"] is True
    assert set(c.embeddings(["p0", "r1", "r2"])) == {"p0", "r1"}


def test_real_mode_pipeline_with_mocked_apis():
    from paperlens.analysis.pipeline import Analyzer

    class NoGitHub:
        def repo_info(self, url, source="paper"):
            return {"url": url, "source": source}

        def search_by_arxiv(self, arxiv_id, n=3):
            return []

    a = Analyzer(s2=_mock_s2(), arxiv=False, github=NoGitHub(), demo=False)
    res = a.analyze("https://arxiv.org/abs/2101.00001")
    json.dumps(res)
    assert res["paper"]["arxiv"] == "2101.00001" and res["summary"]["tldr"] == "A focus paper."
    assert res["references"][0]["paperId"] == "r1" and res["references"][0]["intent"] == "method"
    assert res["references"][1]["intent"] == "unknown"
    assert res["citations"][0]["paperId"] == "c1"
