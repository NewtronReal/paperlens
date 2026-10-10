"""FastAPI backend. Serves the built React app (frontend/dist) at / and the JSON API under /api.

    uvicorn paperlens.app:app --reload            # real data (set S2_API_KEY)
    PAPERLENS_DEMO=1 uvicorn paperlens.app:app    # offline synthetic demo
    # frontend dev server with hot reload: cd frontend && npm run dev  (proxies /api to :8000)
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .analysis.pipeline import Analyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="PaperLens", version="1.1")
# Only needed if the Vite dev server calls the API directly instead of through its proxy.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict] = {}
_results: dict[str, dict] = {}   # query -> result (per-process memo)
_lock = threading.Lock()
_analyzer: Analyzer | None = None


def analyzer() -> Analyzer:
    global _analyzer
    with _lock:
        if _analyzer is None:
            _analyzer = Analyzer()
        return _analyzer


def _run(job_id: str, query: str | None, pdf: bytes | None):
    job = _jobs[job_id]

    def progress(stage: str, frac: float):
        job.update(stage=stage, progress=round(frac, 3))

    try:
        key = (query or "").strip().lower()
        if not pdf and key and key in _results:
            job.update(status="done", progress=1.0, stage="Done (cached)", result=_results[key])
            return
        res = analyzer().analyze(query=query, pdf_bytes=pdf, progress=progress)
        if not pdf and key:
            _results[key] = res
        job.update(status="done", result=res, finished=time.time())
    except Exception as e:  # noqa: BLE001
        logging.error(traceback.format_exc())
        job.update(status="error", error=str(e), finished=time.time())


@app.get("/api/health")
def health():
    a = analyzer()
    return {"ok": True, "demo": a.demo, "models": {"intent": a.intent.name, "influence": a.influence.name},
            "s2_key": bool(config.S2_API_KEY), "llm": bool(config.LLM_BASE_URL and config.LLM_MODEL)}


@app.post("/api/analyze")
async def analyze(query: str = Form(""), file: UploadFile | None = File(None)):
    pdf = await file.read() if file is not None else None
    if pdf is not None and not pdf.startswith(b"%PDF"):
        raise HTTPException(400, "Uploaded file is not a PDF")
    if not query.strip() and not pdf and not config.DEMO:
        raise HTTPException(400, "Enter an arXiv ID / URL, DOI, or title, or upload a PDF")
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "stage": "Queued", "progress": 0.0, "started": time.time()}
    _pool.submit(_run, job_id, query.strip() or None, pdf)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = _jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j


@app.get("/api/examples")
def examples():
    if analyzer().demo:
        w = analyzer().s2.w
        focus = w.default_focus()
        picks = [focus] + [p["paperId"] for p in sorted(w.papers, key=lambda p: -len(w.cits[w._by_id[p["paperId"]]]))
                           if 2017 <= p["year"] <= 2022][:3]
        return [{"query": pid, "label": w.papers[w._by_id[pid]]["title"]} for pid in dict.fromkeys(picks)]
    return [{"query": "1706.03762", "label": "Attention Is All You Need"},
            {"query": "2106.09685", "label": "LoRA"},
            {"query": "2005.11401", "label": "Retrieval-Augmented Generation"},
            {"query": "2312.00752", "label": "Mamba"}]


_NOT_BUILT = """<!doctype html><meta charset=utf-8><title>PaperLens</title>
<body style="font:16px system-ui;max-width:640px;margin:10vh auto;line-height:1.5">
<h2>PaperLens API is running</h2><p>The React frontend has not been built yet. Either build it once:</p>
<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>and reload this page, or run the dev server with hot reload and open <a href="http://localhost:5173">localhost:5173</a>:</p>
<pre>cd frontend &amp;&amp; npm run dev</pre></body>"""

if (config.FRONTEND_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIR / "assets"), name="assets")


@app.get("/", response_model=None)
def index():
    built = config.FRONTEND_DIR / "index.html"
    return FileResponse(built) if built.exists() else HTMLResponse(_NOT_BUILT)
