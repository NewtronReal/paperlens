# PaperLens: understand a research paper at a glance

A FastAPI + Vite/React web app that takes an arXiv ID, DOI, title or PDF and shows, on one page: what the paper is about, **which
references it actually builds on**, **why** each one is cited, its ancestry (lineage), how the surrounding
ideas evolved over time, and where the code lives.

Under the hood there are two trained NLP/graph models (see [DESIGN.md](DESIGN.md) for the full rationale and
evaluation plan):

| | Task | Data | Model | Metric |
|---|---|---|---|---|
| **Model 1** | Citation intent: *background / method / result* for every citation sentence | SciCite (~11k sentences) | SciBERT fine-tuned (baseline: TF-IDF + LR) | macro-F1 |
| **Model 2** | Citation influence: does paper A truly build on reference B? | Semantic Scholar graph, `isInfluential` as distant label | GraphSAGE edge classifier over SPECTER2 node features + Model 1 edge features | AUROC / AUPRC / F1 |

Lineage, idea evolution and the "read these first" list are all computed from these two models' outputs.

## Quick start

Needs Python 3.10+ and Node 18+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..    # builds the React app into frontend/dist

# 1) Offline demo with a synthetic citation world (no network, no training needed)
PAPERLENS_DEMO=1 uvicorn paperlens.app:app --port 8000
# open http://localhost:8000  (FastAPI serves the built React app and the API)

# 2) Real papers (get a free key at https://www.semanticscholar.org/product/api)
export S2_API_KEY=...            # strongly recommended; the keyless pool is heavily rate-limited
export GITHUB_TOKEN=...          # optional, raises GitHub search limits
uvicorn paperlens.app:app --port 8000
```

### Frontend development (hot reload)

```bash
uvicorn paperlens.app:app --reload --port 8000     # terminal 1 (add PAPERLENS_DEMO=1 to work offline)
cd frontend && npm run dev                          # terminal 2, open http://localhost:5173
```

Vite proxies `/api` to `:8000` (override with `PAPERLENS_API=http://host:port npm run dev`), so no CORS setup
is needed. Run `npm run build` again when you want FastAPI to serve your changes on `:8000`.

Before training, the app runs with clearly-labelled heuristics in place of the models (the header shows
"M1 intent: untrained / M2 influence: untrained"). After training, it picks up the checkpoints automatically.

## Training (GPU recommended; a free Colab T4 is enough)

`notebooks/train_colab.ipynb` runs all of the steps below. Locally:

```bash
# Model 1 — citation intent
python -m paperlens.models.intent.train_baseline          # TF-IDF + LR, seconds on CPU
python -m paperlens.models.intent.train_scibert           # ~15 min on a T4
#   ablation: --model-name bert-base-uncased --tag intent_bert_base --out checkpoints/intent_bert_base
#   if the HF hub copy fails: bash scripts/download_scicite.sh   (then the loader finds data/scicite/)

# Model 2 — citation influence (needs S2_API_KEY; ~10-20 min of API calls, cached in data/cache.sqlite)
python -m paperlens.models.influence.collect               # writes data/influence/
python -m paperlens.models.influence.train                 # baselines + 5 variants x 3 seeds
```

Train Model 1 **before** Model 2: Model 2's edge features are Model 1's predictions, and are cached per
intent model so you can compare intent models inside the influence ablation.

Outputs:

* `checkpoints/intent_scibert/`, `checkpoints/intent_tfidf.joblib`, `checkpoints/influence_model.pt`
* `results/intent_*_metrics.json`, `results/intent_*_confusion.png`, `results/influence_results.{md,json}`

## Optional: LLM summary

Any OpenAI-compatible endpoint, e.g. a local Ollama model:

```bash
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b
```

Without it the "At a glance" card uses Semantic Scholar's TL;DR plus cue-phrase sentence routing on the abstract.

## Project layout

```
paperlens/
  app.py                      FastAPI: /api/analyze (job), /api/jobs/{id}, /api/health; serves frontend/dist
  config.py                   all settings via environment variables
  sources/                    Semantic Scholar, arXiv, GitHub clients (cached), PDF parsing, demo world
  analysis/                   pipeline, identifier resolution, idea evolution, summary
  models/intent/              SciCite loading, TF-IDF baseline, SciBERT fine-tuning, inference wrapper
  models/influence/           dataset collection, shared featurizer, GraphSAGE model, training, scorer
frontend/                     Vite + React app
  vite.config.js              dev server on :5173 with /api proxy; build -> dist/
  src/App.jsx                 page state: search, job polling, tabs, drawer
  src/api.js                  fetch wrappers for /api/analyze and job polling
  src/components/             TopBar, Status, Hero, Overview, CitationGraph, Lineage, Evolution,
                              PaperTable, Drawer, Cytoscape (shared graph wrapper), bits (badges, meters)
  src/lib/format.js           intent colours, labels, number formatting
tests/test_offline.py         end-to-end tests on the synthetic world (pytest -q)
DESIGN.md                     gap analysis, architecture, evaluation protocol, report tables
```

## API

```
POST /api/analyze   form: query=<arXiv id | URL | DOI | title>, file=<optional PDF>  -> {job_id}
GET  /api/jobs/{id} -> {status, stage, progress, result | error}
GET  /api/health    -> which models are loaded, demo flag
```

## Limitations

* Citation sentences come from Semantic Scholar; some publishers do not allow them. Uploading the PDF
  recovers them for numeric (`[3]`) and author-year citation styles.
* `isInfluential` is a model-generated label (distant supervision). See DESIGN.md §4.
* The synthetic demo world is for testing only. Its numbers say nothing about real papers.
