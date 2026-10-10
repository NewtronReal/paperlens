# PaperLens: design notes and evaluation plan

## 1. What the original idea lacked as a deep learning project

| Original feature | Problem as a DL project | Fix in PaperLens |
|---|---|---|
| "LLM for parsing" | Calling a pretrained LLM is inference through an API. Nothing is trained, there is no dataset and no metric. | The LLM is demoted to an optional summary layer. Two models are **trained and evaluated**: a citation-intent classifier and a citation-influence model. |
| "GNNs for relationships" | No task, no labels, no loss. | GNN is an **edge classifier**: given a citation edge (A cites B), predict whether B is influential to A. |
| "Citation importance prediction" | No definition of importance, so no ground truth. | Importance = Semantic Scholar's `isInfluential` flag, used as **distant supervision**; evaluated with AUROC / AUPRC / F1. |
| "Paper lineage" | Just a citation graph drawn twice. | Lineage = recursively follow the references with the highest **predicted** influence. It is a direct consumer of Model 2. |
| "Idea evolution graphs" | Undefined. | Cluster ancestors + descendants in SPECTER2 space, order clusters by time, name them with class-based TF-IDF. |
| Metadata, arXiv search, repo backtracking | Pure engineering. | Kept for the GUI, not claimed as DL contribution. |
| (missing) | No baselines, ablations or error analysis. | Baselines + ablations for both models, split designed to avoid leakage, scripts write result tables and confusion matrices. |

The project stays in **NLP**: Model 1 is a sentence classification model over scientific text, and Model 2's
most important edge features are produced by Model 1.

## 2. Architecture

```
query / PDF ──► resolve (arXiv id, DOI, S2 id, title) ──► Semantic Scholar: metadata, references, citations
                                                              │  each edge carries its citation sentences
                                                              ▼
            ┌───────────────── Model 1: citation intent (SciBERT, SciCite) ─────────────────┐
            │  every citation sentence → P(background), P(method), P(result)                 │
            └────────────────────────────────────────────────────────────────────────────────┘
                                                              │ per-edge intent features
                                                              ▼
            ┌──────────── Model 2: citation influence (GraphSAGE edge classifier) ───────────┐
            │  nodes: SPECTER2 embedding + log citations + year                                │
            │  edges: intent mean/max, #mentions, has-context, year gap                         │
            │  output: P(influential) for every reference and every citing paper              │
            └────────────────────────────────────────────────────────────────────────────────┘
                     │                    │                       │
                     ▼                    ▼                       ▼
              reading list        lineage tree          idea-evolution clusters
              (top influence)     (recursive top-k)     (k-means + c-TF-IDF)
                     └──────────── one-glance dashboard (FastAPI + Cytoscape.js) ────────────┘
```

PDF upload adds two things: the paper's own arXiv ID / title / GitHub links, and citation sentences for
references that Semantic Scholar has none for (numeric `[n]` and author-year styles are both handled).

## 3. Model 1: citation intent classification

* **Data**: SciCite (Cohan et al., NAACL 2019), ~11k citation sentences; labels background / method / result.
  Official train/dev/test split. Class imbalance: *result* is the minority class.
* **Model**: `allenai/scibert_scivocab_uncased` fine-tuned with a linear head; class-weighted cross-entropy,
  AdamW 2e-5, linear warmup 10 %, 4 epochs, max length 128, fp16 on GPU. Best epoch chosen on dev macro-F1.
* **Preprocessing**: citation markers (`[12]`, `(Smith et al., 2019)`) normalised to `[CIT]` in train and at
  inference, so S2 and PDF sentences look like training data.
* **Baseline**: TF-IDF word 1-2 grams + char 3-5 grams, logistic regression (C picked on dev).
* **Metric**: macro-F1 (primary), accuracy, per-class F1, confusion matrix.
* **Reference point**: the SciBERT paper (Beltagy et al., EMNLP 2019) reports roughly 85 macro-F1 on SciCite.
  Landing within about a point of that means the fine-tuning is correct.
* **Ablations** (flags on `train_scibert.py`): `--model-name bert-base-uncased` (domain pretraining),
  `--no-class-weights`, `--with-section` (adds section name; not used for the deployed model because
  S2 contexts carry no section).

## 4. Model 2: citation influence

* **Data**: `collect.py` snowballs from ~56 well-known arXiv seed papers plus a sample of papers citing them,
  storing every reference edge with its citation sentences and the `isInfluential` label, then fetches
  SPECTER2 embeddings for all nodes in batches.
* **Label caveat (say this in the report)**: `isInfluential` is itself the output of S2's classifier
  (Valenzuela et al., 2015) run on full text, so this is **distant supervision**. The task is: can a model
  that sees only citation sentences, embeddings and graph structure recover a full-text signal? The
  number-of-mentions feature is close to one of Valenzuela's own features, which is why it is a baseline.
* **Split**: by *citing paper* (70/15/15). All references of one paper land in one split; a random edge split
  would leak sibling edges.
* **Model**: input projection → 2 × (GraphSAGE mean-aggregation + LayerNorm + residual), jumping-knowledge
  concatenation of layer 0 and layer L, then an MLP over `[h_src, h_dst, h_src⊙h_dst, |h_src−h_dst|, edge features]`.
  BCE with positive-class weighting, DropEdge 0.3, early stopping on validation AUPRC.
* **Baselines**: number of mentions; cited paper's citation count; logistic regression on edge features;
  MLP with the same decoder but no message passing.
* **Ablations**: `gnn-no-intent` (Model 1 features zeroed: measures what the NLP model contributes),
  `gnn-no-emb` (SPECTER2 zeroed), `mlp`, `mlp-no-intent`. Three seeds each, mean ± std.
* **Deployment**: the run with the best *validation* AUPRC across all variants is saved; the test split
  is never used for selection. If the graph does not help on your data, the app will serve the MLP and say so.
  That is a legitimate finding; report it.
* **Known train/serve gap**: training graphs are dense; at inference the local graph around one paper is a star.
  DropEdge reduces the gap; the honest check is to evaluate the deployed model on held-out star graphs.

## 5. Results to put in the report

Fill these from `results/*.md` / `*.json` after training:

| Model 1 (SciCite test) | macro-F1 | acc |
|---|---|---|
| TF-IDF + LR | | |
| BERT-base | | |
| **SciBERT** | | |

| Model 2 (test, split by citing paper) | AUROC | AUPRC | F1 |
|---|---|---|---|
| #mentions | | | |
| cited-paper citations | | | |
| LR on edge features | | | |
| MLP | | | |
| MLP − intent | | | |
| **GraphSAGE** | | | |
| GraphSAGE − intent | | | |
| GraphSAGE − SPECTER2 | | | |

Error analysis worth one page each: confusion matrix of Model 1 (background vs result is usually the
hardest pair), and 10 false positives / negatives of Model 2 read by hand.

## 6. What is *not* claimed

* No novelty is claimed; the faculty requirement is a sound NLP project.
* The LLM summary (optional) and the extractive summary are presentation, not evaluated contributions.
* The synthetic demo world exists only so the app and the training code can be tested offline.
  Its numbers mean nothing about real papers and must not appear in the report.

## 7. References

* Cohan, Ammar, van Zuylen, Cady. *Structural Scaffolds for Citation Intent Classification in Scientific Publications.* NAACL 2019.
* Beltagy, Lo, Cohan. *SciBERT: A Pretrained Language Model for Scientific Text.* EMNLP 2019.
* Valenzuela, Ha, Etzioni. *Identifying Meaningful Citations.* AAAI Workshop on Scholarly Big Data, 2015.
* Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 2017.
* Singh et al. *SciRepEval: A Multi-Format Benchmark for Scientific Document Representations (SPECTER2).* EMNLP 2023.
* Rong et al. *DropEdge: Towards Deep Graph Convolutional Networks on Node Classification.* ICLR 2020.
* Grootendorst. *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* 2022.
