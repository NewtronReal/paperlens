# Star-graph evaluation (influence_star_eval_v1)

Checkpoint: GraphSAGE influence model 'gnn-no-emb' (intent feats: SciBERT fine-tuned (SciCite)). Test split: 2882 edges from 64 citing papers.

| Graph at inference | Test AUROC | Test AUPRC | Test F1 (val thr) | Test F1 (ckpt thr) |
|---|---|---|---|---|
| dense | 0.9541 | 0.8997 | 0.8393 | 0.8393 |
| star | 0.9541 | 0.8990 | 0.8405 | 0.8408 |
