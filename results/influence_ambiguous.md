# Ambiguous edges (1-3 mentions), test split: 2134 edges, positive rate 0.039

| Model | AUROC | AUPRC |
|---|---|---|
| #mentions | 0.7093 | 0.1124 |
| logreg | 0.7741 | 0.1748 |
| logreg-no-intent | 0.7275 | 0.1398 |
| mlp-no-emb | 0.7630 ± 0.0007 | 0.1440 ± 0.0024 |
| mlp-no-emb-no-intent | 0.7172 ± 0.0034 | 0.1307 ± 0.0040 |
| gnn-no-emb | 0.7497 ± 0.0030 | 0.1273 ± 0.0018 |
| gnn-no-emb-no-intent | 0.7087 ± 0.0043 | 0.1176 ± 0.0032 |
