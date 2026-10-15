# Citation-influence results (influence_v2)

Nodes 13722, labelled edges 18930, positive rate 0.134, split by citing paper {'train': 13109, 'val': 2939, 'test': 2882}. Intent features from: SciBERT fine-tuned (SciCite).

| Model | Test AUROC | Test AUPRC | Test F1 |
|---|---|---|---|
| heuristic: #mentions | 0.9504 | 0.8842 | 0.8849 |
| heuristic: cited-paper citations | 0.4870 | 0.1435 | 0.2158 |
| logreg: edge features | 0.9614 | 0.9140 | 0.8711 |
| **gnn** | 0.9316 ± 0.0027 | 0.7994 ± 0.0099 | 0.6941 ± 0.0085 |
| **gnn-no-intent** | 0.9405 ± 0.0022 | 0.8431 ± 0.0076 | 0.7466 ± 0.0172 |
| **gnn-no-emb** | 0.9574 ± 0.0003 | 0.9079 ± 0.0001 | 0.8669 ± 0.0036 |
| **gnn-no-emb-no-intent** | 0.9506 ± 0.0004 | 0.9043 ± 0.0005 | 0.8763 ± 0.0006 |
| **mlp** | 0.9416 ± 0.0005 | 0.8321 ± 0.0037 | 0.7419 ± 0.0098 |
| **mlp-no-intent** | 0.9489 ± 0.0005 | 0.8738 ± 0.0028 | 0.8009 ± 0.0077 |
| **mlp-no-emb** | 0.9595 ± 0.0001 | 0.9119 ± 0.0004 | 0.8739 ± 0.0022 |
| **mlp-no-emb-no-intent** | 0.9517 ± 0.0006 | 0.9054 ± 0.0005 | 0.8771 ± 0.0034 |
