# Citation-influence results (influence)

Nodes 13722, labelled edges 18930, positive rate 0.134, split by citing paper {'train': 13109, 'val': 2939, 'test': 2882}. Intent features from: SciBERT fine-tuned (SciCite).

| Model | Test AUROC | Test AUPRC | Test F1 |
|---|---|---|---|
| heuristic: #mentions | 0.9504 | 0.8842 | 0.8849 |
| heuristic: cited-paper citations | 0.4870 | 0.1435 | 0.2158 |
| logreg: edge features | 0.9614 | 0.9140 | 0.8711 |
| **gnn** | 0.8281 ± 0.0175 | 0.5095 ± 0.0528 | 0.4909 ± 0.0521 |
| **gnn-no-intent** | 0.6747 ± 0.0262 | 0.2952 ± 0.0295 | 0.3164 ± 0.0138 |
| **gnn-no-emb** | 0.9538 ± 0.0006 | 0.8970 ± 0.0027 | 0.8339 ± 0.0111 |
| **mlp** | 0.9416 ± 0.0005 | 0.8321 ± 0.0037 | 0.7419 ± 0.0098 |
| **mlp-no-intent** | 0.9489 ± 0.0005 | 0.8738 ± 0.0028 | 0.8009 ± 0.0077 |
