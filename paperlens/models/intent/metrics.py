"""Shared evaluation/reporting for intent models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import LABELS


def evaluate(y_true, y_pred) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "per_class": classification_report(y_true, y_pred, target_names=LABELS, output_dict=True,
                                           zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }


def save_report(name: str, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(metrics["test"]["confusion"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}: test macro-F1 {metrics['test']['macro_f1']:.3f}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_confusion.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plotting skipped:", e)
