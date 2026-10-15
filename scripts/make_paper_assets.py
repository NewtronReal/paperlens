"""Build the paper's LaTeX tables and figures from results/ and data/influence/.

    .venv/bin/python scripts/make_paper_assets.py --paper ../paper

Every number in the paper's results section comes from here, so re-running after retraining keeps
the paper consistent with the checkpoints.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LABELS = ["background", "method", "result"]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300})


def load(name):
    return json.loads((RES / name).read_text())


def ms(vals, nd=3):
    return f"{np.mean(vals):.{nd}f} $\\pm$ {np.std(vals):.{nd}f}"


# ---------------------------------------------------------------- Model 1
def intent_table(out: Path):
    rows = [("TF-IDF + logistic regression", ["intent_tfidf"]),
            ("BERT-base (general domain)", ["intent_bert_base"]),
            ("SciBERT (scientific domain)", ["intent_scibert", "intent_scibert_s1", "intent_scibert_s2"])]
    lines = [r"\begin{table}[H]",
             r"\caption{Citation intent classification on the SciCite test set (1,859 sentences). SciBERT: mean $\pm$ "
             r"standard deviation over three seeds; the other rows are single runs.}",
             r"\centering", r"\label{tab:intent}", r"\footnotesize",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{L{3.9cm}|C{2.0cm}|C{2.0cm}|C{1.35cm}|C{1.35cm}|C{1.35cm}}", r"\hline",
             r"\textbf{Model} & \textbf{Macro-F1} & \textbf{Accuracy} & \textbf{F1 back\-ground} & "
             r"\textbf{F1 method} & \textbf{F1 result}\\", r"\hline"]
    for name, tags in rows:
        ms_ = [load(f"{t}_metrics.json")["test"] for t in tags]
        f1 = [m["macro_f1"] for m in ms_]
        acc = [m["accuracy"] for m in ms_]
        pc = [np.mean([m["per_class"][c]["f1-score"] for m in ms_]) for c in LABELS]
        cell = (lambda v: ms(v)) if len(tags) > 1 else (lambda v: f"{v[0]:.3f}")
        bold = name.startswith("SciBERT")
        r = [name, cell(f1), cell(acc)] + [f"{p:.3f}" for p in pc]
        if bold:
            r = [f"\\textbf{{{x}}}" for x in r]
        lines.append(" & ".join(r) + r"\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (out / "intent.tex").write_text("\n".join(lines) + "\n")


def intent_confusion(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, (tag, title) in zip(axes, [("intent_bert_base", "BERT-base"), ("intent_scibert", "SciBERT")]):
        cm = np.array(load(f"{tag}_metrics.json")["test"]["confusion"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j])}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else INK)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.set_title(title, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out / "intent_confusion.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Model 2
VARIANT_ROWS = [  # (key, label, graph, intent, emb)
    ("gnn", "GraphSAGE", "yes", "yes", "yes"),
    ("gnn-no-intent", "GraphSAGE", "yes", "--", "yes"),
    ("gnn-no-emb", "GraphSAGE", "yes", "yes", "--"),
    ("gnn-no-emb-no-intent", "GraphSAGE", "yes", "--", "--"),
    ("mlp", "MLP", "--", "yes", "yes"),
    ("mlp-no-intent", "MLP", "--", "--", "yes"),
    ("mlp-no-emb", "MLP", "--", "yes", "--"),
    ("mlp-no-emb-no-intent", "MLP", "--", "--", "--"),
]


def influence_table(out: Path, r: dict):
    deployed = "mlp-no-emb"
    lines = [r"\begin{table}[H]",
             r"\caption{Citation influence prediction on the test split (2,882 edges from 64 held-out citing papers). "
             r"Learned models: mean $\pm$ standard deviation over three seeds. Intent = Model~1 features; "
             r"Emb.\ = SPECTER2 node embeddings. The marked row was selected on validation AUPRC and is deployed.}",
             r"\centering", r"\label{tab:influence}", r"\footnotesize",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{L{3.6cm}|C{0.9cm}|C{0.9cm}|C{0.9cm}|C{2.0cm}|C{2.0cm}|C{2.0cm}}", r"\hline",
             r"\textbf{Model} & \textbf{Graph} & \textbf{Intent} & \textbf{Emb.} & \textbf{AUROC} & \textbf{AUPRC} & "
             r"\textbf{F1}\\", r"\hline"]
    base = [("heuristic: #mentions", "Number of mentions", "--", "--", "--"),
            ("heuristic: cited-paper citations", "Citations of cited paper", "--", "--", "--"),
            ("logreg: edge features", "Logistic regression", "--", "yes", "--")]
    for key, name, g, i, e in base:
        t = r["baselines"][key]["test"]
        lines.append(f"{name} & {g} & {i} & {e} & {t['auroc']:.3f} & {t['auprc']:.3f} & {t['f1']:.3f}\\\\")
    lines.append(r"\hline")
    for key, name, g, i, e in VARIANT_ROWS:
        a = r["variants"][key]["agg"]
        c = lambda k: f"{a[k][0]:.3f} $\\pm$ {a[k][1]:.3f}"  # noqa: E731
        row = [name + (" $^{\\dagger}$" if key == deployed else ""), g, i, e,
               c("test_auroc"), c("test_auprc"), c("test_f1")]
        if key == deployed:
            row = [f"\\textbf{{{x}}}" for x in row]
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (out / "influence.tex").write_text("\n".join(lines) + "\n")


def star_table(out: Path):
    s = load("influence_star_eval_v1.json")
    lines = [r"\begin{table}[H]",
             r"\caption{The same GraphSAGE checkpoint (no embeddings) scored with message passing over the full "
             r"collected graph and over one star graph per test paper, as the application does at inference.}",
             r"\centering", r"\label{tab:star}", r"\footnotesize",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{L{4.2cm}|C{1.6cm}|C{1.6cm}|C{1.6cm}}", r"\hline",
             r"\textbf{Graph at inference} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{F1}\\", r"\hline"]
    for mode, name in (("dense", "Full collected graph"), ("star", "Star graph per paper")):
        t = s[mode]["test"]
        lines.append(f"{name} & {t['auroc']:.3f} & {t['auprc']:.3f} & {t['f1']:.3f}\\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (out / "star.tex").write_text("\n".join(lines) + "\n")


def ambiguous_table(out: Path):
    a = load("influence_ambiguous.json")
    rows = [("Number of mentions", "#mentions", None), ("Logistic regression", "logreg-no-intent", "logreg"),
            ("MLP (no embeddings)", "mlp-no-emb-no-intent", "mlp-no-emb"),
            ("GraphSAGE (no embeddings)", "gnn-no-emb-no-intent", "gnn-no-emb")]

    def c(m, k):
        v = m[k]
        return f"{v[0]:.3f} $\\pm$ {v[1]:.3f}" if isinstance(v, list) else f"{v:.3f}"

    def mean(m, k):
        return m[k][0] if isinstance(m[k], list) else m[k]

    lines = [r"\begin{table}[H]",
             f"\\caption{{Test edges with one to three citation sentences only ({a['n_edges']:,} edges, "
             f"{100 * a['pos_rate']:.1f}\\% influential), where the mention count does not settle the label. "
             r"Learned models: mean $\pm$ standard deviation over three seeds.}",
             r"\centering", r"\label{tab:ambiguous}", r"\footnotesize",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{L{3.5cm}|C{2.0cm}|C{2.0cm}|C{2.0cm}|C{2.0cm}|C{1.3cm}}", r"\hline",
             r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c|}{\textbf{Without intent}} & "
             r"\multicolumn{2}{c|}{\textbf{With intent (Model 1)}} & \textbf{$\Delta$ AUROC}\\",
             r" & AUROC & AUPRC & AUROC & AUPRC & \\", r"\hline"]
    m = a["models"]
    for name, k0, k1 in rows:
        if k1 is None:
            lines.append(f"{name} & {c(m[k0], 'auroc')} & {c(m[k0], 'auprc')} & -- & -- & --\\\\")
        else:
            dlt = mean(m[k1], "auroc") - mean(m[k0], "auroc")
            lines.append(f"{name} & {c(m[k0], 'auroc')} & {c(m[k0], 'auprc')} & {c(m[k1], 'auroc')} & "
                         f"{c(m[k1], 'auprc')} & +{dlt:.3f}\\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (out / "ambiguous.tex").write_text("\n".join(lines) + "\n")


def influence_figure(out: Path, r: dict, edges_path: Path):
    n_ctx, lab = [], []
    with open(edges_path) as f:
        for line in f:
            e = json.loads(line)
            n_ctx.append(len(e["contexts"]))
            lab.append(e["label"])
    n_ctx, lab = np.array(n_ctx), np.array(lab)
    buckets = [("0", n_ctx == 0), ("1", n_ctx == 1), ("2", n_ctx == 2), ("3", n_ctx == 3),
               ("4-5", (n_ctx >= 4) & (n_ctx <= 5)), ("6+", n_ctx >= 6)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    rate = [lab[m].mean() for _, m in buckets]
    cnt = [int(m.sum()) for _, m in buckets]
    x = np.arange(len(buckets))
    a1.bar(x, rate, color=BLUE, width=0.62)
    a1.set_xticks(x, [f"{b}\n{c / 1000:.1f}k" for (b, _), c in zip(buckets, cnt)], fontsize=7.5)
    a1.set_xlabel("mentions per edge (number of edges)")
    a1.set_ylabel("share labelled influential")
    a1.set_ylim(0, 1.05)
    a1.set_title("(a) Label vs. number of mentions", fontsize=9)
    a1.grid(axis="y", color="#e5e5e2", linewidth=0.6)
    a1.set_axisbelow(True)

    pairs = [("GraphSAGE\n+emb", "gnn", "gnn-no-intent"), ("GraphSAGE\nno emb", "gnn-no-emb", "gnn-no-emb-no-intent"),
             ("MLP\n+emb", "mlp", "mlp-no-intent"), ("MLP\nno emb", "mlp-no-emb", "mlp-no-emb-no-intent")]
    y = np.arange(len(pairs))[::-1]
    for off, idx, color, marker, lbl in ((0.12, 1, BLUE, "o", "with intent features"),
                                         (-0.12, 2, ORANGE, "s", "without intent features")):
        m = [r["variants"][p[idx]]["agg"]["test_auprc"][0] for p in pairs]
        s = [r["variants"][p[idx]]["agg"]["test_auprc"][1] for p in pairs]
        a2.errorbar(m, y + off, xerr=s, fmt=marker, color=color, markersize=6, elinewidth=1, capsize=2,
                    label=lbl)
    lr = r["baselines"]["logreg: edge features"]["test"]["auprc"]
    a2.axvline(lr, color=INK, linestyle=":", linewidth=1, label="logistic regression")
    a2.set_yticks(y, [p[0].replace("\n", " ") for p in pairs])
    a2.set_xlabel("test AUPRC")
    a2.set_ylim(-0.6, len(pairs) - 0.4)
    a2.set_title("(b) Ablation (mean $\\pm$ sd, 3 seeds)", fontsize=9)
    a2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=3, handletextpad=0.3, columnspacing=1.0)
    a2.grid(axis="x", color="#e5e5e2", linewidth=0.6)
    a2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "influence.png", bbox_inches="tight")
    plt.close(fig)
    return {"edges": int(len(lab)), "zero_ctx_share": float((n_ctx == 0).mean()),
            "rate_by_bucket": dict(zip([b for b, _ in buckets], [round(float(v), 3) for v in rate])),
            "count_by_bucket": dict(zip([b for b, _ in buckets], cnt))}


def scicite_stats():
    pc = load("intent_scibert_metrics.json")["test"]["per_class"]
    return {"test_support": {c: int(pc[c]["support"]) for c in LABELS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(ROOT.parent / "paper"))
    args = ap.parse_args()
    paper = Path(args.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "images").mkdir(parents=True, exist_ok=True)

    r = load("influence_v2_results.json")
    intent_table(paper / "tables")
    intent_confusion(paper / "images")
    influence_table(paper / "tables", r)
    star_table(paper / "tables")
    ambiguous_table(paper / "tables")
    stats = influence_figure(paper / "images", r, ROOT / "data" / "influence" / "edges.jsonl")
    nodes = sum(1 for _ in open(ROOT / "data" / "influence" / "nodes.jsonl"))
    stats.update(nodes=nodes, citing_papers=len({json.loads(l)["src"] for l in open(ROOT / "data/influence/edges.jsonl")}),
                 splits=r["meta"]["splits"], pos_rate=r["meta"]["pos_rate"], scicite=scicite_stats(),
                 lr_coef=r["baselines"]["logreg: edge features"]["coef"])
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
