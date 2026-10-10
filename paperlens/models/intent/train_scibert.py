"""Fine-tune SciBERT for citation intent on SciCite.

    python -m paperlens.models.intent.train_scibert --epochs 4 --batch-size 32
    # Colab T4: ~4 min/epoch.  CPU works but is slow (~1 h/epoch).

Ablations worth reporting (flags):
  --model-name bert-base-uncased        general-domain vs scientific pretraining
  --with-section                         prepend the section name ("introduction", "methods", ...)
  --no-class-weights                     effect of class imbalance handling ('result' is the minority class)
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ... import config
from .data import LABELS, clean_context, load_scicite
from .metrics import evaluate, save_report


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def prep(texts, sections, with_section):
    if with_section:
        return [f"{(s or 'unknown').lower()} [SEP] {clean_context(t)}" for t, s in zip(texts, sections)]
    return [clean_context(t) for t in texts]


@torch.no_grad()
def predict(model, tok, texts, device, bs=64, max_len=128):
    model.eval()
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        preds.extend(logits.argmax(-1).tolist())
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-section", action="store_true")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--max-train", type=int, default=None, help="subsample for quick smoke tests")
    ap.add_argument("--out", default=str(config.CKPT_DIR / "intent_scibert"))
    ap.add_argument("--tag", default="intent_scibert")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_scicite(args.data_dir)
    Xtr, ytr, Str = data["train"]
    if args.max_train:
        Xtr, ytr, Str = Xtr[:args.max_train], ytr[:args.max_train], Str[:args.max_train]
    Xtr = prep(Xtr, Str, args.with_section)
    Xdv = prep(data["dev"][0], data["dev"][2], args.with_section)
    Xte = prep(data["test"][0], data["test"][2], args.with_section)
    ydv, yte = data["dev"][1], data["test"][1]
    print(f"device {device} | train {len(Xtr)} dev {len(Xdv)} test {len(Xte)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)

    counts = np.bincount(ytr, minlength=len(LABELS)).astype(float)
    weights = None if args.no_class_weights else torch.tensor(counts.sum() / (len(LABELS) * counts.clip(min=1)),
                                                              dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    loader = DataLoader(list(zip(Xtr, ytr)), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(**batch).logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"  ep{ep} step {step}/{len(loader)} loss {run / step:.4f}")
        dev = evaluate(ydv, predict(model, tok, Xdv, device, max_len=args.max_len))
        history.append({"epoch": ep, "train_loss": run / len(loader), "dev_macro_f1": dev["macro_f1"]})
        print(f"epoch {ep}: loss {run / len(loader):.4f}  dev macro-F1 {dev['macro_f1']:.4f}  "
              f"({time.time() - t0:.0f}s)")
        if dev["macro_f1"] > best_f1:
            best_f1 = dev["macro_f1"]
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)

    best = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    metrics = {
        "model": args.model_name, "args": vars(args), "history": history,
        "dev": evaluate(ydv, predict(best, tok, Xdv, device, max_len=args.max_len)),
        "test": evaluate(yte, predict(best, tok, Xte, device, max_len=args.max_len)),
    }
    print(f"TEST macro-F1 {metrics['test']['macro_f1']:.4f}  acc {metrics['test']['accuracy']:.4f}")
    save_report(args.tag, metrics, config.RESULTS_DIR)
    with open(f"{args.out}/training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
