"""Fine-tune PhoBERT-base cho phân loại đa nhãn (sigmoid + BCE).

Chạy từ thư mục ati_final:
    python -m src.train_phobert --emoji text --loss bce
    python -m src.train_phobert --emoji keep --loss bce
    python -m src.train_phobert --emoji text --loss pw_sqrt

Quy tắc: siêu tham số / early stopping chỉ dựa trên val (Macro-F1 @ 0.5). Test chỉ dự đoán một lần sau khi nạp
checkpoint tốt nhất. Xác suất val/test được lưu trước khi áp ngưỡng.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import SEED, label_matrix, label_tiers, load_all
from .metrics import compute_metrics, f1_by_tier, save_preds
from .preprocess import emoji_to_text, normalize
from .segment import segmented_split

CHECKPOINT = "vinai/phobert-base"
ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_texts(data, emoji_mode):
    out = {}
    for s, df in data.items():
        texts = df.text.map(normalize)
        if emoji_mode == "text":
            texts = texts.map(emoji_to_text)
        tag = "emojitext" if emoji_mode == "text" else "keep"
        out[s] = segmented_split(s, texts.tolist(), tag)
    return out


def encode(tokenizer, texts, max_len):
    return tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]


def collate(ids_list, pad_id, device):
    L = max(len(x) for x in ids_list)
    X = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    for i, x in enumerate(ids_list):
        X[i, :len(x)] = torch.tensor(x)
    return X.to(device), (X != pad_id).long().to(device)


@torch.no_grad()
def predict(model, ids, pad_id, device, bs=64):
    model.eval()
    order = np.argsort([len(x) for x in ids])  # gom câu cùng độ dài để nhanh hơn
    out = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(ids), bs):
        idx = order[i:i + bs]
        X, M = collate([ids[j] for j in idx], pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=X, attention_mask=M).logits
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emoji", choices=["text", "keep"], default="text")
    ap.add_argument("--loss", choices=["bce", "pw_sqrt"], default="bce")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="chỉ để chạy thử nhanh: giới hạn số mẫu train")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"phobert_{a.emoji}_{a.loss}" + (f"_seed{a.seed}" if a.seed != SEED else "")
    if a.limit:
        name += "_smoke"

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] device={device} torch={torch.__version__} transformers={transformers.__version__}", flush=True)

    data = load_all()
    Y = {s: label_matrix(df) for s, df in data.items()}
    tiers = label_tiers(Y["train"].sum(0))
    seg = prepare_texts(data, a.emoji)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    ids = {s: encode(tok, seg[s], a.max_len) for s in seg}
    unk = np.mean([t == tok.unk_token_id for s in ids for x in ids[s] for t in x])
    print(f"tỉ lệ <unk>: {unk:.4f} | mẫu bị cắt (> {a.max_len}): {np.mean([len(x) >= a.max_len for x in ids['train']]):.4f}", flush=True)

    ntr = a.limit or len(ids["train"])
    tr_ids, tr_y = ids["train"][:ntr], torch.tensor(Y["train"][:ntr], dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=Y["train"].shape[1], problem_type="multi_label_classification").to(device)
    revision = getattr(model.config, "_commit_hash", None)
    n_params = sum(p.numel() for p in model.parameters())

    pw = None
    if a.loss == "pw_sqrt":
        pos = Y["train"].sum(0)
        pw = torch.tensor(np.sqrt((len(Y["train"]) - pos) / pos), dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": a.wd},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=a.lr)
    steps_per_epoch = (ntr + a.bs - 1) // a.bs
    total = steps_per_epoch * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt_dir = ROOT / "data" / "ckpt"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{name}_best.pt"
    best, bad, hist = -1.0, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); perm = np.random.permutation(ntr); tot = 0.0; te = time.time()
        for i in range(0, ntr, a.bs):
            idx = perm[i:i + a.bs]
            X, M = collate([tr_ids[j] for j in idx], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=X, attention_mask=M).logits
            loss = lossf(logits.float(), tr_y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx)
        Pv = predict(model, ids["val"], tok.pad_token_id, device)
        m = compute_metrics(Y["val"], Pv)
        hist.append({"epoch": ep, "train_loss": tot / ntr, "val_macro_f1": m["macro_f1"], "val_micro_f1": m["micro_f1"],
                     "val_lrap": m["lrap"], "epoch_seconds": time.time() - te})
        print(f"epoch {ep}: loss={tot / ntr:.4f} val Macro-F1={m['macro_f1']:.4f} Micro-F1={m['micro_f1']:.4f} ({time.time() - te:.0f}s)", flush=True)
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping", flush=True); break
    train_s = time.time() - t0

    model.load_state_dict(torch.load(best_path, map_location=device))
    P = {s: predict(model, ids[s], tok.pad_token_id, device) for s in ("val", "test")}
    for s in P:
        save_preds(ROOT / "results" / "preds" / f"{name}_{s}.npz", data[s].id.values, Y[s], P[s])
    res = {s: {**compute_metrics(Y[s], P[s]), **f1_by_tier(Y[s], P[s], tiers)} for s in P}
    print(json.dumps({s: {k: round(v, 4) for k, v in r.items()} for s, r in res.items()}, indent=1), flush=True)

    tab = ROOT / "results" / "tables"
    pd.DataFrame(hist).to_csv(tab / f"{name}_history.csv", index=False)
    best_epoch = int(np.argmax([h["val_macro_f1"] for h in hist])) + 1
    cost = {"name": name, "checkpoint": CHECKPOINT, "revision": revision, "n_params": n_params,
            "train_seconds": round(train_s, 1), "best_epoch": best_epoch, "epochs_run": len(hist),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
            "args": vars(a), "unk_rate": float(unk), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "test": res["test"], "val": res["val"]}
    (tab / f"{name}_cost.json").write_text(json.dumps(cost, indent=1, ensure_ascii=False), encoding="utf-8")
    print("xong:", name, flush=True)


if __name__ == "__main__":
    main()
