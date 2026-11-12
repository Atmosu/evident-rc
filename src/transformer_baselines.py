"""Fine-tuned transformer baselines with temperature scaling (RQ1 comparison methods, RQ2 confidence baseline).

  bert     BERT-base-uncased, NoRBERT-style replica: per-label heads on the [CLS] vector, lr 2e-5, 10 epochs, batch 16
  roberta  RoBERTa-large, per-label heads on the first-token vector, lr 1e-5, 10 epochs, batch 16
  minilm   all-MiniLM-L6-v2, mean pooling (the pilot's best model), lr 5e-5, 8 epochs, batch 16

Every model has the same heads: two sigmoid outputs (F, Q) and a 7-way softmax over quality attributes, trained
jointly (attribute loss only on items with an attribute). Fixed published or pilot settings, no tuning on any
project, so all baselines get the same (zero) tuning budget; 10% linear warm-up, then linear decay.

Splits are the LLM's LOPO splits (evident.lopo_split): train on 13 projects, the next project is the validation
project, test on one. Only the 13 evaluation projects are test folds (P02 and P04 stay development-only). After
training, one temperature per head (F, Q, attribute) is fitted on the validation project by minimising its loss.
External (Dalpiaz-4): trained on PROMISE except P02, temperatures fitted on P02 (a development project).

Tuple confidence = probability of the predicted F value x of the predicted Q value x (if Q = 1) of the predicted
attribute, from the temperature-scaled heads (conf_ts) and the raw heads (conf_raw).

Usage: python src/transformer_baselines.py --model {bert,roberta,minilm} [--folds eval|dalpiaz|all] [--epochs N] [--limit N]
Writes results/baselines/<model>_<split>.csv (one row per requirement) and <model>_<split>_meta.json
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
import data as D
from evident import lopo_split, DEV_PROJECTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/baselines"
ATTRS = sorted(set(D.ATTR_MAP.values()))
CONFIGS = {"bert": {"hf": "bert-base-uncased", "pool": "cls", "lr": 2e-5, "epochs": 10},
           "roberta": {"hf": "roberta-large", "pool": "cls", "lr": 1e-5, "epochs": 10},
           "minilm": {"hf": "sentence-transformers/all-MiniLM-L6-v2", "pool": "mean", "lr": 5e-5, "epochs": 8}}
BS, MAXLEN = 16, 128


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Net(torch.nn.Module):
    def __init__(self, hf, pool):
        super().__init__()
        self.enc = AutoModel.from_pretrained(hf); self.pool = pool
        h = self.enc.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.fq = torch.nn.Linear(h, 2); self.attr = torch.nn.Linear(h, len(ATTRS))

    def forward(self, input_ids, attention_mask):
        x = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool == "mean":
            m = attention_mask.unsqueeze(-1).float(); x = (x * m).sum(1) / m.sum(1)
        else:
            x = x[:, 0]
        x = self.drop(x)
        return self.fq(x), self.attr(x)


def targets(df):
    y = torch.tensor(df[["F", "Q"]].values, dtype=torch.float32)
    a = torch.tensor([ATTRS.index(t) if q == 1 and isinstance(t, str) else -100 for t, q in zip(df.attr, df.Q)],
                     dtype=torch.long)
    return y, a


def batches(tok, texts, dev, order):
    for i in range(0, len(order), BS):
        j = order[i:i + BS]
        b = tok([texts[k] for k in j], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        yield j, b["input_ids"].to(dev), b["attention_mask"].to(dev)


def train(cfg, tok, tr, dev, seed, epochs):
    seed_all(seed)
    net = Net(cfg["hf"], cfg["pool"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps = epochs * ((len(tr) + BS - 1) // BS)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    y, a = targets(tr); texts = tr.text.tolist()
    bce, ce = torch.nn.BCEWithLogitsLoss(), torch.nn.CrossEntropyLoss(ignore_index=-100)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    net.train()
    for _ in range(epochs):
        order = np.random.permutation(len(tr))
        for j, ids, mask in batches(tok, texts, dev, order):
            with torch.autocast(**amp):
                zf, za = net(ids, mask)
            yb, ab = y[j].to(dev), a[j].to(dev)
            loss = bce(zf.float(), yb)
            if (ab >= 0).any(): loss = loss + ce(za.float(), ab)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    net.eval()
    return net


@torch.no_grad()
def logits(net, tok, df, dev):
    zf, za = [], []
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    for _, ids, mask in batches(tok, df.text.tolist(), dev, np.arange(len(df))):
        with torch.autocast(**amp):
            f, a = net(ids, mask)
        zf.append(f.float().cpu()); za.append(a.float().cpu())
    return torch.cat(zf), torch.cat(za)


def fit_temperature(z, y, kind):
    """One temperature minimising the validation loss (log-parameterised, LBFGS)."""
    if len(y) == 0: return 1.0
    logt = torch.zeros(1, requires_grad=True)
    lossf = torch.nn.BCEWithLogitsLoss() if kind == "bce" else torch.nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); l = lossf(z / logt.exp(), y); l.backward(); return l
    opt.step(closure)
    t = float(logt.detach().exp().clamp(0.05, 20))
    return t if np.isfinite(t) else 1.0


def predict(df, zf, za, T):
    pf, pq = torch.sigmoid(zf[:, 0] / T["F"]).numpy(), torch.sigmoid(zf[:, 1] / T["Q"]).numpy()
    pa = torch.softmax(za / T["attr"], 1).numpy()
    F, Q = (pf >= 0.5).astype(int), (pq >= 0.5).astype(int)
    A = pa.argmax(1)
    conf = np.where(F == 1, pf, 1 - pf) * np.where(Q == 1, pq, 1 - pq) * np.where(Q == 1, pa.max(1), 1.0)
    return F, Q, [ATTRS[k] if q == 1 else None for k, q in zip(A, Q)], conf


def run_split(name, cfg, tok, tr, va, te, dev, seed, epochs):
    t0 = time.time()
    net = train(cfg, tok, tr, dev, seed, epochs)
    zf_v, za_v = logits(net, tok, va, dev); zf_t, za_t = logits(net, tok, te, dev)
    yv, av = targets(va); keep = av >= 0
    T = {"F": fit_temperature(zf_v[:, 0], yv[:, 0], "bce"), "Q": fit_temperature(zf_v[:, 1], yv[:, 1], "bce"),
         "attr": fit_temperature(za_v[keep], av[keep], "ce")}
    F, Q, A, c_ts = predict(te, zf_t, za_t, T)
    _, _, _, c_raw = predict(te, zf_t, za_t, {"F": 1.0, "Q": 1.0, "attr": 1.0})
    rows = pd.DataFrame({"req_id": te.req_id, "project": te.project, "source": te.source, "F": te.F, "Q": te.Q,
                         "attr": te.attr, "pred_F": F, "pred_Q": Q, "pred_attr": A, "conf_ts": c_ts, "conf_raw": c_raw,
                         "val_project": va.project.iloc[0]})
    del net; torch.cuda.empty_cache()
    info = {"test": name, "val_project": va.project.iloc[0], "n_train": len(tr), "n_test": len(te), "T": T,
            "exact": float(((rows.F == rows.pred_F) & (rows.Q == rows.pred_Q)).mean()), "secs": round(time.time() - t0, 1)}
    print(json.dumps(info), flush=True)
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--folds", choices=["eval", "dalpiaz", "itrust", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=0); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: number of test folds and items per split")
    a = ap.parse_args()
    cfg = CONFIGS[a.model]; epochs = a.epochs or cfg["epochs"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["hf"]); OUT.mkdir(parents=True, exist_ok=True)
    P = D.promise(); cut = (lambda d: d.head(a.limit)) if a.limit else (lambda d: d)
    meta = {"model": a.model, **cfg, "epochs": epochs, "batch": BS, "max_len": MAXLEN, "seed": a.seed, "device": dev,
            "torch": torch.__version__}
    if a.folds in ("eval", "all"):
        test_projs = [p for p in sorted(P.project.unique()) if p not in DEV_PROJECTS][: a.limit or None]
        rows, infos = [], []
        for p in test_projs:
            tr, va, te = lopo_split(P, p)
            r, i = run_split(p, cfg, tok, cut(tr), cut(va), cut(te), dev, a.seed, epochs); rows.append(r); infos.append(i)
        pd.concat(rows).to_csv(OUT / f"{a.model}_lopo.csv", index=False)
        (OUT / f"{a.model}_lopo_meta.json").write_text(json.dumps({**meta, "folds": infos}, indent=1))
    for ext, load in (("dalpiaz", D.dalpiaz_public), ("itrust", D.itrust)):
        if a.folds not in (ext, "all") or (ext == "itrust" and a.folds == "all"): continue
        va = P[P.project == "P02"].reset_index(drop=True); tr = P[P.project != "P02"].reset_index(drop=True)
        te = load().reset_index(drop=True)
        r, i = run_split(ext, cfg, tok, cut(tr), cut(va), cut(te), dev, a.seed, epochs)
        r.to_csv(OUT / f"{a.model}_{ext}.csv", index=False)
        (OUT / f"{a.model}_{ext}_meta.json").write_text(json.dumps({**meta, "folds": [i]}, indent=1))


if __name__ == "__main__":
    main()
