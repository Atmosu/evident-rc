"""Pilot experiment: requirement type (F/Q multi-label) and quality attribute.

Baselines
  B1  TF-IDF (1-2 grams) + LinearSVC, one binary classifier per label (conventional)
  B2  MiniLM sentence embeddings (frozen) + logistic regression (model-based, feature extraction)
  B3  MiniLM fine-tuned with a sigmoid multi-label head (model-based, fine-tuning)
  Z   MiniLM zero-shot cosine similarity to label descriptions (Alhoshan et al. 2023 style)

Protocols
  random  stratified 10-fold CV on PROMISE (optimistic, as in prior work)
  lopo    leave-one-project-out over the 15 PROMISE projects (unseen-project setting)
  ext     train on all PROMISE, test on Dalpiaz public industrial sets and iTrust (no retraining)
"""
import json, random, sys, time
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, accuracy_score, f1_score
from safetensors.torch import load_file
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))
import data as D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
SEED = 42
torch.set_num_threads(2)
LABELS = ["F", "Q"]


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------- MiniLM ----------------
MDIR = ROOT / "models/minilm"
TOK = PreTrainedTokenizerFast(tokenizer_file=str(MDIR / "tokenizer.json"), unk_token="[UNK]",
                              pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def load_encoder():
    m = BertModel(BertConfig.from_pretrained(MDIR), add_pooling_layer=False)
    sd = {k: v.float() for k, v in load_file(MDIR / "model.fp16.safetensors").items()}
    m.load_state_dict(sd, strict=False)
    return m


def mean_pool(h, mask):
    return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


@torch.no_grad()
def embed(texts, enc=None, bs=64):
    enc = enc or load_encoder(); enc.eval(); out = []
    for i in range(0, len(texts), bs):
        b = TOK(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        b.pop("token_type_ids", None)
        out.append(torch.nn.functional.normalize(mean_pool(enc(**b).last_hidden_state, b["attention_mask"]), dim=-1))
    return torch.cat(out).numpy()


class FT(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__(); self.enc = load_encoder(); self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(384, n_out)

    def forward(self, b):
        h = self.enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        return self.head(self.drop(mean_pool(h, b["attention_mask"])))


def finetune(texts, Y, multilabel=True, epochs=8, lr=5e-5, bs=16):
    seed_all(); n_out = Y.shape[1] if multilabel else int(Y.max()) + 1
    model = FT(n_out); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
    idx = np.arange(len(texts)); model.train()
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            b = TOK([texts[k] for k in j], padding=True, truncation=True, max_length=128, return_tensors="pt")
            y = torch.tensor(Y[j], dtype=torch.float32) if multilabel else torch.tensor(Y[j].ravel(), dtype=torch.long)
            loss = lossf(model(b), y); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def ft_predict(model, texts, multilabel=True, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = TOK(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        out.append(model(b))
    z = torch.cat(out)
    return (torch.sigmoid(z) >= 0.5).int().numpy() if multilabel else z.argmax(1).numpy()


# ---------------- per-label binary relevance baselines ----------------
def b1_fit_predict(tr_text, Ytr, te_text):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform(tr_text); Xte = vec.transform(te_text)
    return np.column_stack([LinearSVC(class_weight="balanced", C=1.0, random_state=SEED).fit(Xtr, Ytr[:, k]).predict(Xte)
                            if len(set(Ytr[:, k])) > 1 else np.full(Xte.shape[0], Ytr[0, k])
                            for k in range(Ytr.shape[1])])


def b2_fit_predict(Etr, Ytr, Ete):
    return np.column_stack([LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0).fit(Etr, Ytr[:, k]).predict(Ete)
                            for k in range(Ytr.shape[1])])


# zero-shot label descriptions (positive vs negative), Alhoshan-style
ZS = {"F": ("The system shall perform a function, action, or behaviour for the user, processing inputs to produce outputs.",
            "The requirement describes a quality, constraint, or property of the system such as performance, security, or usability."),
      "Q": ("The requirement specifies a quality of the system such as performance, security, usability, availability, reliability, maintainability, look and feel, legal compliance, or operating environment.",
            "The requirement describes only a function or feature the system shall perform, without any quality criterion.")}


def zeroshot_predict(E):
    enc = load_encoder(); cols = []
    for lab in LABELS:
        pos, neg = embed([ZS[lab][0], ZS[lab][1]], enc)
        cols.append((E @ pos > E @ neg).astype(int))
    return np.column_stack(cols)


# ---------------- metrics ----------------
def metrics(Y, P, df):
    p, r, f, _ = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    fq = ((df.F == 1) & (df.Q == 1)).values
    res = {"F_P": p[0], "F_R": r[0], "F_F1": f[0], "Q_P": p[1], "Q_R": r[1], "Q_F1": f[1],
           "macroF1": f1_score(Y, P, average="macro", zero_division=0),
           "hamming": hamming_loss(Y, P), "subset_acc": accuracy_score(Y, P),
           "Q_recall_on_FQ": P[fq, 1].mean() if fq.any() else np.nan,
           "both_correct_on_FQ": ((P[fq] == 1).all(1)).mean() if fq.any() else np.nan, "n": len(Y)}
    return {k: round(float(v), 3) if not isinstance(v, int) else v for k, v in res.items()}


def run_l1():
    P = D.promise(); Y = P[LABELS].values; texts = P.text.tolist()
    E = embed(texts)
    ext = pd.concat([D.dalpiaz_public(), D.itrust()], ignore_index=True)
    rows, preds = [], {}
    for proto in ["random", "lopo"]:
        splits = (StratifiedKFold(10, shuffle=True, random_state=SEED).split(texts, P.F * 2 + P.Q) if proto == "random"
                  else LeaveOneGroupOut().split(texts, groups=P.project))
        Pb = {m: np.zeros_like(Y) for m in ["B1", "B2", "B3"]}
        for k, (tr, te) in enumerate(splits):
            t0 = time.time()
            assert proto == "random" or not set(P.project.iloc[tr]) & set(P.project.iloc[te]), "project leakage"
            Pb["B1"][te] = b1_fit_predict([texts[i] for i in tr], Y[tr], [texts[i] for i in te])
            Pb["B2"][te] = b2_fit_predict(E[tr], Y[tr], E[te])
            m = finetune([texts[i] for i in tr], Y[tr])
            Pb["B3"][te] = ft_predict(m, [texts[i] for i in te])
            print(f"  {proto} fold {k} done {time.time()-t0:.0f}s", flush=True)
        for mname, Pm in Pb.items():
            rows.append({"protocol": proto, "model": mname, **metrics(Y, Pm, P)})
            preds[(proto, mname)] = Pm
    # zero-shot (no training)
    Pz = zeroshot_predict(E); preds[("zero-shot (all PROMISE)", "Z")] = Pz
    rows.append({"protocol": "zero-shot (all PROMISE)", "model": "Z", **metrics(Y, Pz, P)})
    # external: train on all PROMISE
    Eext = embed(ext.text.tolist()); mft = finetune(texts, Y)
    ext_preds = {"B1": b1_fit_predict(texts, Y, ext.text.tolist()), "B2": b2_fit_predict(E, Y, Eext),
                 "B3": ft_predict(mft, ext.text.tolist()), "Z": zeroshot_predict(Eext)}
    for src in ["dronology", "leeds", "reqview", "wasp", "Dalpiaz-4 (pooled)", "iTrust"]:
        mask = (ext.source.isin(["dronology", "leeds", "reqview", "wasp"]) if src.startswith("Dalpiaz") else ext.source == src).values
        for mname, Pm in ext_preds.items():
            rows.append({"protocol": f"ext:{src}", "model": mname, **metrics(ext[LABELS].values[mask], Pm[mask], ext[mask])})
    res = pd.DataFrame(rows); res.to_csv(OUT / "l1_results.csv", index=False)
    # per-item predictions for every protocol and model (all later metrics are computed from this file)
    item = lambda proto, m, d, Pm: pd.DataFrame({"protocol": proto, "model": m, "req_id": d.req_id, "source": d.source,
                                                 "project": d.project, "F": d.F, "Q": d.Q, "pred_F": Pm[:, 0], "pred_Q": Pm[:, 1]})
    pd.concat([item(pr, m, P, Pm) for (pr, m), Pm in preds.items()] +
              [item("ext", m, ext, Pm) for m, Pm in ext_preds.items()], ignore_index=True
              ).to_csv(OUT / "l1_predictions.csv", index=False)
    # failure cases (LOPO, best model chosen later) + external
    fails = []
    for mname in ["B1", "B3"]:
        Pm = preds[("lopo", mname)]
        for i in np.where((Y != Pm).any(1))[0]:
            fails.append({"setting": "lopo", "model": mname, "req_id": P.req_id[i], "project": P.project[i],
                          "text": texts[i], "gold": f"F={Y[i,0]},Q={Y[i,1]}", "pred": f"F={Pm[i,0]},Q={Pm[i,1]}",
                          "orig_class": P.orig_class[i]})
        Pm = ext_preds[mname]
        for i in np.where((ext[LABELS].values != Pm).any(1))[0]:
            fails.append({"setting": "ext:" + ext.source[i], "model": mname, "req_id": ext.req_id[i], "project": ext.project[i],
                          "text": ext.text[i], "gold": f"F={ext.F[i]},Q={ext.Q[i]}", "pred": f"F={Pm[i,0]},Q={Pm[i,1]}", "orig_class": ""})
    pd.DataFrame(fails).to_csv(OUT / "l1_failures.csv", index=False)
    return res


def run_l2():
    P = D.promise(); P = P[P.attr.notna()].reset_index(drop=True)
    classes = sorted(P.attr.unique()); y = P.attr.map({c: i for i, c in enumerate(classes)}).values
    texts = P.text.tolist(); E = embed(texts); rows = []
    for proto in ["random", "lopo"]:
        splits = (StratifiedKFold(10, shuffle=True, random_state=SEED).split(texts, y) if proto == "random"
                  else LeaveOneGroupOut().split(texts, groups=P.project))
        pr = {m: np.zeros_like(y) for m in ["B1", "B2", "B3"]}
        for tr, te in splits:
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); Xtr = vec.fit_transform([texts[i] for i in tr])
            pr["B1"][te] = LinearSVC(class_weight="balanced", random_state=SEED).fit(Xtr, y[tr]).predict(vec.transform([texts[i] for i in te]))
            pr["B2"][te] = LogisticRegression(class_weight="balanced", max_iter=3000).fit(E[tr], y[tr]).predict(E[te])
            m = finetune([texts[i] for i in tr], y[tr].reshape(-1, 1), multilabel=False, epochs=10)
            pr["B3"][te] = ft_predict(m, [texts[i] for i in te], multilabel=False)
        for mname, p in pr.items():
            pc = precision_recall_fscore_support(y, p, labels=range(len(classes)), zero_division=0)[2]
            rows.append({"protocol": proto, "model": mname, "macroF1": round(f1_score(y, p, average="macro"), 3),
                         "weightedF1": round(f1_score(y, p, average="weighted"), 3), "acc": round(accuracy_score(y, p), 3),
                         **{f"F1_{c}": round(v, 2) for c, v in zip(classes, pc)}})
    res = pd.DataFrame(rows); res.to_csv(OUT / "l2_results.csv", index=False)
    return res


if __name__ == "__main__":
    t = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "l1"): print(run_l1().to_string())
    if which in ("all", "l2"): print(run_l2().to_string())
    print(f"total {time.time()-t:.0f}s")
