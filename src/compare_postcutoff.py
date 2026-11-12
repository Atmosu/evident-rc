"""Post-cutoff external set (public tenders, Oct 2024 to Dec 2025): EVIDENT-RC, the no-quote variant and the
baselines against the human labels. PROVISIONAL: one annotator per item until the second labels arrive.
Item-level paired bootstrap (10,000) of EVIDENT-RC minus each method, Holm per metric; RQ2 measures with
correct = F and Q right (and the attribute right where a gold attribute exists).

Usage: python src/compare_postcutoff.py      writes results/postcutoff_rq1.csv and results/postcutoff_rq2.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, analyze_llm as A
from compare_l1 import holm

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "results/evident"


def llm(run, variant=False):
    recs = [json.loads(l) for f in sorted(Path(run).glob("*.jsonl")) for l in open(f)]
    if variant:
        d = pd.DataFrame([{"req_id": r["req_id"], "project": r["project"], "F": r["gold"]["F"], "Q": r["gold"]["Q"],
                           "v_none": r["F"] is None, "v_F": r["F"]["label"] if r["F"] else 0,
                           "v_Q": r["Q"]["label"] if r["Q"] else 0} for r in recs])
    else:
        d = A.l1_frame(recs, 5); d["ok"] = d.v_tuple_ok.astype(bool); d["conf"] = d.v_c
    d["pred_F"] = np.where(d.v_none, -1, d.v_F); d["pred_Q"] = np.where(d.v_none, -1, d.v_Q)
    return d.set_index("req_id")


def base(name):
    b = pd.read_csv(ROOT / f"results/baselines/{name}_postcutoff.csv").set_index("req_id")
    if "conf_ts" in b:
        ga = b.attr.where(b.Q == 1); pa = b.pred_attr.where(b.pred_Q == 1)
        b["ok"] = (b.F == b.pred_F) & (b.Q == b.pred_Q) & (ga.isna() | (ga == pa)); b["conf"] = b.conf_ts
    return b


def main():
    main_run = next(EV.glob("*_k8_n10_*_constrained_q4_postcutoff"))
    l0_run = next(EV.glob("*_noevidence_postcutoff"))
    sysm = {"EVIDENT-RC": llm(main_run), "No quotes (L0)": llm(l0_run, True), "RoBERTa-large": base("roberta"),
            "BERT-base": base("bert"), "MiniLM fine-tuned": base("minilm"), "TF-IDF + SVM": base("tfidf")}
    idx_all = sysm["EVIDENT-RC"].index
    for k in sysm: sysm[k] = sysm[k].loc[idx_all]
    N = len(idx_all); ii = np.random.default_rng(42).integers(0, N, (10000, N))
    def counts(d): return M.fq_counts(d.reset_index())
    stats = {k: M.count_stats({c: v.values[ii].sum(1) for c, v in counts(d).items()}) for k, d in sysm.items()}
    rows = []
    for k, d in sysm.items():
        c = counts(d); pt = {m: float(v[0]) for m, v in M.count_stats({x: np.array([y.sum()]) for x, y in c.items()}).items()}
        m = M.l1(d.F, d.Q, d.pred_F.clip(0), d.pred_Q.clip(0))
        row = {"method": k, "FQ_F1": round(pt["FQ_F1"], 3), "exact_match": round(pt["exact_match"], 3),
               "FQ_FPR": round(pt["FQ_FPR"], 3), "FQ_P": round(float(m["FQ_P"]), 3), "FQ_R": round(float(m["FQ_R"]), 3),
               "FQ_pred": int(c.pred.sum()), "FQ_gold": int(c.gold.sum()), "n": N}
        if k != "EVIDENT-RC":
            for x in ("FQ_F1", "exact_match"):
                dd = stats["EVIDENT-RC"][x] - stats[k][x]; dd = dd[np.isfinite(dd)]; lo, hi = np.percentile(dd, [2.5, 97.5])
                row[f"{x}_diff_ci"] = f"{lo:.2f} to {hi:.2f}"; row[f"{x}_p"] = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
        rows.append(row)
    r1 = pd.DataFrame(rows)
    for x in ("FQ_F1", "exact_match"):
        sel = r1[f"{x}_p"].notna(); r1.loc[sel, f"{x}_p_holm"] = holm(r1.loc[sel, f"{x}_p"].values).round(4)
    rows2 = []
    for k in ("EVIDENT-RC", "RoBERTa-large", "BERT-base", "MiniLM fine-tuned"):
        d = sysm[k]; q = M.confidence_quality(d.conf.values, d.ok.values, bins=None if k == "EVIDENT-RC" else 10)
        rows2.append({"system": k, **{x: round(float(q[x]), 3) for x in ("accuracy", "AUROC", "ECE", "AURC")}})
    r2 = pd.DataFrame(rows2)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(r1.to_string(index=False)); print(); print(r2.to_string(index=False))
    r1.to_csv(ROOT / "results/postcutoff_rq1.csv", index=False); r2.to_csv(ROOT / "results/postcutoff_rq2.csv", index=False)
    cm = M.confusion(sysm["EVIDENT-RC"].F, sysm["EVIDENT-RC"].Q, sysm["EVIDENT-RC"].v_F, sysm["EVIDENT-RC"].v_Q)
    print(); print(cm)


if __name__ == "__main__":
    main()
