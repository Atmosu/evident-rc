"""RQ1 and RQ2 comparison of EVIDENT-RC with the fine-tuned transformer baselines (src/transformer_baselines.py)
and TF-IDF + SVM (pilot), on the 13 evaluation projects (project-clustered paired bootstrap) and Dalpiaz-4
(item-level paired bootstrap; 4 projects are too few). Holm correction per metric and test set.
RQ2: each system with its own predictions; correct = full tuple (F, Q, attribute); baselines use their
temperature-scaled tuple confidence (conf_ts), and the raw confidence is reported alongside. RQ2 differences use
10,000 item-level paired resamples, as for RQ1.

Usage: python src/compare_baselines.py <lopo_run> <dalpiaz_run> [--n 5]
Writes results/rq1_baselines.csv and results/rq2_baselines.csv"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, analyze_llm as A
from compare_l1 import holm
from evident import DEV_PROJECTS

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"roberta": "RoBERTa-large", "bert": "BERT-base (NoRBERT)", "minilm": "MiniLM fine-tuned", "B1": "TF-IDF + SVM"}
DALPIAZ = ["dronology", "leeds", "reqview", "wasp"]


def llm_frame(run, n):
    recs = [json.loads(l) for f in sorted(Path(run).glob("*.jsonl")) for l in open(f)]
    d = A.l1_frame(recs, n)
    d["pred_F"] = np.where(d.v_none, -1, d.v_F); d["pred_Q"] = np.where(d.v_none, -1, d.v_Q)
    d["ok"] = d.v_tuple_ok.astype(bool); d["conf"] = d.v_c
    return d.set_index("req_id")


def base_frames(split):
    out = {}
    for m in ("roberta", "bert", "minilm"):
        b = pd.read_csv(ROOT / f"results/baselines/{m}_{'lopo' if split == 'lopo' else 'dalpiaz'}.csv").set_index("req_id")
        ga = b.attr.where(b.Q == 1); pa = b.pred_attr.where(b.pred_Q == 1)
        b["ok"] = (b.F == b.pred_F) & (b.Q == b.pred_Q) & (ga.isna() | (ga == pa))
        out[m] = b
    p = pd.read_csv(ROOT / "results/l1_predictions.csv")
    p = p[(p.protocol == "lopo") & ~p.project.isin(DEV_PROJECTS)] if split == "lopo" else p[(p.protocol == "ext") & p.source.isin(DALPIAZ)]
    out["B1"] = p[p.model == "B1"].set_index("req_id")
    return out


def clusters(d, split, B=10000):
    rng = np.random.default_rng(42)
    if split == "lopo":
        projs = sorted(d.project.unique()); return projs, rng.integers(0, len(projs), (B, len(projs)))
    return None, rng.integers(0, len(d), (B, len(d)))


def stats(d, split, projs, idx):
    c = M.fq_counts(d.reset_index())
    per = c.groupby(d.project.values).sum().loc[projs] if split == "lopo" else c
    point = {k: float(v[0]) for k, v in M.count_stats({k: np.array([per[k].sum()]) for k in per}).items()}
    return point, M.count_stats({k: per[k].values[idx].sum(1) for k in per})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("lopo_run"); ap.add_argument("dalpiaz_run"); ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args(); rows1, rows2 = [], []
    for split, run in (("lopo", a.lopo_run), ("dalpiaz", a.dalpiaz_run)):
        llm = llm_frame(run, a.n); bases = base_frames(split)
        projs, idx = clusters(llm, split)
        pl, sl = stats(llm, split, projs, idx)
        m = M.l1(llm.F, llm.Q, llm.v_F, llm.v_Q)
        rows1.append({"set": split, "method": "EVIDENT-RC", **{k: round(pl[k], 3) for k in ("FQ_F1", "exact_match", "FQ_FPR")},
                      "FQ_P": round(float(m["FQ_P"]), 3), "FQ_R": round(float(m["FQ_R"]), 3), "FQ_pred": m["FQ_predicted"],
                      "macroF1": round(float(m["macroF1"]), 3)})
        for k, b in bases.items():
            b = b.loc[llm.index]; assert (b.F.values == llm.F.values).all() and (b.Q.values == llm.Q.values).all()
            pb, sb = stats(b, split, projs, idx); mb = M.l1_from_frame(b)
            row = {"set": split, "method": NAMES[k], **{x: round(pb[x], 3) for x in ("FQ_F1", "exact_match", "FQ_FPR")},
                   "FQ_P": round(float(mb["FQ_P"]), 3), "FQ_R": round(float(mb["FQ_R"]), 3), "FQ_pred": mb["FQ_predicted"],
                   "macroF1": round(float(mb["macroF1"]), 3)}
            for x in ("FQ_F1", "exact_match"):
                dd = sl[x] - sb[x]; dd = dd[np.isfinite(dd)]; lo, hi = np.percentile(dd, [2.5, 97.5])
                row[f"{x}_diff"] = round(pl[x] - pb[x], 3); row[f"{x}_ci"] = f"{lo:.2f} to {hi:.2f}"
                row[f"{x}_p"] = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
            rows1.append(row)
            if k == "B1": continue
            # RQ2 end to end, item-level paired bootstrap
            ii = np.random.default_rng(7).integers(0, len(llm), (10000, len(llm)))
            for conf_col in ("conf_ts", "conf_raw"):
                q = M.confidence_quality(b[conf_col].values, b.ok.values, bins=10)
                r2 = {"set": split, "system": NAMES[k] + (" (temperature-scaled)" if conf_col == "conf_ts" else " (raw)"),
                      **{x: round(float(q[x]), 3) for x in ("accuracy", "AUROC", "ECE", "AURC", "Brier")}}
                if conf_col == "conf_ts":
                    for x, f in (("AUROC", lambda c, y: roc_auc_score(y, c)), ("ECE", None), ("AURC", M.aurc)):
                        vl = np.array([(f(llm.conf.values[i], llm.ok.values[i]) if f else M.ece(llm.conf.values[i], llm.ok.values[i])) for i in ii])
                        vb = np.array([(f(b[conf_col].values[i], b.ok.values[i]) if f else M.ece(b[conf_col].values[i], b.ok.values[i], 10)) for i in ii])
                        dd = vl - vb; r2[f"{x}_diff_ci"] = f"{np.percentile(dd, 2.5):.3f} to {np.percentile(dd, 97.5):.3f}"
                        r2[f"{x}_p"] = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
                rows2.append(r2)
        q = M.confidence_quality(llm.conf.values, llm.ok.values)
        rows2.append({"set": split, "system": "EVIDENT-RC (verified vote)", **{x: round(float(q[x]), 3) for x in ("accuracy", "AUROC", "ECE", "AURC", "Brier")}})
    r1 = pd.DataFrame(rows1)
    for x in ("FQ_F1", "exact_match"):
        for s in r1.set.unique():
            sel = (r1.set == s) & r1[f"{x}_p"].notna(); r1.loc[sel, f"{x}_p_holm"] = holm(r1.loc[sel, f"{x}_p"].values).round(4)
    r2 = pd.DataFrame(rows2)
    for x in ("AUROC", "ECE", "AURC"):
        for s in r2.set.unique():
            sel = (r2.set == s) & r2[f"{x}_p"].notna(); r2.loc[sel, f"{x}_p_holm"] = holm(r2.loc[sel, f"{x}_p"].values).round(4)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(r1.drop(columns=[c for c in r1 if c.endswith("_p")]).to_string(index=False)); print()
    print(r2.drop(columns=[c for c in r2 if c.endswith("_p")]).to_string(index=False))
    r1.to_csv(ROOT / "results/rq1_baselines.csv", index=False); r2.to_csv(ROOT / "results/rq2_baselines.csv", index=False)


if __name__ == "__main__":
    main()
