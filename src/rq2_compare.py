"""RQ2 comparison on the 13 evaluation projects (530 items). Correct = full tuple (F, Q, attribute) matches gold.

(a) Confidence quality on identical predictions (the EVIDENT-RC verified vote at n samples):
    c_verified (EVIDENT-RC), support_all (plain agreement), 1 - vote entropy (verified and plain), and
    S3 = mean of c_verified and the stated confidence of the verbal run.
(b) End to end: each system with its own predictions and own confidence:
    EVIDENT-RC (verified vote, c), S1 plain self-consistency (plain vote, its share), S2 verbalized (one sample).
Paired bootstrap over items (10,000 resamples) of the difference to EVIDENT-RC, Holm within each metric family.
The temperature-scaled fine-tuned baseline is added once its probabilities exist.

Usage: python src/rq2_compare.py <main_run> <verbal_run> [--n 5] [--out results/rq2_compare.csv]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, analyze_llm as A
from compare_l1 import holm

FAST = {"AUROC": lambda c, y: roc_auc_score(y, c), "AURC": M.aurc}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("main_run"); ap.add_argument("verbal_run")
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--B", type=int, default=10000); ap.add_argument("--out")
    a = ap.parse_args()
    d = A.l1_frame(A.load(a.main_run), a.n).set_index("req_id")
    vb = {r["req_id"]: r for r in A.load(a.verbal_run)}
    gold = {r["req_id"]: (r["gold"]["F"], r["gold"]["Q"], r["gold"]["attribute"]) for r in vb.values()}
    def verbal(rid):
        r = vb[rid]; s = r["stated_confidence"][0]
        t = tuple(r["plain_vote"]["tuple"]) if r["plain_vote"]["tuple"] else None
        return (s if s is not None else 0.0), A.tuple_ok(t, gold[rid])
    S2 = pd.DataFrame([verbal(i) for i in d.index], index=d.index, columns=["conf", "ok"])
    y = d.v_tuple_ok.values.astype(bool)
    same = {"EVIDENT-RC c_verified": (d.sig_c_verified.values, None), "plain agreement (support_all)": (d.sig_support_all.values, None),
            "1 - vote entropy (verified)": (d["sig_1-H_verified"].values, 10), "1 - vote entropy (plain)": (d["sig_1-H_plain"].values, 10),
            "S3 mean(c_verified, stated)": ((d.sig_c_verified.values + S2.conf.values) / 2, 10)}
    e2e = {"EVIDENT-RC (verified vote, c)": (d.v_c.values, d.v_tuple_ok.values.astype(bool), None),
           "S1 plain self-consistency": (d.p_c.values, d.p_tuple_ok.values.astype(bool), None),
           "S2 verbalized (1 sample)": (S2.conf.values, S2.ok.values.astype(bool), 10)}
    rng = np.random.default_rng(42); N = len(d); idx = rng.integers(0, N, size=(a.B, N))
    def boot(conf, ok, bins):
        return {"AUROC": np.array([FAST["AUROC"](conf[i], ok[i]) for i in idx]),
                "ECE": np.array([M.ece(conf[i], ok[i], bins) for i in idx]),
                "AURC": np.array([M.aurc(conf[i], ok[i]) for i in idx])}
    rows = []
    for fam, systems in (("identical predictions", {k: (c, y, b) for k, (c, b) in same.items()}), ("end to end", e2e)):
        names = list(systems); ref = names[0]; bs = {k: boot(*systems[k]) for k in names}
        for k in names:
            q = M.confidence_quality(systems[k][0], systems[k][1], systems[k][2])
            row = {"family": fam, "signal": k, "accuracy": round(q["accuracy"], 3), **{m: round(float(q[m]), 3) for m in ("AUROC", "ECE", "AURC", "Brier", "agree_but_wrong")}}
            for m in ("AUROC", "ECE", "AURC"):
                if k == ref: continue
                diff = bs[ref][m] - bs[k][m]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                row[f"{m}_diff_CI"] = f"{lo:.3f} to {hi:.3f}"
                row[f"{m}_p"] = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
            rows.append(row)
    r = pd.DataFrame(rows)
    for fam in r.family.unique():
        for m in ("AUROC", "ECE", "AURC"):
            sel = (r.family == fam) & r[f"{m}_p"].notna()
            r.loc[sel, f"{m}_p_holm"] = holm(r.loc[sel, f"{m}_p"].values).round(4)
    pd.set_option("display.width", 250); print(f"n = {a.n}, items {N}, diff = EVIDENT-RC minus signal (AUROC: >0 favours EVIDENT-RC; ECE, AURC: <0 favours it)")
    print(r.to_string(index=False))
    if a.out: r.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
