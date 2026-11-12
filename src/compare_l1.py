"""RQ1 comparison on the 13 evaluation projects: EVIDENT-RC (verified vote) against each baseline, pooled over
LOPO folds, with paired project-clustered bootstrap 95% CIs of the difference (both systems see the same
resampled projects) and Holm correction per metric family. An LLM item with no verified sample is an abstention:
it never counts as F+Q and its exact match is wrong.

Usage: python src/compare_l1.py <run_dir> [--n 5] [--out results/rq1_compare.csv]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, analyze_llm as A
from evident import DEV_PROJECTS

NAMES = {"B1": "TF-IDF + SVM", "B2": "MiniLM frozen + LR", "B3": "MiniLM fine-tuned"}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None); ap.add_argument("--B", type=int, default=10000); a = ap.parse_args()
    meta = json.loads((Path(a.run_dir) / "run_meta.json").read_text())
    if meta.get("variant", "evident") == "evident":
        llm = A.l1_frame(A.load(a.run_dir), a.n)
    else:  # comparison variants: the run's own vote (no-evidence: format checks only), all its samples
        recs = A.load(a.run_dir)
        llm = pd.DataFrame([{"req_id": r["req_id"], "project": r["project"], "F": r["gold"]["F"], "Q": r["gold"]["Q"],
                             "v_none": r["F"] is None, "v_F": r["F"]["label"] if r["F"] else 0,
                             "v_Q": r["Q"]["label"] if r["Q"] else 0} for r in recs])
    llm["pred_F"] = np.where(llm.v_none, -1, llm.v_F); llm["pred_Q"] = np.where(llm.v_none, -1, llm.v_Q)
    m = M.l1(llm.F, llm.Q, llm.v_F, llm.v_Q)
    print("macro-F1", round(float(m["macroF1"]), 3), "F+Q predicted", m["FQ_predicted"], "P", round(float(m["FQ_P"]), 3),
          "R", round(float(m["FQ_R"]), 3))
    base = pd.read_csv(Path(__file__).resolve().parents[1] / "results/l1_predictions.csv")
    base = base[(base.protocol == "lopo") & ~base.project.isin(DEV_PROJECTS)]
    rows, rng = [], np.random.default_rng(42)
    projs = sorted(llm.project.unique()); idx = rng.integers(0, len(projs), size=(a.B, len(projs)))
    per_l = M.fq_counts(llm).groupby(llm.project.values).sum().loc[projs]
    sl = {k: per_l[k].values[idx].sum(1) for k in per_l}; boot_l = M.count_stats(sl)
    pt_l = M.count_stats({k: np.array([per_l[k].sum()]) for k in per_l})
    for m, g in base.groupby("model"):
        g = g.set_index("req_id").loc[llm.req_id]
        assert (g.F.values == llm.F.values).all() and (g.Q.values == llm.Q.values).all()
        per_b = M.fq_counts(g.reset_index()).groupby(g.project.values).sum().loc[projs]
        boot_b = M.count_stats({k: per_b[k].values[idx].sum(1) for k in per_b})
        pt_b = M.count_stats({k: np.array([per_b[k].sum()]) for k in per_b})
        for key in ("FQ_F1", "exact_match", "FQ_FPR"):
            d = boot_l[key] - boot_b[key]; d = d[np.isfinite(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))  # two-sided bootstrap p-value
            rows.append({"metric": key, "baseline": NAMES[m], "EVIDENT-RC": round(float(pt_l[key][0]), 3),
                         "baseline_value": round(float(pt_b[key][0]), 3), "difference": round(float(pt_l[key][0] - pt_b[key][0]), 3),
                         "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3), "p_boot": round(p, 4)})
    r = pd.DataFrame(rows)
    r["p_holm"] = r.groupby("metric").p_boot.transform(lambda s: holm(s.values)).round(4)
    r["CI_excludes_0"] = (r.ci_low > 0) | (r.ci_high < 0)
    print(f"items {len(llm)}, projects {len(projs)}, n = {a.n}, abstained {int(llm.v_none.sum())}")
    print(r.to_string(index=False))
    if a.out: r.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
