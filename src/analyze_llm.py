"""Analyse an EVIDENT-RC run directory (per-item JSONL files) for the 2-fold LLM pilot.

Reports, per run and pooled over its folds:
  V1 pass rate with and without the span rules, and the reasons samples fail
  verified vote vs plain vote (self-consistency): how often they predict the same tuple, and the accuracy of each
  Level 1 metrics for both votes (F+Q precision/recall/F1, exact match, F+Q false-positive rate, confusion matrix)
  confidence quality: AUROC for error detection on each vote's own predictions, and distinct confidence values
  subsampling: the same measures using only the first 5, 10, ... samples of each item

Usage: python src/analyze_llm.py results/evident/<run_dir> [--out report.json]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import evident as ev, metrics as M


def load(run_dir):
    return [json.loads(l) for f in sorted(Path(run_dir).glob("P*.jsonl")) for l in open(f)]


def votes(rec, n_sub):
    """Re-vote one item using only its first n_sub samples (raw outputs are re-parsed and re-verified).
    Also returns the confidence signals compared in RQ2, all scored against the verified vote's prediction."""
    smp = []
    for s in rec["samples"][:n_sub]:
        obj = ev.parse(s["raw"]); ok, _ = ev.verify(obj, rec["text"])
        smp.append({"obj": obj, "verified": ok})
    (vt, vc), (pt, pc) = ev.vote(smp, n_sub), ev.plain_vote(smp, n_sub)
    valid = [ev.tuple_of(x["obj"]) for x in smp if ev.schema_ok(x["obj"])]
    ver = [ev.tuple_of(x["obj"]) for x in smp if x["verified"]]
    sig = {"c_verified": vc,                                   # EVIDENT-RC confidence
           "support_all": valid.count(vt) / n_sub if vt else 0.0,  # same answer, unverified samples counted too
           "1-H_plain": 1 - M.vote_entropy(valid, n_sub),      # vote entropy over all answers
           "1-H_verified": 1 - M.vote_entropy(ver, n_sub, unverified=n_sub - len(ver))}  # each failure its own outcome
    return (vt, vc), (pt, pc), sig


def tuple_ok(t, g):
    """F and Q must match; the attribute is scored only when a gold attribute exists (Dalpiaz-4 and 24 PROMISE items have none)."""
    return t is not None and tuple(t[:2]) == tuple(g[:2]) and (g[2] is None or t[2] == g[2])


def l1_frame(recs, n_sub):
    rows = []
    for r in recs:
        (vt, vc), (pt, pc), sig = votes(r, n_sub)
        g = (r["gold"]["F"], r["gold"]["Q"], r["gold"]["attribute"])
        rows.append({"req_id": r["req_id"], "project": r["project"], "F": g[0], "Q": g[1],
                     "v_F": vt[0] if vt else 0, "v_Q": vt[1] if vt else 0, "v_c": vc, "v_tuple_ok": tuple_ok(vt, g),
                     "p_F": pt[0] if pt else 0, "p_Q": pt[1] if pt else 0, "p_c": pc, "p_tuple_ok": tuple_ok(pt, g),
                     "same": vt == pt, "v_none": vt is None, "p_none": pt is None, **{f"sig_{k}": v for k, v in sig.items()}})
    return pd.DataFrame(rows)


def auroc(correct, conf):
    """AUROC of confidence for separating correct from wrong predictions (0.5 = no better than chance)."""
    correct = np.asarray(correct, bool)
    return round(float(roc_auc_score(correct, conf)), 3) if 0 < correct.sum() < len(correct) else None


def summary(recs, n_sub):
    d = l1_frame(recs, n_sub)
    out = {"n_samples": n_sub, "items": len(d), "items_without_verified_sample": int(d.v_none.sum()),
           "same_prediction_verified_vs_plain": round(float(d.same.mean()), 3)}
    for tag, pre in (("verified_vote", "v"), ("plain_vote", "p")):
        # An item with no usable sample is an abstention: it never predicts F+Q and its exact match counts as wrong.
        m = M.l1(d.F, d.Q, d[f"{pre}_F"], d[f"{pre}_Q"])
        l1_ok = (d.F == d[f"{pre}_F"]) & (d.Q == d[f"{pre}_Q"]) & ~d[f"{pre}_none"]
        m["exact_match"] = l1_ok.mean(); m["coverage"] = 1 - d[f"{pre}_none"].mean()
        out[tag] = {**{k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in m.items()},
                    "tuple_exact": round(float(d[f"{pre}_tuple_ok"].mean()), 3),
                    "auroc_L1_errors": auroc(l1_ok, d[f"{pre}_c"]),
                    "auroc_tuple_errors": auroc(d[f"{pre}_tuple_ok"], d[f"{pre}_c"]),
                    "distinct_confidence_values": int(d[f"{pre}_c"].nunique())}
    # RQ2 confidence quality: every signal judged on the SAME predictions (the verified vote's), correct = full tuple
    rnd = lambda q: {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in q.items()}
    out["confidence_quality_on_verified_predictions"] = {
        k[4:]: rnd(M.confidence_quality(d[k], d.v_tuple_ok, bins=None if k == "sig_c_verified" or k == "sig_support_all" else 10))
        for k in d if k.startswith("sig_")}
    # RQ2 end to end: each complete system with its own predictions and its own confidence
    out["end_to_end"] = {"verified_vote_system": rnd(M.confidence_quality(d.v_c, d.v_tuple_ok)),
                         "plain_vote_system": rnd(M.confidence_quality(d.p_c, d.p_tuple_ok))}
    ans = d[~d.v_none]  # rows = gold, columns = predicted; abstentions listed separately
    out["confusion_verified_vote"] = {"rows_gold_cols_pred": M.confusion(ans.F, ans.Q, ans.v_F, ans.v_Q).to_dict("index"),
                                      "abstained_by_gold": d[d.v_none].pipe(lambda x: dict(zip(*np.unique(M.combo(x.F, x.Q), return_counts=True)))) if d.v_none.any() else {}}
    return out


def probe_effect(recs):
    """Per-aspect probe (ablation): add an aspect the verified vote left out when more than half of the probe's n
    samples answer 'present' with a verified span. Reports omissions recovered and false additions."""
    rows = []
    for r in recs:
        if "probe" not in r: return None
        smp = [{"obj": (o := ev.parse(s["raw"])), "verified": ev.verify(o, r["text"])[0]} for s in r["samples"]]
        vt, _ = ev.vote(smp, len(smp)); mF, mQ = (vt[0], vt[1]) if vt else (0, 0)
        add = {lab: r["probe"][lab]["present"] > r["probe"][lab]["n"] / 2 for lab in ("F", "Q")}
        rows.append({"project": r["project"], "F": r["gold"]["F"], "Q": r["gold"]["Q"], "m_F": mF, "m_Q": mQ,
                     "p_F": int(mF or add["F"]), "p_Q": int(mQ or add["Q"]), "abstained": vt is None})
    d = pd.DataFrame(rows); out = {}
    for lab in ("F", "Q"):
        out[f"{lab}_omissions_by_main"] = int(((d[lab] == 1) & (d[f"m_{lab}"] == 0)).sum())
        out[f"{lab}_recovered_by_probe"] = int(((d[lab] == 1) & (d[f"m_{lab}"] == 0) & (d[f"p_{lab}"] == 1)).sum())
        out[f"{lab}_false_additions"] = int(((d[lab] == 0) & (d[f"m_{lab}"] == 0) & (d[f"p_{lab}"] == 1)).sum())
    for tag, pre in (("main", "m"), ("main_plus_probe", "p")):
        m = M.l1(d.F, d.Q, d[f"{pre}_F"], d[f"{pre}_Q"])
        out[tag] = {k: round(float(m[k]), 3) for k in ("FQ_P", "FQ_R", "FQ_F1", "FQ_FPR", "exact_match", "macroF1")} | {"FQ_predicted": m["FQ_predicted"]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = load(a.run_dir); n = len(recs[0]["samples"])
    smp = [s for r in recs for s in r["samples"]]
    rep = {"run": Path(a.run_dir).name, "folds": sorted({r["project"] for r in recs}), "items": len(recs), "n": n,
           "v1_pass_rate_with_span_rules": round(float(np.mean([s["verified"] for s in smp])), 3),
           "v1_pass_rate_basic_rules": round(float(np.mean([s["verified_basic"] for s in smp])), 3),
           "fail_reasons": dict(Counter(s["reason"] for s in smp if not s["verified"]).most_common()),
           "mean_span_agreement": {lab: round(float(np.mean(v)), 3) if (v := [r["span_agreement"][lab] for r in recs
                                    if r["span_agreement"][lab] is not None]) else None for lab in ("F", "Q")},
           "by_n": [summary(recs, k) for k in sorted({5, 10, n}) if k <= n],
           "probe": probe_effect(recs)}
    txt = json.dumps(rep, indent=1, default=str); print(txt)
    if a.out: Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
