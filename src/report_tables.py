"""Every number and figure of the results-analysis report, recomputed from the per-item result files.
Writes results/report/tables.json and report/figures/{reliability,risk_coverage}.png.
Usage: python src/report_tables.py"""
import glob, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, analyze_llm as A, evident as ev
from compare_l1 import holm
from evident import DEV_PROJECTS

ROOT = Path(__file__).resolve().parents[1]
RES, EV = ROOT / "results", ROOT / "results/evident"
FIG = ROOT / "results/report/figures"; FIG.mkdir(parents=True, exist_ok=True)
(RES / "report").mkdir(exist_ok=True)
R = "Llama-3.1-8B-Instruct-bnb-4bit_hf_"
RUN = {"main": R + "k8_n10_T0.7_s42_p443b9c_constrained_q4_eval", "dal": R + "k8_n10_T0.7_s42_p443b9c_constrained_q4_dalpiaz",
       "l0": R + "k8_n5_T0.7_s42_p65a137_constrained_q4_noevidence_eval", "l0_dal": R + "k8_n5_T0.7_s42_p65a137_constrained_q4_noevidence_dalpiaz",
       "a1": R + "k0_n5_T0.7_s42_p443b9c_constrained_q4_eval", "a1_dal": R + "k0_n5_T0.7_s42_p443b9c_constrained_q4_dalpiaz",
       "a2": R + "k8_n5_T0.7_s42_p443b9c_constrained_q4_noFQ_eval", "itrust": R + "k8_n5_T0.7_s42_p443b9c_constrained_q4_itrust",
       "verbal": R + "k8_n1_T0.7_s42_p6b8f57_constrained_q4_verbal_eval", "verbal_dal": R + "k8_n1_T0.7_s42_p6b8f57_constrained_q4_verbal_dalpiaz",
       "free": R + "k8_n5_T0.7_s42_p443b9c_free_q4_eval", "pc": R + "k8_n10_T0.7_s42_p443b9c_constrained_q4_postcutoff",
       "s43": R + "k8_n5_T0.7_s43_p443b9c_constrained_q4_eval", "s44": R + "k8_n5_T0.7_s44_p443b9c_constrained_q4_eval"}
DAL = ["dronology", "leeds", "reqview", "wasp"]
r3 = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


def load(run): return [json.loads(l) for f in sorted((EV / run).glob("*.jsonl")) for l in open(f)]


def llm(run, variant=False):
    recs = load(run)
    if variant or "noevidence" in run:
        d = pd.DataFrame([{"req_id": r["req_id"], "project": r["project"], "F": r["gold"]["F"], "Q": r["gold"]["Q"],
                           "v_none": r["F"] is None, "v_F": r["F"]["label"] if r["F"] else 0,
                           "v_Q": r["Q"]["label"] if r["Q"] else 0} for r in recs])
    else:
        d = A.l1_frame(recs, 5)
    d["pred_F"] = np.where(d.v_none, -1, d.v_F); d["pred_Q"] = np.where(d.v_none, -1, d.v_Q)
    return d.set_index("req_id")


def pilot(protocol, model, src=None):
    p = pd.read_csv(RES / "l1_predictions.csv")
    p = p[(p.protocol == protocol) & (p.model == model)]
    if src: p = p[p.source.isin(src)]
    return p.set_index("req_id")


def tb(name, split): return pd.read_csv(RES / f"baselines/{name}_{split}.csv").set_index("req_id")


def knn():
    k = pd.read_csv(RES / "knn_predictions.csv"); k = k[(k.k == 8) & (k.fq_guarantee) & ~k.project.isin(DEV_PROJECTS)]
    return k.set_index("req_id")


def point(d):
    c = M.fq_counts(d.reset_index()); pt = M.count_stats({k: np.array([v.sum()]) for k, v in c.items()})
    m = M.l1(d.F, d.Q, d.pred_F.clip(0), d.pred_Q.clip(0))
    return {"FQ_F1": r3(pt["FQ_F1"][0]), "exact": r3(pt["exact_match"][0]), "FQ_FPR": r3(pt["FQ_FPR"][0]),
            "FQ_P": r3(m["FQ_P"]), "FQ_R": r3(m["FQ_R"]), "FQ_pred": int(c.pred.sum()), "FQ_found": int(c.tp.sum()),
            "FQ_gold": int(c.gold.sum()), "macroF1": r3(m["macroF1"]), "n": len(d)}


def boot_stats(d, idx, clusters=None):
    c = M.fq_counts(d.reset_index())
    if clusters is not None: c = c.groupby(d.project.values).sum().loc[clusters]
    return M.count_stats({k: v.values[idx].sum(1) for k, v in c.items()})


def rq1_table(systems, ref, clustered, family):
    idxs = list(systems[ref].index)
    for k in systems: systems[k] = systems[k].loc[idxs]
    rng = np.random.default_rng(42)
    if clustered:
        projs = sorted(systems[ref].project.unique()); idx = rng.integers(0, len(projs), (10000, len(projs)))
    else:
        projs = None; idx = rng.integers(0, len(idxs), (10000, len(idxs)))
    bs = {k: boot_stats(d, idx, projs) for k, d in systems.items()}
    rows = {}
    for k, d in systems.items():
        row = point(d)
        for x in ("FQ_F1", "exact_match"):
            v = bs[k][x]; v = v[np.isfinite(v)]; lo, hi = np.percentile(v, [2.5, 97.5])
            row[f"{x}_ci"] = [r3(lo), r3(hi)]
            if k != ref:
                dd = bs[ref][x] - bs[k][x]; dd = dd[np.isfinite(dd)]; lo, hi = np.percentile(dd, [2.5, 97.5])
                row[f"{x}_diff"] = [r3(lo), r3(hi)]; row[f"{x}_p"] = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
        rows[k] = row
    for x in ("FQ_F1", "exact_match"):
        ks = [k for k in family if k in rows]; ps = holm([rows[k][f"{x}_p"] for k in ks])
        for k, p in zip(ks, ps): rows[k][f"{x}_p_holm"] = r3(p)
    return rows


def confusion(d):
    ans = d[~d.v_none]
    cm = M.confusion(ans.F, ans.Q, ans.v_F, ans.v_Q)
    out = {g: {p: int(cm.loc[g, p]) if p in cm.columns and g in cm.index else 0 for p in ("F only", "Q only", "F+Q")}
           for g in ("F only", "Q only", "F+Q", "neither")}
    ab = d[d.v_none]
    for g in out:
        out[g]["no answer"] = int(sum(1 for f, q in zip(ab.F, ab.Q) if M.combo(np.array([f]), np.array([q]))[0] == g))
    return out


def main():
    T = {}
    # ---------- RQ1: PROMISE 13 projects ----------
    pro = {"EVIDENT-RC": llm(RUN["main"]), "L0 no quotes": llm(RUN["l0"]), "A1 no retrieval": llm(RUN["a1"]),
           "A2 no forced F+Q example": llm(RUN["a2"]), "RoBERTa-large": tb("roberta", "lopo"), "BERT-base": tb("bert", "lopo"),
           "MiniLM fine-tuned": tb("minilm", "lopo"),
           "Frozen MiniLM + LR": pilot("lopo", "B2").pipe(lambda x: x[~x.project.isin(DEV_PROJECTS)]),
           "kNN vote (k = 8)": knn(), "TF-IDF + SVM": pilot("lopo", "B1").pipe(lambda x: x[~x.project.isin(DEV_PROJECTS)])}
    T["promise"] = rq1_table(pro, "EVIDENT-RC", True, ["RoBERTa-large", "BERT-base", "MiniLM fine-tuned", "TF-IDF + SVM"])
    dal = {"EVIDENT-RC": llm(RUN["dal"]), "L0 no quotes": llm(RUN["l0_dal"]), "A1 no retrieval": llm(RUN["a1_dal"]),
           "RoBERTa-large": tb("roberta", "dalpiaz"), "BERT-base": tb("bert", "dalpiaz"), "MiniLM fine-tuned": tb("minilm", "dalpiaz"),
           "Frozen MiniLM + LR": pilot("ext", "B2", DAL), "TF-IDF + SVM": pilot("ext", "B1", DAL)}
    T["dalpiaz"] = rq1_table(dal, "EVIDENT-RC", False, ["RoBERTa-large", "BERT-base", "MiniLM fine-tuned", "TF-IDF + SVM"])
    pc = {"EVIDENT-RC": llm(RUN["pc"]), "L0 no quotes": llm(next(EV.glob("*_noevidence_postcutoff")).name),
          "RoBERTa-large": tb("roberta", "postcutoff"), "BERT-base": tb("bert", "postcutoff"),
          "MiniLM fine-tuned": tb("minilm", "postcutoff"), "TF-IDF + SVM": tb("tfidf", "postcutoff")}
    T["postcutoff"] = rq1_table(pc, "EVIDENT-RC", False, ["L0 no quotes", "RoBERTa-large", "BERT-base", "MiniLM fine-tuned", "TF-IDF + SVM"])
    # ---------- iTrust ----------
    it = {"EVIDENT-RC": llm(RUN["itrust"]), "RoBERTa-large": tb("roberta", "itrust"), "BERT-base": tb("bert", "itrust"),
          "MiniLM fine-tuned": tb("minilm", "itrust"), "Frozen MiniLM + LR": pilot("ext", "B2", ["iTrust"]),
          "TF-IDF + SVM": pilot("ext", "B1", ["iTrust"])}
    T["itrust"] = {}
    for k, d in it.items():
        F, Q, pF, pQ = d.F.values, d.Q.values, d.pred_F.values, d.pred_Q.values
        tp = ((F == 1) & (pF == 1)).sum()
        T["itrust"][k] = {"F_recall": r3(tp / (F == 1).sum()), "F_precision": r3(tp / max((pF == 1).sum(), 1)),
                          "false_Q": int(((Q == 0) & (pQ == 1)).sum()), "exact": r3(((F == pF) & (Q == pQ)).mean()), "n": len(d)}
    # ---------- per project ----------
    e, rb = pro["EVIDENT-RC"], pro["RoBERTa-large"]
    T["per_project"] = []
    for p in sorted(e.project.unique()):
        a, b = e[e.project == p], rb[rb.project == p]
        ex = lambda d: ((d.F == d.pred_F) & (d.Q == d.pred_Q)).mean()
        fq = lambda d: int(((d.F == 1) & (d.Q == 1) & (d.pred_F == 1) & (d.pred_Q == 1)).sum())
        T["per_project"].append({"project": p, "n": len(a), "FQ_gold": int(((a.F == 1) & (a.Q == 1)).sum()),
                                 "exact_ev": r3(ex(a)), "exact_rb": r3(ex(b)), "found_ev": fq(a), "found_rb": fq(b),
                                 "false_fq_ev": int(((a.pred_F == 1) & (a.pred_Q == 1) & ~((a.F == 1) & (a.Q == 1))).sum())})
    # ---------- errors ----------
    T["confusion"] = {"promise": confusion(e), "dalpiaz": confusion(dal["EVIDENT-RC"]), "postcutoff": confusion(pc["EVIDENT-RC"])}
    recs = {r["req_id"]: r for r in load(RUN["main"])}
    smp = lambda r: [{"obj": (o := ev.parse(s["raw"])), "verified": ev.verify(o, r["text"])[0]} for s in r["samples"][:5]]
    qonly = Counter(); ex_fp, ex_fn = [], []
    for rid, r in recs.items():
        vt, _ = ev.vote(smp(r), 5); g = r["gold"]
        if vt and vt[:2] == (1, 1) and (g["F"], g["Q"]) == (0, 1):
            qonly[g["attribute"] or "no attribute"] += 1
            best = next(o for o in (x["obj"] for x in smp(r)) if o and ev.tuple_of(o) == vt)
            ex_fp.append({"id": rid, "text": r["text"], "type": g["attribute"], "F_quote": best.get("F_span")})
        if g["F"] == 1 and g["Q"] == 1 and vt and vt[:2] != (1, 1):
            ex_fn.append({"id": rid, "text": r["text"], "pred": "F only" if vt[0] else "Q only"})
    T["false_fq_qonly_by_type"] = dict(qonly.most_common())
    T["examples_false_fq"] = sorted(ex_fp, key=lambda x: len(x["text"]))[:6]
    T["examples_missed_fq"] = sorted(ex_fn, key=lambda x: len(x["text"]))[:4]
    # ---------- RQ2 ----------
    T["rq2_identical"] = pd.read_csv(RES / "rq2_compare_n5.csv").astype(object).where(lambda x: x.notna(), None).to_dict("records")
    T["rq2_baselines"] = pd.read_csv(RES / "rq2_baselines.csv").astype(object).where(lambda x: x.notna(), None).to_dict("records")
    T["rq2_postcutoff"] = pd.read_csv(RES / "postcutoff_rq2.csv").astype(object).where(lambda x: x.notna(), None).to_dict("records")
    dd = A.l1_frame(load(RUN["dal"]), 5)
    vb = {r["req_id"]: r for r in load(RUN["verbal_dal"])}
    g = {r["req_id"]: (r["gold"]["F"], r["gold"]["Q"], r["gold"]["attribute"]) for r in vb.values()}
    s2c = np.array([vb[i]["stated_confidence"][0] or 0.0 for i in dd.req_id])
    s2ok = np.array([A.tuple_ok(tuple(vb[i]["plain_vote"]["tuple"]) if vb[i]["plain_vote"]["tuple"] else None, g[i]) for i in dd.req_id])
    T["rq2_dalpiaz_llm"] = {"EVIDENT-RC": {k: r3(v) for k, v in M.confidence_quality(dd.v_c, dd.v_tuple_ok).items()},
                            "S1 plain voting": {k: r3(v) for k, v in M.confidence_quality(dd.p_c, dd.p_tuple_ok).items()},
                            "S2 stated": {k: r3(v) for k, v in M.confidence_quality(s2c, s2ok, bins=10).items()}}
    # ---------- robustness ----------
    T["faithfulness"] = json.loads((EV / RUN["main"] / "faithfulness/summary.json").read_text())["overall"]
    T["seeds"] = {}
    for s, run in (("42", RUN["main"]), ("43", RUN["s43"]), ("44", RUN["s44"])):
        x = llm(run); q = M.confidence_quality(x.v_c, x.v_tuple_ok)
        T["seeds"][s] = {**{k: point(x)[k] for k in ("exact", "FQ_F1", "FQ_FPR", "FQ_pred")}, "AUROC": r3(q["AUROC"]), "ECE": r3(q["ECE"])}
    fr = load(RUN["free"]); s = [x for r in fr for x in r["samples"]]
    T["free"] = {"items": len(fr), "samples": len(s), "invalid_json": sum(x["reason"] in ("json", "schema") for x in s),
                 "no_answer": sum(r["F"] is None for r in fr)}
    T["memorisation"] = {**json.loads((RES / "contamination/summary.json").read_text())["sets"],
                         **json.loads((RES / "contamination/summary_PostCutoff.json").read_text())["sets"]}
    sm = [x for r in load(RUN["main"]) for x in r["samples"][:5]]
    T["v1_pass"] = r3(np.mean([x["verified"] for x in sm]))
    m5 = A.l1_frame(load(RUN["main"]), 5); T["same_vote"] = r3(m5.same.mean())
    # ---------- figures ----------
    figures(e, m5, rb)
    (RES / "report/tables.json").write_text(json.dumps(T, indent=1, default=str))
    print("ok", list(T))


def figures(e, m5, rb):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
    rb = rb.loc[m5.req_id]
    ga = rb.attr.where(rb.Q == 1); pa = rb.pred_attr.where(rb.pred_Q == 1)
    rb_ok = ((rb.F == rb.pred_F) & (rb.Q == rb.pred_Q) & (ga.isna() | (ga == pa))).values
    series = [("EVIDENT-RC (checked votes)", m5.v_c.values, m5.v_tuple_ok.values.astype(bool), None, C1, "o"),
              ("Plain voting", m5.p_c.values, m5.p_tuple_ok.values.astype(bool), None, C2, "s"),
              ("RoBERTa-large, temperature-scaled", rb.conf_ts.values, rb_ok, 10, C3, "^")]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--", zorder=1)
    ax.text(0.83, 0.88, "perfect calibration", color="#898781", fontsize=9, rotation=34)
    for name, c, ok, bins, col, mk in series:
        idx = np.unique(c, return_inverse=True)[1] if bins is None else np.minimum((c * bins).astype(int), bins - 1)
        xs, ys, ns = [], [], []
        for b in np.unique(idx):
            sel = idx == b
            if sel.sum() < 5: continue
            xs.append(c[sel].mean()); ys.append(ok[sel].mean()); ns.append(sel.sum())
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.2, label=name, zorder=3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xlabel("Stated confidence (bin mean)"); ax.set_ylabel("Share of answers that are correct")
    ax.grid(color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "reliability.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
    offs = {"EVIDENT-RC": -9, "Plain voting": 9, "RoBERTa-large": 0}
    for name, c, ok, bins, col, mk in series:
        ts = np.unique(c)[::-1]
        if bins is not None: ts = np.sort(c)[::-1][9:]          # continuous: every item from the 10th on
        cov = np.array([(c >= t).mean() for t in ts]); risk = np.array([1 - ok[c >= t].mean() for t in ts])
        ax.plot(cov, risk, color=col, lw=2, marker=mk if bins is None else None, ms=6, mec="white", mew=1.2,
                label=name, drawstyle="default")
        short = name.split(" (")[0].split(",")[0]
        ax.annotate(short, (cov[-1], risk[-1]), xytext=(6, offs[short]), textcoords="offset points", color="#52514e", fontsize=9, va="center")
    ax.axvline(0.85, color="#c3c2b7", lw=1, ls="--"); ax.text(0.855, 0.03, "85% coverage", color="#898781", fontsize=9)
    ax.set_xlim(0, 1.22); ax.set_xticks([0,.2,.4,.6,.8,1]); ax.set_ylim(0, 0.6); ax.set_xlabel("Coverage (share of items answered, most confident first)"); ax.set_ylabel("Error rate among answered items")
    ax.grid(color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "risk_coverage.png"); plt.close(fig)


if __name__ == "__main__":
    main()
