"""Level 1 metrics from per-item pilot predictions, with the measures added after the first-draft review
(F+Q precision/recall/F1, exact match, F+Q false-positive rate, confusion matrices) and project-clustered
bootstrap 95% CIs for the LOPO results. Writes results/l1_fq_metrics.csv and results/l1_confusion.csv."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M
from evident import DEV_PROJECTS

OUT = Path(__file__).resolve().parents[1] / "results"
NAMES = {"B1": "TF-IDF + SVM", "B2": "MiniLM frozen + LR", "B3": "MiniLM fine-tuned", "Z": "Zero-shot"}
DALPIAZ = ["dronology", "leeds", "reqview", "wasp"]


def main():
    d = pd.read_csv(OUT / "l1_predictions.csv")
    sets = {"random": d[d.protocol == "random"], "lopo": d[d.protocol == "lopo"],
            "lopo-eval13": d[(d.protocol == "lopo") & ~d.project.isin(DEV_PROJECTS)],  # same projects as LLM results
            "zero-shot": d[d.protocol == "zero-shot (all PROMISE)"],
            "Dalpiaz-4": d[(d.protocol == "ext") & d.source.isin(DALPIAZ)], "iTrust": d[(d.protocol == "ext") & (d.source == "iTrust")]}
    rows, conf = [], []
    for sname, s in sets.items():
        for m, g in s.groupby("model"):
            r = {"set": sname, "model": m, "name": NAMES[m], **M.l1_from_frame(g)}
            if sname in ("lopo", "lopo-eval13", "Dalpiaz-4"):  # cluster = project (LOPO) or source dataset (Dalpiaz-4)
                b = M.fast_cluster_bootstrap(g)
                for key in ("FQ_F1", "exact_match"):
                    r[f"{key}_ci"] = f"{b[key][1]:.2f} to {b[key][2]:.2f}"
            rows.append(r)
            c = M.confusion(g.F, g.Q, g.pred_F, g.pred_Q).reset_index().melt(id_vars="gold", value_name="count")
            conf.append(c.assign(set=sname, model=m))
    res = pd.DataFrame(rows); res.to_csv(OUT / "l1_fq_metrics.csv", index=False)
    pd.concat(conf).to_csv(OUT / "l1_confusion.csv", index=False)
    cols = ["set", "name", "FQ_predicted", "FQ_gold", "FQ_P", "FQ_R", "FQ_F1", "FQ_F1_ci", "FQ_FPR", "exact_match", "exact_match_ci", "macroF1"]
    print(res[[c for c in cols if c in res]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
