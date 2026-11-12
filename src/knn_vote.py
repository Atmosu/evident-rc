"""kNN vote baseline (first-draft review, comment 6): how far does retrieval alone get, with no LLM?

For each LOPO test item, take the same k neighbours EVIDENT-RC would put in its prompt (MiniLM cosine, training
projects of the fold only, same F+Q guarantee) and predict F and Q by a similarity-weighted vote of their gold
labels (label = 1 when the weighted share is at least 0.5). Runs k in {4, 8, 16}, with and without the F+Q
guarantee. Writes results/knn_predictions.csv (per item) and results/knn_metrics.csv.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data as D, metrics as M
from evident import lopo_split, retrieve, DEV_PROJECTS
from pilot import embed

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    P = D.promise(); E = embed(P.text.tolist()); idx = {r: i for i, r in enumerate(P.req_id)}
    items = []
    for proj in sorted(P.project.unique()):
        tr, va, te = lopo_split(P, proj)
        Etr, Ete = E[[idx[r] for r in tr.req_id]], E[[idx[r] for r in te.req_id]]
        for k in (4, 8, 16):
            for guar in (True, False):
                nb = retrieve(Ete, Etr, tr, k, fq_guarantee=guar)
                for i, r in enumerate(te.itertuples()):
                    w = np.clip(Ete[i] @ Etr[nb[i]].T, 1e-6, None)
                    vF = (w * tr.F.values[nb[i]]).sum() / w.sum(); vQ = (w * tr.Q.values[nb[i]]).sum() / w.sum()
                    items.append({"k": k, "fq_guarantee": guar, "req_id": r.req_id, "project": proj, "F": r.F, "Q": r.Q,
                                  "pred_F": int(vF >= 0.5), "pred_Q": int(vQ >= 0.5)})
    d = pd.DataFrame(items); d.to_csv(OUT / "knn_predictions.csv", index=False)
    rows = []
    d = pd.concat([d.assign(projects="all15"), d[~d.project.isin(DEV_PROJECTS)].assign(projects="eval13")])
    for (proj_set, k, guar), g in d.groupby(["projects", "k", "fq_guarantee"]):
        m = M.l1_from_frame(g); b = M.fast_cluster_bootstrap(g)
        rows.append({"projects": proj_set, "k": k, "fq_guarantee": guar, **{x: round(float(m[x]), 3) for x in
                     ("FQ_P", "FQ_R", "FQ_F1", "FQ_FPR", "exact_match", "macroF1", "F_R", "Q_R")},
                     "FQ_predicted": m["FQ_predicted"], "FQ_F1_ci": f"{b['FQ_F1'][1]:.2f} to {b['FQ_F1'][2]:.2f}",
                     "exact_match_ci": f"{b['exact_match'][1]:.2f} to {b['exact_match'][2]:.2f}"})
    res = pd.DataFrame(rows); res.to_csv(OUT / "knn_metrics.csv", index=False); print(res.to_string(index=False))


if __name__ == "__main__":
    main()
