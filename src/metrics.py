"""Shared evaluation metrics, computed from per-item predictions (Level 1 for RQ1, confidence measures for RQ2).

Level 1 (F, Q as two binaries)
  per-label P/R/F1, macro-F1, Hamming loss, exact match (both labels right, all items),
  F+Q class P/R/F1 (both-correct rate = F+Q recall), F+Q false-positive rate, 4x4 confusion matrix

Uncertainty
  project-clustered bootstrap: resample whole projects with replacement, never single items,
  because requirements from one project are not independent
"""
from collections import Counter
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, hamming_loss, roc_auc_score

COMBOS = ["F only", "Q only", "F+Q", "neither"]


def combo(F, Q):
    F, Q = np.asarray(F), np.asarray(Q)
    return np.select([(F == 1) & (Q == 0), (F == 0) & (Q == 1), (F == 1) & (Q == 1)], COMBOS[:3], "neither")


def confusion(gF, gQ, pF, pQ):
    """Rows = gold combination, columns = predicted combination."""
    return pd.crosstab(pd.Categorical(combo(gF, gQ), COMBOS), pd.Categorical(combo(pF, pQ), COMBOS),
                       rownames=["gold"], colnames=["pred"], dropna=False)


def l1(gF, gQ, pF, pQ):
    gF, gQ, pF, pQ = map(np.asarray, (gF, gQ, pF, pQ))
    Y, P = np.column_stack([gF, gQ]), np.column_stack([pF, pQ])
    p, r, f, _ = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    g_fq, p_fq = (gF == 1) & (gQ == 1), (pF == 1) & (pQ == 1)
    tp = (g_fq & p_fq).sum()
    fq_p = tp / p_fq.sum() if p_fq.sum() else 0.0
    fq_r = tp / g_fq.sum() if g_fq.sum() else np.nan
    fq_f1 = 2 * fq_p * fq_r / (fq_p + fq_r) if fq_p + fq_r > 0 else 0.0
    return {"F_P": p[0], "F_R": r[0], "F_F1": f[0], "Q_P": p[1], "Q_R": r[1], "Q_F1": f[1],
            "macroF1": f1_score(Y, P, average="macro", zero_division=0), "hamming": hamming_loss(Y, P),
            "exact_match": ((gF == pF) & (gQ == pQ)).mean(),
            "FQ_P": fq_p, "FQ_R": fq_r, "FQ_F1": fq_f1,
            "FQ_FPR": (p_fq & ~g_fq).sum() / (~g_fq).sum() if (~g_fq).sum() else np.nan,
            "FQ_predicted": int(p_fq.sum()), "FQ_gold": int(g_fq.sum()), "n": len(gF)}


def l1_from_frame(d, pred=("pred_F", "pred_Q"), gold=("F", "Q")):
    return l1(d[gold[0]], d[gold[1]], d[pred[0]], d[pred[1]])


def fq_counts(d, pred=("pred_F", "pred_Q"), gold=("F", "Q")):
    """Per-item 0/1 indicators that the count-based metrics (F+Q P/R/F1, FPR, exact match) are sums of."""
    gF, gQ, pF, pQ = (d[c].values for c in (*gold, *pred))
    g, p = (gF == 1) & (gQ == 1), (pF == 1) & (pQ == 1)
    return pd.DataFrame({"tp": g & p, "pred": p, "gold": g, "fp": p & ~g, "neg": ~g,
                         "exact": (gF == pF) & (gQ == pQ), "n": 1}, index=d.index).astype(int)


def count_stats(c):
    """F+Q F1 and exact match from summed indicators; c has one row per resample (or a single row)."""
    prec = np.divide(c["tp"].astype(float), c["pred"], out=np.zeros(len(c["tp"]), float), where=c["pred"] > 0)
    rec = np.divide(c["tp"].astype(float), c["gold"], out=np.full(len(c["tp"]), np.nan), where=c["gold"] > 0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
    return {"FQ_F1": f1, "exact_match": c["exact"] / c["n"], "FQ_FPR": c["fp"] / c["neg"]}


def fast_cluster_bootstrap(d, cluster="project", B=10000, seed=42, alpha=0.05, **cols):
    """Project-clustered bootstrap for the count-based metrics, vectorised over resamples (same resampling
    scheme as cluster_bootstrap, seconds instead of minutes). Returns {metric: (estimate, low, high)}."""
    per = fq_counts(d, **cols).groupby(d[cluster].values).sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(B, len(per)))
    sums = {k: per[k].values[idx].sum(1) for k in per}
    boot, point = count_stats(sums), count_stats({k: np.array([per[k].sum()]) for k in per})
    return {k: (float(point[k][0]), *np.nanpercentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])) for k, v in boot.items()}


def cluster_bootstrap(d, stat, cluster="project", B=10000, seed=42, alpha=0.05):
    """Point estimate and percentile CI of stat(frame), resampling whole clusters.

    For a paired comparison, pass a stat that returns the difference between two prediction
    columns of the same frame, so both systems see the same resampled projects."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in d.groupby(cluster)}
    keys = np.array(list(groups))
    vals = []
    for _ in range(B):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys), replace=True)], ignore_index=True)
        v = stat(s)
        if np.isfinite(v): vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": stat(d), "ci_low": lo, "ci_high": hi, "B_valid": len(vals)}


# ---------------- RQ2: uncertainty and confidence (first-draft review, comment 3) ----------------
def vote_entropy(tuples, n, unverified=0):
    """Shannon entropy (natural log) of the answer distribution over n samples, normalised to 0..1 by log(n).
    Each unverified sample counts as its own outcome (no two failures agree), so failing verification raises
    uncertainty and an item whose samples all fail gets the maximum, 1. (Pooling failures into one outcome
    would make 'all failed' look certain.)"""
    counts = list(Counter(tuples).values()) + [1] * unverified
    p = np.array(counts, float) / n
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(n)) if n > 1 else 0.0


def ece(conf, correct, bins=None):
    """Expected calibration error. bins=None uses each distinct confidence value as a bin (vote shares take
    n + 1 values); an int gives that many equal-width bins on [0, 1] (continuous signals)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = (np.unique(conf, return_inverse=True)[1] if bins is None
           else np.minimum((conf * bins).astype(int), bins - 1))
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx)))


def aurc(conf, correct):
    """Area under the risk-coverage curve: accept items from most to least confident; risk = error rate among
    accepted items, averaged over coverage 1/N .. N/N. Ties in confidence are resolved in expectation over
    random order, so tied signals (vote shares) are not favoured or penalised by sort order."""
    conf, err = np.asarray(conf, float), 1 - np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable"); conf, err = conf[order], err[order]
    N, total, k0, e0 = len(conf), 0.0, 0, 0.0
    for v in np.unique(conf)[::-1]:
        g = conf == v; ng, eg = g.sum(), err[g].sum()
        ks = np.arange(1, ng + 1)
        total += ((e0 + ks * eg / ng) / (k0 + ks)).sum()
        k0, e0 = k0 + ng, e0 + eg
    return total / N


def risk_at_coverage(conf, correct, cov):
    """Error rate among the cov share of items with the highest confidence (ties resolved in expectation)."""
    conf, err = np.asarray(conf, float), 1 - np.asarray(correct, float)
    k = max(1, int(round(cov * len(conf)))); thr = np.sort(conf)[::-1][k - 1]
    above, at = conf > thr, conf == thr
    return float((err[above].sum() + (k - above.sum()) * err[at].mean()) / k)


def confidence_quality(conf, correct, bins=None):
    """All RQ2 measures for one confidence signal against one fixed set of predictions."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    auroc = roc_auc_score(correct, conf) if 0 < correct.sum() < len(correct) else np.nan
    return {"AUROC": auroc, "ECE": ece(conf, correct, bins), "Brier": float(((conf - correct) ** 2).mean()),
            "AURC": aurc(conf, correct), "risk@85": risk_at_coverage(conf, correct, 0.85),
            "agree_but_wrong": float((~correct[conf == 1]).mean()) if (conf == 1).any() else np.nan,
            "accuracy": float(correct.mean()), "n": len(conf)}
