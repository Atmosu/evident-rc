"""Near-duplicate check across PROMISE projects (first-draft review, comment 5).

LOPO keeps projects apart, but the same or almost the same requirement can still appear in two projects and
leak its label into training. For every pair of requirements from different projects this script computes
  meaning similarity: cosine of MiniLM embeddings (the same encoder used for retrieval)
  wording similarity: character-level ratio of the normalised texts (difflib)
and flags a pair when either is high. Writes results/near_duplicates.csv with both labels of every pair.
"""
import sys
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data as D
from pilot import embed

OUT = Path(__file__).resolve().parents[1] / "results"
COS_MIN, TEXT_MIN = 0.90, 0.85   # flag thresholds, chosen before looking at the pairs


def main():
    P = D.promise(); E = embed(P.text.tolist())
    S = E @ E.T; proj = P.project.values; norm = P.text.str.lower().str.split().str.join(" ").values
    i, j = np.triu_indices(len(P), 1)
    cross = proj[i] != proj[j]; i, j = i[cross], j[cross]
    cand = S[i, j] >= 0.80                       # only score wording for pairs that are at least similar in meaning
    rows = []
    for a, b in zip(i[cand], j[cand]):
        txt = SequenceMatcher(None, norm[a], norm[b]).ratio()
        if S[a, b] >= COS_MIN or txt >= TEXT_MIN:
            rows.append({"req_a": P.req_id[a], "req_b": P.req_id[b], "cosine": round(float(S[a, b]), 3),
                         "text_ratio": round(txt, 3), "labels_a": f"F={P.F[a]},Q={P.Q[a]}", "labels_b": f"F={P.F[b]},Q={P.Q[b]}",
                         "same_labels": bool(P.F[a] == P.F[b] and P.Q[a] == P.Q[b]),
                         "involves_FQ": bool((P.F[a] & P.Q[a]) or (P.F[b] & P.Q[b])),
                         "text_a": P.text[a], "text_b": P.text[b]})
    d = pd.DataFrame(rows).sort_values("cosine", ascending=False); d.to_csv(OUT / "near_duplicates.csv", index=False)
    items = set(d.req_a) | set(d.req_b) if len(d) else set()
    print(f"cross-project pairs scored: {cross.sum()}; flagged pairs: {len(d)}; items involved: {len(items)}")
    if len(d):
        print(f"exact-text duplicates (ratio 1.0): {(d.text_ratio == 1).sum()}; flagged pairs with different labels: "
              f"{(~d.same_labels).sum()}; pairs involving an F+Q item: {d.involves_FQ.sum()}")
        print(d[["req_a", "req_b", "cosine", "text_ratio", "same_labels", "text_a", "text_b"]].head(12).to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    main()
