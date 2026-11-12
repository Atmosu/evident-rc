"""Load and harmonise the verified datasets into one schema.

Schema (one row per requirement):
  req_id, source, project, text, F (0/1), Q (0/1), attr (str or None), orig_class
"""
import pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data"

# Level-2 quality-attribute grouping (ISO/IEC 25010-inspired merge of PROMISE NFR classes)
ATTR_MAP = {"SE": "Security", "US": "Usability", "LF": "Usability",
            "PE": "Performance", "SC": "Performance", "A": "Reliability",
            "FT": "Reliability", "MN": "Maintainability", "PO": "Maintainability",
            "O": "Operational", "L": "Legal"}


def clean(t):
    return " ".join(str(t).strip().strip("'\"").split())


def promise():
    r = pd.read_csv(D / "NoRBERT/Code/Task4_relabeled_Promise_NFR_dataset/promise-reclass.csv", sep=";")
    df = pd.DataFrame({
        "req_id": [f"P{p:02d}-{i:03d}" for i, p in enumerate(r.ProjectID)],
        "source": "PROMISE", "project": "P" + r.ProjectID.astype(str).str.zfill(2),
        "text": r.RequirementText.map(clean), "F": r.IsFunctional, "Q": r.IsQuality,
        "orig_class": r.Class})
    # attribute only defined when the requirement is Q and the original class is an NFR class
    df["attr"] = [ATTR_MAP.get(c) if q == 1 else None for c, q in zip(df.orig_class, df.Q)]
    return df


def dalpiaz_public():
    base = D / "dalpiaz/Manually tagged datasets"
    out = []
    for name in ["dronology", "leeds", "reqview", "wasp"]:
        d = pd.read_csv(base / f"{name}.csv")
        out.append(pd.DataFrame({
            "req_id": [f"{name[:3].upper()}-{i:03d}" for i in range(len(d))],
            "source": name, "project": name, "text": d.RequirementText.map(clean),
            "F": d.IsFunctional.astype(int), "Q": d.IsQuality.astype(int),
            "orig_class": None, "attr": None}))
    return pd.concat(out, ignore_index=True)


def itrust():
    c = pd.read_csv(D / "ftlr/datasets/classification_dataset.csv")
    c = c[c.ProjectID == "iTrust"].reset_index(drop=True)
    return pd.DataFrame({
        "req_id": [f"IT-{f.replace('.txt','')}-{i:03d}" for i, f in enumerate(c.file)],
        "source": "iTrust", "project": "iTrust", "text": c.text.map(clean),
        "F": c.F.astype(int), "Q": c.Q.astype(int), "orig_class": None, "attr": None,
        "uc_file": c.file})


if __name__ == "__main__":
    for name, f in [("PROMISE", promise), ("Dalpiaz public", dalpiaz_public), ("iTrust (FTLR)", itrust)]:
        d = f()
        fq = ((d.F == 1) & (d.Q == 1)).sum()
        print(f"{name}: n={len(d)} projects={d.project.nunique()} F={d.F.sum()} Q={d.Q.sum()} F+Q={fq} "
              f"neither={((d.F==0)&(d.Q==0)).sum()} mean_words={d.text.str.split().str.len().mean():.1f}")
    p = promise()
    print("attr counts:", p.attr.value_counts(dropna=False).to_dict())
    print(dalpiaz_public().groupby("source")[["F", "Q"]].agg(["sum", "count"]))


def post_cutoff(path=None):
    """Post-cutoff external requirements (public tenders, Oct 2024 to Dec 2025) with human labels.
    Two annotators on 13 items (adjudicated by the guideline), one on the rest; see annotation/owner/adjudication_log.csv."""
    p = Path(path) if path else D / "external/post_cutoff_labels.csv"
    d = pd.read_csv(p)
    d["attr"] = d.attr.where(d.attr.notna(), None)
    return d[["req_id", "source", "project", "text", "F", "Q", "orig_class", "attr"]].reset_index(drop=True)
