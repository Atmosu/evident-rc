"""Faithfulness tests for the quoted evidence (first-draft review, comment 7). Evaluation only.

For every item of a finished EVIDENT-RC run whose prediction sets a label to 1 with a verified quote, the same
model is asked again, with the same frozen prompt and the same retrieved examples, about altered requirements:

  span_mask    the quoted words replaced by "[...]"                 (comprehensiveness: the label should drop)
  random_mask  as many words elsewhere replaced by "[...]"          (control: the label should mostly stay)
  sufficiency  only the quoted words given as the requirement       (sufficiency: the label should stay)
  full_mask    the whole requirement replaced by "[...]"            (control: base rate of the label, once per item)

The label for each altered input is the majority over n samples of that label's value among schema-valid samples
(no quote check, since quotes cannot match a masked text). Evidence counts as faithful when the span_mask drop
rate is clearly above the random_mask drop rate.

Usage: python src/faithfulness.py <run_dir> --model <path> [--quant 4bit] [--n 5] [--limit 0]
Writes <run_dir>/faithfulness/<fold>.jsonl and <run_dir>/faithfulness/summary.json
"""
import argparse, json, random, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import data as D
import evident as ev

MASK = "[...]"


def locate(text, span):
    """(start, end) character offsets of the span in the text, ignoring case and runs of whitespace."""
    pat = r"\s+".join(re.escape(w) for w in span.split())
    m = re.search(pat, text, re.I)
    return (m.start(), m.end()) if m else None


def token_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def random_window(text, loc, rng):
    """A window of as many whitespace tokens as the span, not overlapping it when possible."""
    toks = token_spans(text); n_span = len(text[loc[0]:loc[1]].split())
    starts = [i for i in range(len(toks) - n_span + 1)]
    free = [i for i in starts if toks[i + n_span - 1][1] <= loc[0] or toks[i][0] >= loc[1]]
    pool = free or [i for i in starts if (toks[i][0], toks[i + n_span - 1][1]) != loc] or starts
    i = rng.choice(pool)
    return toks[i][0], toks[i + n_span - 1][1]


def masked(text, a, b):
    return " ".join((text[:a] + MASK + text[b:]).split())


def label_share(raws, lab):
    """Share of schema-valid samples that set lab to 1, and the majority value (ties count as 1)."""
    vals = [o[lab] for o in map(ev.parse, raws) if ev.schema_ok(o)]
    if not vals: return None, None
    share = sum(vals) / len(vals)
    return round(share, 3), int(share >= 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--model", required=True)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run_dir); meta = json.loads((run / "run_meta.json").read_text())
    assert meta.get("variant", "evident") == "evident", "faithfulness is tested on the EVIDENT-RC method run"
    system = ev.PROMPTS[meta["prompt"]]
    assert ev.prompt_hash(system) == meta["prompt_hash"], "prompt changed since the run"
    out = run / "faithfulness"; out.mkdir(exist_ok=True)
    pool = D.promise().set_index("req_id")
    llm = ev.HFBackend(a.model, meta["seed"], constrained=meta["decoding"] == "constrained", quant=a.quant)
    rng = random.Random(meta["seed"])
    summary = {"model": a.model, "run": run.name, "n": a.n, "temperature": meta["temperature"], "folds": {}}
    rows_all = []
    for f in sorted(run.glob("*.jsonl")):
        recs = [json.loads(l) for l in open(f)]
        recs = [r for r in recs if r["F"] is not None][: a.limit or None]
        jobs, rows = [], []  # one job = (row index, condition, label, messages)
        for r in recs:
            ex = pool.loc[r["retrieved"]].reset_index()
            row = {"req_id": r["req_id"], "project": r["project"], "pred": {"F": r["F"]["label"], "Q": r["Q"]["label"]},
                   "gold": r["gold"], "tests": {}}
            jobs.append((len(rows), "full_mask", None, ev.build_messages(MASK, ex, system)))
            for lab in ("F", "Q"):
                sp = r[lab]["span"]
                if r[lab]["label"] != 1 or not sp: continue
                loc = locate(r["text"], sp)
                if loc is None: continue
                ra, rb = random_window(r["text"], loc, rng)
                row["tests"][lab] = {"span": sp, "random_words": r["text"][ra:rb]}
                for cond, txt in (("span_mask", masked(r["text"], *loc)), ("random_mask", masked(r["text"], ra, rb)),
                                  ("sufficiency", r["text"][loc[0]:loc[1]])):
                    jobs.append((len(rows), cond, lab, ev.build_messages(txt, ex, system)))
            rows.append(row)
        t0 = time.time()
        raws = llm.generate([j[3] for j in jobs], a.n, meta["temperature"], meta["max_tokens"])
        for (i, cond, lab, _), raw in zip(jobs, raws):
            if cond == "full_mask":
                rows[i]["full_mask"] = {l: dict(zip(("share", "label"), label_share(raw, l))) for l in ("F", "Q")}
            else:
                share, maj = label_share(raw, lab)
                rows[i]["tests"][lab][cond] = {"share": share, "label": maj}
        with open(out / f.name, "w") as fh:
            for row in rows: fh.write(json.dumps(row) + "\n")
        rows_all += rows
        summary["folds"][f.stem] = {"items": len(rows), "queries": len(jobs), "gen_s": round(time.time() - t0, 1)}
        print(f.stem, json.dumps(summary["folds"][f.stem]), flush=True)
    summary["overall"] = summarise(rows_all)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["overall"], indent=1))


def summarise(rows):
    """Per label: how often the label drops (span_mask, random_mask) or stays (sufficiency), plus the full-mask
    base rate. Undecidable cases (no schema-valid sample) are counted separately."""
    res = {}
    for lab in ("F", "Q"):
        t = [r["tests"][lab] for r in rows if lab in r["tests"]]
        d = {"tested": len(t)}
        for cond, good in (("span_mask", 0), ("random_mask", 0), ("sufficiency", 1)):
            vals = [x[cond]["label"] for x in t if x.get(cond, {}).get("label") is not None]
            key = "kept_rate" if cond == "sufficiency" else "drop_rate"
            d[cond] = {key: round(float(np.mean([v == good for v in vals])), 3) if vals else None,
                       "n": len(vals), "undecided": len(t) - len(vals)}
        fm = [r["full_mask"][lab]["label"] for r in rows if r.get("full_mask", {}).get(lab, {}).get("label") is not None]
        d["full_mask_label_rate"] = round(float(np.mean(fm)), 3) if fm else None
        res[lab] = d
    return res


if __name__ == "__main__":
    main()
