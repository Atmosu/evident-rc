"""Contamination probe: guided instruction completion (Golchin & Surdeanu, ICLR 2024). Evaluation only.

The model gets the first half of each requirement (in words) and is asked to complete it twice:
  guided   names the source dataset and asks for the rest exactly as it appears there
  general  asks for a plausible completion of a software requirement, with no dataset named
Greedy decoding, no grammar constraint. A model that has memorised the dataset reproduces the true second half
more closely under the guided instruction, so the signal is the paired guided minus general difference in
ROUGE-L, plus the number of near-exact (ROUGE-L >= 0.75) and exact completions.

Sets: PROMISE (all 625) and Dalpiaz-4 (331; public since 2019, so not a clean control). The post-cutoff external
set is added once it is annotated. Requirements under 8 words are skipped.

Usage: python src/contamination.py --model <path> [--quant 4bit] [--limit N]
Writes results/contamination/<set>.csv and results/contamination/summary.json
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import data as D
import evident as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/contamination"
NAMES = {"PROMISE": "the PROMISE NFR dataset of software requirements (tera-PROMISE repository)",
         "Dalpiaz-4": "the requirements datasets published by Dalpiaz et al. (Dronology, Leeds, ReqView, WASP)"}
GUIDED = ("You are given the first part of a requirement from {name}. Complete the second part exactly as it "
          "appears in the dataset. Answer with the second part only.\nFirst part: {first}")
GENERAL = ("Complete the following software requirement with a plausible ending. Answer with the rest of the "
           "requirement only.\nFirst part: {first}")
BATCH, MIN_WORDS = 16, 8


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def rouge_l(a, b):
    """ROUGE-L F1 over lowercased word tokens (longest common subsequence)."""
    x, y = toks(a), toks(b)
    if not x or not y: return 0.0
    dp = [0] * (len(y) + 1)
    for xi in x:
        prev = 0
        for j, yj in enumerate(y, 1):
            cur = dp[j]; dp[j] = prev + 1 if xi == yj else max(dp[j], dp[j - 1]); prev = cur
    l = dp[-1]; p, r = l / len(x), l / len(y)
    return 0.0 if l == 0 else 2 * p * r / (p + r)


def clean(out, first):
    """Drop answer labels ("Second part:"), leading ellipses and quotes, and a repeated first part."""
    s = re.sub(r"^\s*(second part|answer|completion)\s*:\s*", "", out, flags=re.I).strip().strip('"').lstrip(".… ").strip()
    if toks(s)[: len(toks(first))] == toks(first) and toks(first):
        s = " ".join(s.split()[len(first.split()):])
    return s


@torch.no_grad()
def complete(llm, prompts, max_new=60):
    tok, m = llm.tok, llm.m
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(prompts), BATCH):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + BATCH]]
        b = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.dev)
        g = m.generate(**b, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        outs += [s.strip().split("\n")[0] for s in tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    llm = ev.HFBackend(a.model, 42, constrained=False, quant=a.quant)
    sets = {"PROMISE": D.promise(), "Dalpiaz-4": D.dalpiaz_public()}
    summary = {"model": a.model, "quant": a.quant, "decoding": "greedy", "guided": GUIDED, "general": GENERAL, "sets": {}}
    rng = np.random.default_rng(42)
    for name, df in sets.items():
        df = df[df.text.str.split().str.len() >= MIN_WORDS].reset_index(drop=True)
        if a.limit: df = df.head(a.limit)
        w = df.text.str.split()
        first = [" ".join(x[: len(x) // 2]) for x in w]; second = [" ".join(x[len(x) // 2:]) for x in w]
        t0 = time.time()
        g = complete(llm, [GUIDED.format(name=NAMES[name], first=f) for f in first])
        n = complete(llm, [GENERAL.format(first=f) for f in first])
        r = pd.DataFrame({"req_id": df.req_id, "project": df.project, "first": first, "true_second": second,
                          "guided": [clean(o, f) for o, f in zip(g, first)], "general": [clean(o, f) for o, f in zip(n, first)],
                          "guided_raw": g, "general_raw": n})
        cut = lambda out, ref: " ".join(out.split()[: len(ref.split()) + 5])  # same length budget for both prompts
        r["rouge_guided"] = [rouge_l(cut(o, s), s) for o, s in zip(r.guided, r.true_second)]
        r["rouge_general"] = [rouge_l(cut(o, s), s) for o, s in zip(r.general, r.true_second)]
        r["exact_guided"] = [toks(cut(o, s))[: len(toks(s))] == toks(s) for o, s in zip(r.guided, r.true_second)]
        r["exact_general"] = [toks(cut(o, s))[: len(toks(s))] == toks(s) for o, s in zip(r.general, r.true_second)]
        r.to_csv(OUT / f"{name}.csv", index=False)
        d = (r.rouge_guided - r.rouge_general).values
        boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        summary["sets"][name] = {"items": len(r), "rougeL_guided": round(float(r.rouge_guided.mean()), 3),
                                 "rougeL_general": round(float(r.rouge_general.mean()), 3),
                                 "diff": round(float(d.mean()), 3),
                                 "diff_ci": [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
                                 "near_exact_guided": int((r.rouge_guided >= 0.75).sum()),
                                 "near_exact_general": int((r.rouge_general >= 0.75).sum()),
                                 "exact_guided": int(r.exact_guided.sum()), "exact_general": int(r.exact_general.sum()),
                                 "secs": round(time.time() - t0, 1)}
        print(name, json.dumps(summary["sets"][name]), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
