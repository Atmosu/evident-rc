"""EVIDENT-RC: evidence-verified, confidence-calibrated requirements classification.

Pipeline (one LOPO fold)
  retrieval  top-k MiniLM neighbours from the fold's training projects only, >= 1 F+Q example guaranteed
  labelling  LLM returns schema-checked JSON (F, Q, attribute, one verbatim span per positive label, explanation)
             n samples at temperature T
  V1         a sample is verified if JSON + schema are valid and every span is a strict substring of the
             requirement after identical whitespace/case normalisation
  V2         prediction = most frequent (F, Q, attribute) tuple among verified samples;
             c = verified votes for that tuple / n; needs_review = c < tau

Backends
  hf    Hugging Face transformers on cuda / mps / cpu (Mac development, GPU fallback)
  vllm  vLLM batch inference (GPU runs)

Decoding
  constrained  (default) xgrammar restricts tokens to DECODE_SCHEMA on both backends, so V1 rejects only bad
               evidence (span not a substring, full-sentence span) and attribute-missing tuples
  free         unconstrained text; parse + schema failures also make a sample unverified (descriptive check)

Usage
  python src/evident.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --fold P01 --limit 20
  python src/evident.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --fold all
"""
import argparse, hashlib, json, platform, random, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
import jsonschema

sys.path.insert(0, str(Path(__file__).parent))
import data as D

ROOT = Path(__file__).resolve().parents[1]
# Development-only projects (decided 2026-11-04): pilot and method decisions use these; they are never LLM test
# folds, so reported LOPO results cover the other 13 projects. They stay in training/retrieval pools.
DEV_PROJECTS = ["P02", "P04"]
ATTRS = sorted(set(D.ATTR_MAP.values()))

# ---------------- prompt + schema ----------------
SYSTEM = """You classify software requirements.
Labels:
- F (functional): the requirement states a function or behaviour the system performs (what it does).
- Q (quality): the requirement states a quality the system must have (how well it does something, or a constraint on it).
A requirement can be F only, Q only, or both F and Q.
If Q = 1, choose exactly one attribute:
- Usability: ease of use, learnability, look and feel, user interface appearance.
- Performance: speed, response time, throughput, capacity, scalability.
- Security: access control, authentication, privacy, protection of data.
- Operational: operating environment, platforms, deployment, interfaces with other systems.
- Reliability: availability, fault tolerance, recovery, uptime.
- Maintainability: ease of change, modularity, portability to other platforms.
- Legal: laws, regulations, standards, licences, compliance.
If Q = 0, attribute is null.
For every label set to 1, quote ONE short span copied word for word from the requirement that shows that label.
The span must be shorter than the whole requirement. Use null for the span of a label set to 0.
Answer with one JSON object only, with exactly these keys:
{"F": 0 or 1, "F_span": "..." or null, "Q": 0 or 1, "Q_span": "..." or null, "attribute": one attribute name or null, "explanation": "at most 30 words"}"""

SCHEMA = {
    "type": "object", "required": ["F", "F_span", "Q", "Q_span", "attribute", "explanation"],
    "properties": {
        "F": {"enum": [0, 1]}, "F_span": {"type": ["string", "null"]},
        "Q": {"enum": [0, 1]}, "Q_span": {"type": ["string", "null"]},
        "attribute": {"enum": ATTRS + [None]}, "explanation": {"type": "string"}}}
MAX_EXPL_WORDS = 30
# Grammar for constrained decoding: same keys in prompt order, no extra keys, explanation capped near 30 words.
DECODE_SCHEMA = {**SCHEMA, "additionalProperties": False,
                 "properties": {**SCHEMA["properties"], "explanation": {"type": "string", "maxLength": 240}}}


def label_json(r):
    """Gold labels of a retrieved example, same keys as the answer format (no gold spans exist for training items)."""
    attr = r.attr if r.Q == 1 and isinstance(r.attr, str) else None
    return json.dumps({"F": int(r.F), "Q": int(r.Q), "attribute": attr})


# Prompt v2 (tested on development projects P02/P04, 2026-11-04): dev pilot 1 showed the model quoting the whole
# requirement as F evidence in 23% of samples, which fails V1. v2 only changes how spans are described.
SYSTEM_V2 = SYSTEM.replace(
    """For every label set to 1, quote ONE short span copied word for word from the requirement that shows that label.
The span must be shorter than the whole requirement. Use null for the span of a label set to 0.""",
    """For every label set to 1, quote ONE short span copied word for word from the requirement that shows that label.
Quote only the key words, never the whole requirement:
- for F, the action the system performs, usually a verb and its object (for example "refresh the display");
- for Q, the words that state the quality (for example "every 60 seconds").
Use null for the span of a label set to 0.""")
assert SYSTEM_V2 != SYSTEM

# Prompt v3 (dev, 2026-11-04): v2 raised F+Q predictions from 19 to 29 (gold 17). Both v2 span examples came from one
# F+Q requirement, which may have suggested that requirements usually carry both labels. v3 keeps the key-words rule,
# takes the two examples from two different single-label requirements, and says plainly that most are one label only.
SYSTEM_V3 = SYSTEM.replace(
    "A requirement can be F only, Q only, or both F and Q.",
    """A requirement can be F only, Q only, or both F and Q. Most requirements have only one of the two labels.
Set both to 1 only when the requirement states a function AND, in separate words, a quality of it.""").replace(
    """For every label set to 1, quote ONE short span copied word for word from the requirement that shows that label.
The span must be shorter than the whole requirement. Use null for the span of a label set to 0.""",
    """For every label set to 1, quote ONE short span copied word for word from the requirement that shows that label.
Quote only the key words, never the whole requirement. For example:
- "The system shall send a receipt to the customer after payment." is F only; F_span "send a receipt".
- "The product shall be available 99% of the time." is Q only; Q_span "available 99% of the time".
Use null for the span of a label set to 0.""")
assert SYSTEM_V3.count("For example") == 1 and "Most requirements" in SYSTEM_V3

# Prompt v4 (dev, 2026-11-06): with the main model (Llama-3.1-8B, 4-bit) v3 labelled 67 of 95 development items F+Q
# (gold 17): any verb phrase ("have a professional appearance", "run on Windows CE") was taken as a function. v4 adds
# the hard-case rules of the annotation guideline (the same PROMISE conventions the human annotators follow). Its
# examples are invented, not taken from any dataset.
_V4_RULES = """A requirement can be F only, Q only, or both F and Q. Most requirements have only one of the two labels.
Set both to 1 only when the requirement states a function AND, in separate words, a quality of it.
F needs an action the system performs for its users or on its data (for example send, display, record, calculate,
or let a user do something). Words such as "shall be", "shall have", "shall run on", "shall use" or "shall operate"
do not make a requirement F by themselves. These are Q only (F = 0):
- appearance, look and feel, ease of use or learning, and targets for the share of users who succeed at a task (Usability);
- general speed, capacity or number of users, with no named action (Performance);
- who may use the system in general (Security);
- platforms, operating environment, offline use, interfaces with other systems (Operational);
- availability, uptime, backup and recovery (Reliability).
These are F and Q:
- a named action with a limit on it, such as "The system shall display the order history within 3 seconds." (Performance);
- a named action allowed only for some users, such as "Only a manager shall be able to cancel an invoice." (Security)."""
SYSTEM_V4 = SYSTEM_V3.replace(
    """A requirement can be F only, Q only, or both F and Q. Most requirements have only one of the two labels.
Set both to 1 only when the requirement states a function AND, in separate words, a quality of it.""", _V4_RULES)
assert SYSTEM_V4 != SYSTEM_V3 and "These are Q only" in SYSTEM_V4
PROMPTS = {"v1": SYSTEM, "v2": SYSTEM_V2, "v3": SYSTEM_V3, "v4": SYSTEM_V4}

# ---------------- comparison variants of the frozen v3 prompt (2026-11-06) ----------------
# L0 "noevidence": same labels and definitions, no quotes asked for and none checked (RQ1 ablation).
# S2 "verbal": the v3 task plus a stated probability that the labels are correct (RQ2 comparison, one sample).
_V3_QUOTES = """For every label set to 1, quote ONE short span copied word for word from the requirement that shows that label.
Quote only the key words, never the whole requirement. For example:
- "The system shall send a receipt to the customer after payment." is F only; F_span "send a receipt".
- "The product shall be available 99% of the time." is Q only; Q_span "available 99% of the time".
Use null for the span of a label set to 0.
"""
_V3_KEYS = """{"F": 0 or 1, "F_span": "..." or null, "Q": 0 or 1, "Q_span": "..." or null, "attribute": one attribute name or null, "explanation": "at most 30 words"}"""
assert _V3_QUOTES in SYSTEM_V3 and _V3_KEYS in SYSTEM_V3
def l0_system(base):
    assert _V3_QUOTES in base and _V3_KEYS in base
    return base.replace(_V3_QUOTES, "").replace(
        _V3_KEYS, """{"F": 0 or 1, "Q": 0 or 1, "attribute": one attribute name or null, "explanation": "at most 30 words"}""")


def verbal_system(base):
    assert _V3_KEYS in base
    return base.replace(
        _V3_KEYS, _V3_KEYS[:-1] + """, "confidence": a number from 0 to 1, your probability that your F, Q and attribute are all correct}""")


SYSTEM_L0, SYSTEM_VERBAL = l0_system(SYSTEM_V3), verbal_system(SYSTEM_V3)  # v3-based (record)
SCHEMA_L0 = {"type": "object", "required": ["F", "Q", "attribute", "explanation"],
             "properties": {k: SCHEMA["properties"][k] for k in ("F", "Q", "attribute", "explanation")}}
DECODE_L0 = {**SCHEMA_L0, "additionalProperties": False,
             "properties": {**SCHEMA_L0["properties"], "explanation": {"type": "string", "maxLength": 240}}}
DECODE_VERBAL = {**DECODE_SCHEMA, "required": SCHEMA["required"] + ["confidence"],
                 "properties": {**DECODE_SCHEMA["properties"], "confidence": {"type": "number"}}}
# Variants are built from whichever base prompt --prompt selects, so they always match the method run.
VARIANTS = {"evident": lambda b: (b, DECODE_SCHEMA), "noevidence": lambda b: (l0_system(b), DECODE_L0),
            "verbal": lambda b: (verbal_system(b), DECODE_VERBAL)}


def prompt_hash(system):
    return hashlib.sha256((system + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]


def build_messages(text, examples, system=SYSTEM):
    shots = "\n\n".join(f"Requirement: {r.text}\nLabels: {label_json(r)}" for r in examples.itertuples())
    user = (f"Labelled examples from other projects:\n\n{shots}\n\nNow classify this requirement.\n" if len(examples)
            else "Classify this requirement.\n") + f"Requirement: {text}\nJSON:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


PROMPT_HASH = prompt_hash(SYSTEM)  # v1

# ---------------- per-aspect probe (ablation; first-draft review, comment 2) ----------------
# One separate question per aspect: "is it present? quote it or answer none". It targets omissions, which span
# verification of the joint answer cannot catch. A "none" answer has no span, so absence is never verified by text.
PROBE_ASK = {"F": "Does this requirement state a function or behaviour the system performs (what it does)?",
             "Q": "Does this requirement state a quality the system must have (how well it does something, or a constraint on it)?"}
PROBE_SYSTEM = """You check one aspect of a software requirement.
If the aspect is present, answer {"present": 1, "span": "..."} and quote ONE short span copied word for word from the
requirement that shows it; the span must be shorter than the whole requirement.
If it is not present, answer {"present": 0, "span": null}.
Answer with one JSON object only."""
PROBE_SCHEMA = {"type": "object", "required": ["present", "span"], "additionalProperties": False,
                "properties": {"present": {"enum": [0, 1]}, "span": {"type": ["string", "null"]}}}
PROBE_HASH = hashlib.sha256((PROBE_SYSTEM + json.dumps(PROBE_ASK, sort_keys=True)).encode()).hexdigest()[:12]


def build_probe(text, lab):
    return [{"role": "system", "content": PROBE_SYSTEM},
            {"role": "user", "content": f"{PROBE_ASK[lab]}\nRequirement: {text}\nJSON:"}]


# ---------------- folds + retrieval ----------------
def lopo_split(P, test_proj):
    """Test = one project; validation = next project in sorted order; train = remaining 13."""
    projs = sorted(P.project.unique())
    val_proj = projs[(projs.index(test_proj) + 1) % len(projs)]
    te, va = P[P.project == test_proj], P[P.project == val_proj]
    tr = P[~P.project.isin([test_proj, val_proj])]
    assert not set(tr.project) & set(te.project) and not set(tr.project) & set(va.project) \
        and not set(va.project) & set(te.project), "project leakage"
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def retrieve(Eq, Epool, pool, k, fq_guarantee=True):
    """Top-k cosine neighbours; if none is F+Q (and the guarantee is on), the k-th slot is replaced by the
    most similar F+Q item. k = 0 gives no examples (no-retrieval ablation)."""
    if k == 0: return [[] for _ in range(len(Eq))]
    fq = ((pool.F == 1) & (pool.Q == 1)).values
    assert fq.any(), "retrieval pool has no F+Q example"
    sims = Eq @ Epool.T; out = []
    for s in sims:
        order = np.argsort(-s); top = list(order[:k])
        if fq_guarantee and not fq[top].any():
            top[-1] = next(i for i in order if fq[i])
        out.append(top)
    return out


# ---------------- V1 / V2 ----------------
def norm(s):
    return " ".join(str(s).lower().split())


def core(s):
    return norm(s).strip(" .,;:!?'\"")


def parse(raw):
    """First top-level {...} object in the model output, or None."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except json.JSONDecodeError: return None


# Span rules added after the first-draft review: a span needs at least one content word, and may cover at most
# MAX_SPAN_FRAC of the requirement's words (minimum MIN_SPAN_CAP), so "shall" or near-whole-sentence quotes fail.
STOP = set("""a an the and or of to in on for by with at from as is are be been being it its this that these those
which who whom whose when where while if then than so not no nor all any each every some such only also into onto
upon within without shall must will should would may might can could has have had do does did system product
application software""".split())
MAX_SPAN_FRAC, MIN_SPAN_CAP = 0.6, 4


def words(s):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", norm(s))


def content_words(s):
    return [w for w in words(s) if len(w) >= 3 and w not in STOP and not w.isdigit()]


def schema_ok(obj):
    if obj is None: return False
    try: jsonschema.validate(obj, SCHEMA); return True
    except jsonschema.ValidationError: return False


def span_problem(sp, text, strict=True):
    """Reason a quoted span fails V1, or None if it passes."""
    t = norm(text)
    if not sp or not norm(sp): return "span_missing"
    if norm(sp) not in t: return "span_not_substring"
    if len(core(sp)) >= len(core(t)): return "span_full"  # whole requirement minus punctuation
    if strict and not content_words(sp): return "span_no_content"
    if strict and len(words(sp)) > max(MIN_SPAN_CAP, int(np.ceil(MAX_SPAN_FRAC * len(words(t))))): return "span_too_long"
    return None


def verify_probe(obj, text):
    """Probe V1: 'present' needs a span that passes the same checks; 'absent' must not quote anything."""
    try: jsonschema.validate(obj, PROBE_SCHEMA)
    except (jsonschema.ValidationError, TypeError): return False, "schema"
    if obj["present"] == 0: return (True, "ok") if not obj["span"] else (False, "absent_with_span")
    why = span_problem(obj["span"], text)
    return (False, why) if why else (True, "ok")


def verify(obj, text, strict=True):
    """V1: returns (verified, reason). Reason names the first failed check. strict=False skips the span rules
    added after the first-draft review (used only to measure their effect)."""
    if obj is None: return False, "json"
    if not schema_ok(obj): return False, "schema"
    if len(obj["explanation"].split()) > MAX_EXPL_WORDS: return False, "explanation_len"
    if obj["Q"] == 1 and obj["attribute"] is None: return False, "attribute_missing"
    for lab in ("F", "Q"):
        if obj[lab] == 1:
            why = span_problem(obj[f"{lab}_span"], text, strict)
            if why: return False, f"{lab}_{why}"
    return True, "ok"


def verify_l0(obj, text):
    """V1 for the no-evidence variant: the same format checks, no quotes to check."""
    if obj is None: return False, "json"
    try: jsonschema.validate(obj, SCHEMA_L0)
    except jsonschema.ValidationError: return False, "schema"
    if len(obj["explanation"].split()) > MAX_EXPL_WORDS: return False, "explanation_len"
    if obj["Q"] == 1 and obj["attribute"] is None: return False, "attribute_missing"
    return True, "ok"


def stated_conf(obj):
    """S2: the model's stated probability, clipped to [0, 1]; None if missing or not a number."""
    v = obj.get("confidence") if isinstance(obj, dict) else None
    return min(1.0, max(0.0, float(v))) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def tuple_of(obj):
    """An attribute given with Q = 0 is dropped (same rule for every model; logged as attr_dropped)."""
    return (obj["F"], obj["Q"], obj["attribute"] if obj["Q"] == 1 else None)


def majority(tuples, n):
    """Most frequent tuple (ties -> first seen) and its share of all n samples."""
    if not tuples: return None, 0.0
    cnt = Counter(tuples); best = max(cnt, key=lambda t: (cnt[t], -tuples.index(t)))
    return best, cnt[best] / n


def vote(samples, n):
    """V2: most frequent verified tuple; c = its verified votes / n (unverified samples lower c)."""
    return majority([tuple_of(s["obj"]) for s in samples if s["verified"]], n)


def plain_vote(samples, n):
    """Plain self-consistency (S1): the same vote over every schema-valid sample, verified or not."""
    return majority([tuple_of(s["obj"]) for s in samples if schema_ok(s["obj"])], n)


def span_agreement(samples, lab):
    """Mean pairwise word-set Jaccard of one label's spans across verified samples that set that label to 1."""
    sets = [set(words(s["obj"][f"{lab}_span"])) for s in samples if s["verified"] and s["obj"][lab] == 1]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return round(float(np.mean([len(a & b) / len(a | b) for a, b in pairs])), 3) if pairs else None


# ---------------- backends ----------------
class HFBackend:
    def __init__(self, model, seed, constrained=True, quant="none"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dt = torch.float32 if self.dev == "cpu" else torch.float16
        self.tok = AutoTokenizer.from_pretrained(model)
        if quant == "4bit":
            # GPU used: an A100 slice with 10 GB, so the 7-8B model runs in 4-bit NF4 (bitsandbytes),
            # fp16 compute. A checkpoint saved already in 4-bit carries its own quantization_config.
            assert self.dev == "cuda", "4-bit loading needs a CUDA GPU"
            kw = {}
            if getattr(AutoConfig.from_pretrained(model), "quantization_config", None) is None:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
            self.m = AutoModelForCausalLM.from_pretrained(model, dtype=dt, device_map={"": 0}, **kw).eval()
        else:
            self.m = AutoModelForCausalLM.from_pretrained(model, dtype=dt).to(self.dev).eval()
        self.quant = quant
        self.seed, self.constrained, self.grammars = seed, constrained, {}
        if constrained:
            import xgrammar as xgr
            info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.m.config.vocab_size)
            self.compiler = xgr.GrammarCompiler(info)

    def grammar(self, schema):
        key = json.dumps(schema, sort_keys=True)
        if key not in self.grammars: self.grammars[key] = self.compiler.compile_json_schema(schema, any_whitespace=False)
        return self.grammars[key]

    # Samples per generate() call. On the 10 GB GPU slice, 20 sequences at once ran out of memory, so larger n is
    # drawn in chunks of 5. The first chunk uses the same seed as before, so n = 5 runs are unchanged.
    CHUNK = 5

    @torch.no_grad()
    def generate(self, convs, n, temperature, max_tokens, schema=DECODE_SCHEMA):
        outs = []
        for i, msgs in enumerate(convs):
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True).to(self.dev)
            texts = []
            for c, start in enumerate(range(0, n, self.CHUNK)):
                torch.manual_seed(self.seed + i + 1_000_003 * c)
                lp = None
                if self.constrained:
                    import xgrammar as xgr
                    lp = [xgr.contrib.hf.LogitsProcessor(self.grammar(schema))]  # single-use: one per generate() call
                g = self.m.generate(**ids, do_sample=True, temperature=temperature, top_p=1.0,
                                    num_return_sequences=min(self.CHUNK, n - start), max_new_tokens=max_tokens,
                                    pad_token_id=self.tok.eos_token_id, logits_processor=lp)
                texts += self.tok.batch_decode(g[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(texts)
        return outs


class VLLMBackend:
    def __init__(self, model, seed, dtype="auto", max_model_len=4096, constrained=True):
        from vllm import LLM
        from vllm.config import StructuredOutputsConfig
        self.llm = LLM(model=model, seed=seed, dtype=dtype, max_model_len=max_model_len,
                       structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
        self.seed, self.constrained = seed, constrained

    def generate(self, convs, n, temperature, max_tokens, schema=DECODE_SCHEMA):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        so = StructuredOutputsParams(json=schema, disable_any_whitespace=True) if self.constrained else None
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=self.seed,
                            structured_outputs=so)
        return [[o.text for o in r.outputs] for r in self.llm.chat(convs, sp, use_tqdm=True)]


# ---------------- run ----------------
def system_and_schema(a):
    """System prompt and decoding grammar for the chosen variant (evident = the frozen prompt version)."""
    return VARIANTS[a.variant](PROMPTS[a.prompt])


def run_fold(P, E, test_proj, llm, a, out_dir):
    tr, va, te = lopo_split(P, test_proj)
    idx = {r: i for i, r in enumerate(P.req_id)}
    Etr, Ete = E[[idx[r] for r in tr.req_id]], E[[idx[r] for r in te.req_id]]
    return run_items(tr, Etr, te, Ete, test_proj, llm, a, out_dir,
                     {"val_project": va.project.iloc[0], "n_train": len(tr), "n_test": len(te)})


def run_items(tr, Etr, te, Ete, name, llm, a, out_dir, info):
    """Label every item of te with examples retrieved from tr; writes <name>.jsonl. Used for LOPO folds and for
    external sets (tr = all of PROMISE)."""
    system, decode = system_and_schema(a)
    check = verify_l0 if a.variant == "noevidence" else verify
    if a.limit:  # stratified sample: all F+Q items first, then the rest, up to limit
        fq = (te.F == 1) & (te.Q == 1)
        keep = pd.concat([te[fq], te[~fq].sample(frac=1, random_state=a.seed)]).index[:a.limit]
        te, Ete = te.loc[sorted(keep)].reset_index(drop=True), Ete[sorted(keep)]
    nb = retrieve(Ete, Etr, tr, a.k, fq_guarantee=not a.no_fq_guarantee)
    convs = [build_messages(t, tr.iloc[nb[i]], system) for i, t in enumerate(te.text)]
    t0 = time.time(); raw = llm.generate(convs, a.n, a.temperature, a.max_tokens, schema=decode)
    if a.probe:  # two extra questions per item, same n and temperature
        raw_probe = llm.generate([build_probe(t, lab) for t in te.text for lab in ("F", "Q")], a.n, a.temperature, 80,
                                 schema=PROBE_SCHEMA)
    gen_s = time.time() - t0
    recs = []
    for i, r in enumerate(te.itertuples()):
        samples = []
        for txt in raw[i]:
            obj = parse(txt); ok, why = check(obj, r.text)
            ok_basic = ok if a.variant == "noevidence" else verify(obj, r.text, strict=False)[0]
            samples.append({"raw": txt, "obj": obj, "verified": ok, "verified_basic": ok_basic, "reason": why,
                            "attr_dropped": ok and obj["Q"] == 0 and obj["attribute"] is not None})
        pred, c = vote(samples, a.n)
        valid = [s for s in samples if (check(s["obj"], r.text)[1] not in ("json", "schema"))]
        ppred, pshare = majority([tuple_of(s["obj"]) for s in valid], a.n)
        best = next((s["obj"] for s in samples if s["verified"] and tuple_of(s["obj"]) == pred), None)
        gold_attr = r.attr if r.Q == 1 and isinstance(r.attr, str) else None
        recs.append({
            "req_id": r.req_id, "project": r.project, "text": r.text,
            "gold": {"F": int(r.F), "Q": int(r.Q), "attribute": gold_attr},
            "F": {"label": pred[0], "span": best.get("F_span") if pred[0] else None} if pred else None,
            "Q": {"label": pred[1], "span": best.get("Q_span") if pred[1] else None} if pred else None,
            "attribute": pred[2] if pred else None, "explanation": best["explanation"] if best else None,
            "samples_verified": sum(s["verified"] for s in samples), "confidence": c,
            "needs_review": c < a.tau, "retrieved": tr.req_id.iloc[nb[i]].tolist(),
            "plain_vote": {"tuple": ppred, "share": pshare},
            "span_agreement": {lab: span_agreement(samples, lab) if a.variant != "noevidence" else None for lab in ("F", "Q")},
            **({"stated_confidence": [stated_conf(s["obj"]) for s in samples]} if a.variant == "verbal" else {}),
            **({"probe": {lab: probe_record(raw_probe[2 * i + j], r.text) for j, lab in enumerate(("F", "Q"))}} if a.probe else {}),
            "samples": [{"raw": s["raw"], "verified": s["verified"], "verified_basic": s["verified_basic"],
                         "reason": s["reason"], "attr_dropped": s["attr_dropped"],
                         "tuple": tuple_of(s["obj"]) if s in valid else None} for s in samples]})
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for rec in recs: f.write(json.dumps(rec) + "\n")
    return recs, gen_s, {**info, "n_test": len(te)}


def probe_record(raws, text):
    res = [(parse(x), *verify_probe(parse(x), text)) for x in raws]
    return {"present": sum(ok and o["present"] == 1 for o, ok, _ in res), "absent": sum(ok and o["present"] == 0 for o, ok, _ in res),
            "n": len(raws), "reasons": dict(Counter(w for _, ok, w in res if not ok)),
            "spans": [o["span"] for o, ok, _ in res if ok and o["present"] == 1], "raw": raws}


def summarise(recs):
    pred = [r for r in recs if r["F"] is not None]
    l1 = [r["F"]["label"] == r["gold"]["F"] and r["Q"]["label"] == r["gold"]["Q"] for r in pred]
    fq = [r for r in recs if r["gold"]["F"] == 1 and r["gold"]["Q"] == 1]
    both = [r["F"] is not None and r["F"]["label"] == 1 and r["Q"]["label"] == 1 for r in fq]
    reasons = Counter(s["reason"] for r in recs for s in r["samples"])
    return {"items": len(recs), "items_with_prediction": len(pred),
            "sample_verified_rate": round(sum(s["verified"] for r in recs for s in r["samples"]) /
                                          sum(len(r["samples"]) for r in recs), 3),
            "L1_exact_acc": round(float(np.mean(l1)), 3) if l1 else None,
            "FQ_items": len(fq), "FQ_both_correct": sum(both),
            "mean_confidence": round(float(np.mean([r["confidence"] for r in recs])), 3),
            "needs_review": sum(r["needs_review"] for r in recs), "sample_reasons": dict(reasons),
            "verified_vs_plain_same_prediction": sum(
                (tuple(r["plain_vote"]["tuple"]) if r["plain_vote"]["tuple"] else None) ==
                ((r["F"]["label"], r["Q"]["label"], r["attribute"]) if r["F"] else None) for r in recs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--fold", default="P01", help="'eval' (the 13 evaluation projects), 'dev' (P02,P04), 'dalpiaz', 'itrust', or comma-separated test project ids")
    ap.add_argument("--limit", type=int, default=0, help="max test items per fold (0 = all)")
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.6, help="placeholder until tuned on the validation project")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--dtype", default=None, help="vllm only")
    ap.add_argument("--decoding", choices=["constrained", "free"], default="constrained")
    ap.add_argument("--no-fq-guarantee", action="store_true", help="ablation: do not force an F+Q example")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v4", help="system prompt version (v4 frozen 2026-11-06)")
    ap.add_argument("--probe", action="store_true", help="ablation: also ask one present/absent question per aspect")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="hf only: 4-bit NF4 weights (bitsandbytes)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="evident",
                    help="evident (method), noevidence (L0: labels only), verbal (S2: stated confidence; use --n 1)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from pilot import embed  # MiniLM encoder shared with the pilot
    P = D.promise(); E = embed(P.text.tolist())
    if a.fold in ("dalpiaz", "itrust"):  # external test: retrieval pool = all of PROMISE, test = the external set by project
        X = D.dalpiaz_public() if a.fold == "dalpiaz" else D.itrust()
        EX = embed(X.text.tolist()); folds = sorted(X.project.unique()); split = a.fold
    else:
        folds = ([f for f in sorted(P.project.unique()) if f not in DEV_PROJECTS] if a.fold == "eval"
                 else list(DEV_PROJECTS) if a.fold == "dev" else a.fold.split(","))
        assert set(folds) <= set(P.project), f"unknown project in {folds}"
        split = "dev" if set(folds) <= set(DEV_PROJECTS) else "eval"
        assert split == "dev" or set(folds).isdisjoint(DEV_PROJECTS), "P02/P04 are development-only; run them with --fold dev"
    system, _ = system_and_schema(a)
    tag = (f"{a.model.split('/')[-1]}_{a.backend}_k{a.k}_n{a.n}_T{a.temperature}_s{a.seed}_p{prompt_hash(system)[:6]}_{a.decoding}"
           + ("_q4" if a.quant == "4bit" else "") + ("" if a.variant == "evident" else f"_{a.variant}")
           + ("_noFQ" if a.no_fq_guarantee else "") + ("_probe" if a.probe else "") + f"_{split}")
    out_dir = Path(a.out) if a.out else ROOT / "results/evident" / tag; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = a.decoding == "constrained"
    llm = (HFBackend(a.model, a.seed, constrained=con, quant=a.quant) if a.backend == "hf"
           else VLLMBackend(a.model, a.seed, a.dtype or "auto", constrained=con))
    load_s = time.time() - t0
    meta = {"model": a.model, "backend": a.backend, "quant": a.quant, "variant": a.variant, "decoding": a.decoding,
            "prompt": a.prompt, "prompt_hash": prompt_hash(system), "split": split,
            "span_rules": {"min_content_words": 1, "max_span_frac": MAX_SPAN_FRAC, "min_span_cap": MIN_SPAN_CAP},
            "fq_guarantee": not a.no_fq_guarantee, "probe": a.probe, "probe_hash": PROBE_HASH if a.probe else None, "k": a.k, "n": a.n,
            "temperature": a.temperature, "tau": a.tau, "seed": a.seed, "max_tokens": a.max_tokens,
            "limit": a.limit, "load_s": round(load_s, 1), "python": platform.python_version(),
            "torch": torch.__version__, "device": getattr(llm, "dev", "cuda"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "folds": {}}
    all_recs = []
    for fp in folds:
        if split in ("dalpiaz", "itrust"):
            m = (X.project == fp).values
            recs, gen_s, info = run_items(P, E, X[m].reset_index(drop=True), EX[m], fp, llm, a, out_dir,
                                          {"n_train": len(P), "pool": "all PROMISE"})
        else:
            recs, gen_s, info = run_fold(P, E, fp, llm, a, out_dir)
        all_recs += recs
        meta["folds"][fp] = {**info, "gen_s": round(gen_s, 1), **summarise(recs)}
        print(fp, json.dumps(meta["folds"][fp]), flush=True)
    meta["overall"] = summarise(all_recs)
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=1, default=str)
    print(json.dumps(meta["overall"], indent=1)); print("saved to", out_dir)


if __name__ == "__main__":
    main()
