# EVIDENT-RC: anonymous replication package

Code, prompts, per-requirement predictions and analysis scripts for the paper "Show the Evidence: Evidence-Verified LLM Classification and Calibrated Confidence for Mixed Functional and Quality Requirements on Unseen Projects" (under double-blind review).

## Check the results (CPU, a few minutes, no GPU)

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    R=results/evident/Llama-3.1-8B-Instruct-bnb-4bit_hf_k8_
    .venv/bin/python src/compare_baselines.py ${R}n10_T0.7_s42_p443b9c_constrained_q4_eval ${R}n10_T0.7_s42_p443b9c_constrained_q4_dalpiaz
    .venv/bin/python src/rq2_compare.py ${R}n10_T0.7_s42_p443b9c_constrained_q4_eval ${R}n1_T0.7_s42_p6b8f57_constrained_q4_verbal_eval --B 10000

These recompute the RQ1 and RQ2 tables from the saved per-requirement predictions in `results/`.

Post-cutoff test set (82 requirements quoted from four public software tenders, October 2024 to December 2025; labels in `data/external/post_cutoff_labels.csv`, two annotators on 13 items with disagreements settled by the guideline, see `annotation/post_cutoff_adjudication_log.csv`):

    .venv/bin/python src/compare_postcutoff.py

All tables and the two confidence charts of the results-analysis report (`docs/EVIDENT-RC_Results_Analysis.pdf`) are recomputed into `results/report/` by:

    .venv/bin/python src/report_tables.py

## Re-run the experiments (one CUDA GPU with at least 10 GB, about 7 hours)

Download the datasets into `data/` (PROMISE-reclass from github.com/tobhey/NoRBERT at 00c2f566, Dalpiaz-4 from github.com/explainable-re/RE-2019-Materials at 23ff85b1, iTrust from github.com/tobhey/finegrained-traceability at 197cf2f3, cloned as `data/NoRBERT`, `data/dalpiaz`, `data/ftlr`), then run:

    jobs/run_gpu.sh

## Layout

| Path | Contents |
|---|---|
| `src/evident.py` | Pipeline: retrieval, frozen prompt v4 (hash 443b9c01876d), JSON-constrained decoding, quote check (V1), verified vote (V2) |
| `src/transformer_baselines.py` | RoBERTa-large, BERT-base and MiniLM with temperature scaling |
| `src/pilot.py`, `src/knn_vote.py` | TF-IDF + SVM, frozen MiniLM, kNN vote |
| `src/faithfulness.py`, `src/contamination.py` | Masking and sufficiency tests; memorisation probe |
| `src/metrics.py`, `src/analyze_llm.py`, `src/compare_*.py`, `src/rq2_compare.py` | Metrics, paired bootstrap, Holm correction |
| `results/` | Per-requirement predictions and result tables |
| `annotation/` | Annotation sheets, post-cutoff sources and adjudication log |
| `data/external/` | Post-cutoff requirements and labels |

Datasets are not redistributed; download them from the original authors' repositories (GPL-3.0) as described above. The LLM is `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (revision f15c379fb32b), run with 4-bit weights.
