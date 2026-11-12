# EVIDENT-RC: anonymous replication package

Code, prompts, per-requirement predictions and analysis scripts for the paper "Show the Evidence: Evidence-Verified LLM Classification and Calibrated Confidence for Mixed Functional and Quality Requirements on Unseen Projects" (under double-blind review).

## Check the results (CPU, a few minutes, no GPU)

    ./reproduce.sh

Builds a Python environment and recomputes the RQ1 and RQ2 tables from the saved predictions in `results/`.

## Re-run the experiments (one CUDA GPU with at least 10 GB, about 7 hours)

    ./reproduce.sh --setup   # downloads PROMISE-reclass, Dalpiaz-4 and iTrust at pinned commits
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
| `annotation/` | Annotation sheets |

Datasets are not redistributed; `--setup` fetches them from the original authors' repositories (GPL-3.0). The LLM is `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (revision f15c379fb32b), run with 4-bit weights.
