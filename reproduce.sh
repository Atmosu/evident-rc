#!/usr/bin/env bash
# Executable entry point. Recomputes every table of the paper from the saved per-requirement predictions (CPU, a few minutes).
#   ./reproduce.sh          verify results from results/
#   ./reproduce.sh --setup  also download the public datasets (needed only to re-run training or the LLM)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
if [ "$1" = "--setup" ]; then
  mkdir -p data && cd data
  [ -d NoRBERT ] || { git clone https://github.com/tobhey/NoRBERT.git NoRBERT && git -C NoRBERT checkout 00c2f566c5dbd8a1773629b3145956887952b223; }
  [ -d dalpiaz ] || { git clone https://github.com/explainable-re/RE-2019-Materials.git dalpiaz && git -C dalpiaz checkout 23ff85b1aa49b03aa36e3d25fec0ce199b9403cb; }
  [ -d ftlr ] || { git clone https://github.com/tobhey/finegrained-traceability.git ftlr && git -C ftlr checkout 197cf2f395e9e90636436f30d6383465cb9e9f63; }
  cd ..
fi
R=results/evident/Llama-3.1-8B-Instruct-bnb-4bit_hf_k8_
M=${R}n10_T0.7_s42_p443b9c_constrained_q4_eval
.venv/bin/python src/compare_baselines.py "$M" ${R}n10_T0.7_s42_p443b9c_constrained_q4_dalpiaz
.venv/bin/python src/rq2_compare.py "$M" ${R}n1_T0.7_s42_p6b8f57_constrained_q4_verbal_eval --B 2000 --out results/rq2_compare_n5.csv
.venv/bin/python src/compare_l1.py "$M" --n 5 --out results/rq1_compare_n5.csv
cat "$M/faithfulness/summary.json" | tail -45; cat results/contamination/summary.json | tail -30
