#!/usr/bin/env bash
# All LLM and fine-tuning runs of the paper (one CUDA GPU with at least 10 GB). Run after ./reproduce.sh --setup.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python; pip_extra="xgrammar==0.2.8 bitsandbytes==0.50.2"; .venv/bin/pip install -q $pip_extra
M=${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit}; C="--model $M --quant 4bit --prompt v4"
$PY src/evident.py $C --fold eval --n 10 --k 8
$PY src/evident.py $C --fold dalpiaz --n 10 --k 8
$PY src/evident.py $C --fold dalpiaz --n 5 --k 8 --variant noevidence
$PY src/evident.py $C --fold dalpiaz --n 5 --k 0
$PY src/evident.py $C --fold dalpiaz --n 1 --k 8 --variant verbal
$PY src/evident.py $C --fold itrust --n 5 --k 8
$PY src/evident.py $C --fold P01 --n 5 --k 8 --decoding free
$PY src/evident.py $C --fold eval --n 5 --k 0
$PY src/evident.py $C --fold eval --n 5 --k 8 --variant noevidence
$PY src/evident.py $C --fold eval --n 1 --k 8 --variant verbal
$PY src/evident.py $C --fold eval --n 5 --k 8 --no-fq-guarantee
$PY src/evident.py $C --fold eval --n 5 --k 8 --seed 43
$PY src/evident.py $C --fold eval --n 5 --k 8 --seed 44
$PY src/faithfulness.py results/evident/$(basename $M)_hf_k8_n10_T0.7_s42_p443b9c_constrained_q4_eval --model $M --quant 4bit --n 5
for m in minilm bert roberta; do $PY src/transformer_baselines.py --model $m; $PY src/transformer_baselines.py --model $m --folds itrust; done
$PY src/contamination.py --model $M --quant 4bit
