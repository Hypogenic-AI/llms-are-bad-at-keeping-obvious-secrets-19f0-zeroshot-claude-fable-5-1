#!/bin/bash
# Runs the GPU jobs sequentially (single 48GB GPU). Each step is resumable.
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
while pgrep -f exp1_generate.py >/dev/null; do sleep 20; done
python src/guess.py gemma12 exp1 > logs/guess_gemma_exp1.log 2>&1
python src/exp3_acts.py > logs/exp3_acts.log 2>&1
python src/exp3_ablate.py > logs/exp3_ablate.log 2>&1
python src/guess.py gemma12 exp3 > logs/guess_gemma_exp3.log 2>&1
python src/exp2_generate.py > logs/exp2_generate.log 2>&1
python src/guess.py gemma12 exp2 > logs/guess_gemma_exp2.log 2>&1
python src/guess.py qwen14 exp1 exp2 exp3 > logs/guess_qwen.log 2>&1
echo CHAIN_DONE > logs/chain_done
