#!/bin/bash
# Second chain: mechanistic experiments, run after run_chain.sh has finished.
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
while [ ! -f logs/chain_done ]; do sleep 30; done
python src/exp3_acts.py > logs/exp3_acts.log 2>&1
python src/exp3_ablate.py > logs/exp3_ablate.log 2>&1
python src/guess.py gemma12 exp3 > logs/guess_gemma_exp3.log 2>&1
python src/guess.py qwen14 exp3 > logs/guess_qwen_exp3.log 2>&1
echo CHAIN2_DONE > logs/chain2_done
