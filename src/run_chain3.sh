#!/bin/bash
# Third chain: finish the twist experiment (remaining conditions) after the mechanistic chain.
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f logs/chain2_done ]; do sleep 30; done
python src/exp2_generate.py > logs/exp2_generate.log 2>&1
python src/guess.py gemma12 exp2 > logs/guess_gemma_exp2.log 2>&1
python src/guess.py qwen14 exp2 > logs/guess_qwen_exp2.log 2>&1
echo CHAIN3_DONE > logs/chain3_done
