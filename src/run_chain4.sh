#!/bin/bash
# Fourth chain: ablation generation (after the float16 sanitisation fix), then guessing of the ablated stories.
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f logs/chain3_done ]; do sleep 30; done
python src/exp3_ablate.py > logs/exp3_ablate.log 2>&1
python src/guess.py gemma12 exp3 > logs/guess_gemma_exp3.log 2>&1
python src/guess.py qwen14 exp3 > logs/guess_qwen_exp3.log 2>&1
echo CHAIN4_DONE > logs/chain4_done
