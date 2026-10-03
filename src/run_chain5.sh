#!/bin/bash
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
while [ ! -f logs/chain4_done ]; do sleep 30; done
python src/exp3_lens_fix.py > logs/exp3_lens_fix.log 2>&1
echo CHAIN5_DONE > logs/chain5_done
