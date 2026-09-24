#!/usr/bin/env bash
# Full HRKB pipeline on PrideMM (LGBTQ+ harmful memes).
set -e
export HRKB_DATASET=PrideMM
cd "$(dirname "$0")/../code"

python build_kb.py --dataset PrideMM
python gen_cot.py --dataset PrideMM
python train.py --dataset PrideMM --seeds 42 2024 2025 7 100
