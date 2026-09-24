#!/usr/bin/env bash
# Full HRKB pipeline on Facebook Hateful Memes (FHM).
set -e
export HRKB_DATASET=FHM
cd "$(dirname "$0")/../code"

python build_kb.py --dataset FHM          # Stage 1: build the HRKB (K=30)
python gen_cot.py --dataset FHM           # Stage 2: prototype-grounded CoT
python train.py --dataset FHM --seeds 42 2024 2025 7 100   # Stage 3: detector
