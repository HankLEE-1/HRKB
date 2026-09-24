#!/usr/bin/env bash
# Full HRKB pipeline on HarMeme (harmful memes in COVID-19 & US politics).
set -e
export HRKB_DATASET=HarMeme
cd "$(dirname "$0")/../code"

python build_kb.py --dataset HarMeme
python gen_cot.py --dataset HarMeme
python train.py --dataset HarMeme --seeds 42 2024 2025 7 100
