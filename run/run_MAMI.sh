#!/usr/bin/env bash
# Full HRKB pipeline on MAMI (SemEval-2022 Task 5).
set -e
export HRKB_DATASET=MAMI
cd "$(dirname "$0")/../code"

python build_kb.py --dataset MAMI
python gen_cot.py --dataset MAMI
python train.py --dataset MAMI --seeds 42 2024 2025 7 100
