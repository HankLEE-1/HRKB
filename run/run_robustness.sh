#!/usr/bin/env bash
# Robustness under synthetic reasoning corruption (Fig. 7).
set -e
cd "$(dirname "$0")/../code"

DS=${1:-FHM}
python robustness.py --dataset "$DS" --seed 42
