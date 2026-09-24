#!/usr/bin/env bash
# Controlled retrieval-method comparison (Table 2) + label-construction
# variants (Table 4) + unified multi-domain HRKB (Table 5).
set -e
cd "$(dirname "$0")/../code"

DS=${1:-FHM}

# --- Table 2: retrieval strategies under an identical pipeline -----------
python eval_retrieval.py --dataset "$DS" --seeds 42 2024 2025 7 100

# --- Table 4: label-aware construction vs. label-agnostic inference -------
for split in joint balanced separate; do
  python build_kb.py --dataset "$DS" --label-split "$split"
  python gen_cot.py --dataset "$DS" --variant "prototype_k5_${split}"
  python train.py --dataset "$DS" --variant "prototype_k5_${split}" --seed 42
done

# --- Table 5: unified multi-domain HRKB (pooled training splits) ----------
python build_kb.py --datasets FHM MAMI HarMeme PrideMM --k-total 60 \
  --output hrkb_unified.json
