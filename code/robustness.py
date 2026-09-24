#!/usr/bin/env python3
"""Robustness under synthetic reasoning corruption (Fig. 3/7 in the paper).

Corrupts 30% of the *critical tokens* in the context/mechanism CoT segments
with plausible-but-wrong entities, then measures the accuracy drop with and
without the confidence gate, plus the shift in gate similarity s.

Configurations (matching the paper's Fig. 7a):
    w/o CoT   : corrupt the raw meme text, no CoT segments
    w/o Gating: corrupt CoT, disable the confidence gate (always cross-modal)
    Full      : corrupt CoT, gated fusion active

Usage
-----
    python robustness.py --dataset FHM --seed 42
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from hrkb.config import DATASET_PATHS, TRAIN
from hrkb.dataset import Collator, MemeDataset
from hrkb.utils import load_json, save_json
from train import build_model

WRONG_ENTITIES = [
    "the 2012 budget", "Paris", "a famous actor", "the moon landing",
    "a sports final", "a cooking recipe", "the weather forecast",
    "an unrelated election", "a travel vlog", "a school play",
]


def corrupt(text: str, ratio: float = 0.3, rng=None) -> str:
    rng = rng or random
    words = text.split()
    if not words:
        return text
    # Critical tokens: capitalized, numeric, or long — the entity-bearing words.
    crit = [i for i, w in enumerate(words)
            if w[:1].isupper() or any(c.isdigit() for c in w) or len(w) >= 5]
    if not crit:
        crit = list(range(len(words)))
    n = max(1, int(len(crit) * ratio))
    for i in rng.sample(crit, min(n, len(crit))):
        words[i] = rng.choice(WRONG_ENTITIES)
    return " ".join(words)


def _make_variant(test_data, corr_cot, corr_text):
    out = []
    for item in test_data:
        it = dict(item)
        v = item.get("cot_variants", {}).get("prototype_k5", {})
        it["cot_variants"] = dict(item.get("cot_variants", {}))
        it["cot_variants"]["prototype_k5"] = {
            "visual": v.get("visual", ""),
            "context": corrupt(v.get("context", "")) if corr_cot else v.get("context", ""),
            "mechanism": corrupt(v.get("mechanism", "")) if corr_cot else v.get("mechanism", ""),
            "summary": v.get("summary", ""),
        }
        if corr_text:
            it["text"] = corrupt(item.get("text", ""))
        out.append(it)
    return out


def _evaluate(model, dl, device, use_gate=True):
    model.eval()
    preds, golds, s_vals = [], [], []
    with torch.no_grad():
        for batch in dl:
            logits, s = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["token_type_ids"].to(device),
                use_gate=use_gate,
            )
            preds.extend(logits.argmax(-1).cpu().numpy())
            golds.extend(batch["labels"].cpu().numpy())
            s_vals.extend(s.cpu().numpy())
    acc = float(np.mean(np.array(preds) == np.array(golds))) * 100
    return acc, np.mean(s_vals)


def run(dataset, seed, device):
    rng = random.Random(seed)
    cfg = DATASET_PATHS[dataset]
    test_data = load_json(cfg["test_data"])

    model, tokenizer, processor = build_model(device)
    save_path = os.path.join("trained_models", f"data_{dataset}.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    else:
        print(f"WARN: {save_path} not found; running with untrained weights.")

    coll = Collator(tokenizer)
    tmp_dir = os.path.join("results", "_robust_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    def loader(items, tag):
        path = os.path.join(tmp_dir, f"{dataset}_{tag}.json")
        save_json(path, items)
        ds = MemeDataset(path, processor, tokenizer, cfg["image_dir"],
                         TRAIN["max_len"], "test", False, "prototype_k5")
        return DataLoader(ds, sampler=SequentialSampler(ds),
                          batch_size=TRAIN["batch_size"], collate_fn=coll)

    print(f"\n=== {dataset} robustness ===")
    # w/o CoT
    clean_nocot = _make_variant(test_data, corr_cot=False, corr_text=False)
    for it in clean_nocot:
        it["cot_variants"]["prototype_k5"] = {"visual": "", "context": "",
                                              "mechanism": "", "summary": ""}
    corr_nocot = _make_variant(test_data, corr_cot=False, corr_text=True)
    for it in corr_nocot:
        it["cot_variants"]["prototype_k5"] = {"visual": "", "context": "",
                                              "mechanism": "", "summary": ""}

    # w/o Gating (CoT)
    clean_cot = _make_variant(test_data, corr_cot=False, corr_text=False)
    corr_cot = _make_variant(test_data, corr_cot=True, corr_text=False)

    results = {}
    for name, clean, corr, gate in [
        ("w/o CoT", clean_nocot, corr_nocot, True),
        ("w/o Gating", clean_cot, corr_cot, False),
        ("Full", clean_cot, corr_cot, True),
    ]:
        acc_c, s_c = _evaluate(model, loader(clean, f"{name}_clean"), device, use_gate=gate)
        acc_x, s_x = _evaluate(model, loader(corr, f"{name}_corr"), device, use_gate=gate)
        results[name] = {"clean_acc": round(acc_c, 2), "corr_acc": round(acc_x, 2),
                         "drop": round(acc_c - acc_x, 2),
                         "s_clean": round(float(s_c), 3), "s_corr": round(float(s_x), 3)}
        print(f"{name:<12} clean={acc_c:.2f}  corr={acc_x:.2f}  "
              f"drop={acc_c - acc_x:.2f}  s: {s_c:.3f}->{s_x:.3f}")

    os.makedirs("results", exist_ok=True)
    save_json(f"results/robustness_{dataset}.json", results)
    return results


def main():
    p = argparse.ArgumentParser(description="Synthetic corruption robustness (Fig. 7).")
    p.add_argument("--dataset", default="FHM", choices=list(DATASET_PATHS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    run(args.dataset, args.seed, args.device)


if __name__ == "__main__":
    main()
