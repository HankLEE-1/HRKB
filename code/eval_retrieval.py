#!/usr/bin/env python3
"""Controlled retrieval-method comparison (Table 2 in the paper).

Runs the *same* detection pipeline under five retrieval strategies and reports
macro-F1 (five-seed mean) per benchmark:

    none          -> no retrieved knowledge
    instance_k5   -> top-5 instance retrieval
    instance_k10  -> top-10 instance retrieval
    medoid_k5     -> top-5 cluster medoids
    prototype_k5  -> HRKB prototypes (K=30)

Usage
-----
    python eval_retrieval.py --dataset FHM --seeds 42 2024 2025 7 100
    python eval_retrieval.py --dataset FHM --plot-only   # reuse saved results
"""

import argparse
import json
import os

from hrkb.cot import generate
from hrkb.config import COT, DATASET_PATHS
from train import train_and_eval

STRATEGIES = [
    ("none", "none", 0),
    ("instance", "instance_k5", 5),
    ("instance", "instance_k10", 10),
    ("medoid", "medoid_k5", 5),
    ("prototype", "prototype_k5", 5),
]


def run(dataset, seeds, device, inject_steps=None):
    cfg = DATASET_PATHS[dataset]
    data_dir = os.path.dirname(cfg["train_data"])
    inject_steps = inject_steps or COT["inject_steps"]
    rows = []

    for mode, variant, top_k in STRATEGIES:
        print(f"\n=== {dataset} :: {variant} ===")
        generate(data_dir=data_dir, image_dir=cfg["image_dir"],
                 hrkb_path=cfg["hrkb_output"], mode=mode, top_k=top_k,
                 variant=variant, inject_steps=inject_steps,
                 clip_model=COT["clip_model"], workers=1)
        per_seed = []
        for s in seeds:
            r = train_and_eval(dataset, variant, s, device)
            per_seed.append(r["test"]["f1"])
        rows.append({"method": variant, "seeds": per_seed,
                     "f1_mean": round(sum(per_seed) / len(per_seed), 2)})
        print(f"  {variant}: {rows[-1]['f1_mean']} (mean F1)")

    out = {"dataset": dataset, "seeds": seeds, "rows": rows}
    os.makedirs("results", exist_ok=True)
    with open(f"results/retrieval_{dataset}.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nMethod             F1")
    for r in rows:
        print(f"{r['method']:<18} {r['f1_mean']}")
    return out


def main():
    p = argparse.ArgumentParser(description="Retrieval-method ablation (Table 2).")
    p.add_argument("--dataset", default="FHM", choices=list(DATASET_PATHS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025, 7, 100])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    if args.plot_only:
        with open(f"results/retrieval_{args.dataset}.json") as f:
            out = json.load(f)
        print("\nMethod             F1")
        for r in out["rows"]:
            print(f"{r['method']:<18} {r['f1_mean']}")
        return

    run(args.dataset, args.seeds, args.device)


if __name__ == "__main__":
    main()
