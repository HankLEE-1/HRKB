#!/usr/bin/env python3
"""Stage 1 — Build the Harmful Rhetorical Knowledge Base (HRKB).

Encodes training memes with CLIP, applies K-Means clustering (separately per
label group by default), and summarizes each cluster into a one-sentence
rhetorical prototype via an MLLM API call. Supports label-split variants and
multi-dataset pooling (unified HRKB).

Examples
--------
    python build_kb.py --dataset FHM
    python build_kb.py --dataset MAMI --label-split balanced
    python build_kb.py --dataset FHM --label-split joint --k-total 30
    python build_kb.py --datasets FHM MAMI HarMeme PrideMM --k-total 60 \\
                       --output hrkb_unified.json
"""

import argparse
import logging
import os
import time

from hrkb.config import CLIP_MODEL, DATASET_PATHS, HRKB
from hrkb.utils import format_time, setup_logger

logger = setup_logger(__name__, log_file=None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HRKB Stage 1 — build the Harmful Rhetorical Knowledge Base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset", default=os.environ.get("HRKB_DATASET", "FHM"),
                   help="Single preset dataset name.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Pool multiple datasets into one unified HRKB "
                        "(used for the cross-domain reuse experiment).")

    p.add_argument("--train-data", default=None)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--output", default=None)

    p.add_argument("--clip-model", default=CLIP_MODEL)
    p.add_argument("--k-harmful", type=int, default=HRKB["k_harmful"])
    p.add_argument("--k-benign", type=int, default=HRKB["k_benign"])
    p.add_argument("--k-total", type=int, default=None,
                   help="Total clusters for joint / unified construction.")
    p.add_argument("--label-split", default=HRKB["label_split"],
                   choices=["separate", "joint", "balanced"])
    p.add_argument("--sweep-benign", action="store_true",
                   help="Pick K_b by silhouette sweep over benign clusters.")
    p.add_argument("--top-m", type=int, default=HRKB["top_m"])
    p.add_argument("--restarts", type=int, default=HRKB["kmeans_restarts"])
    p.add_argument("--batch-size", type=int, default=HRKB["batch_size"])
    p.add_argument("--max-text-chars", type=int, default=HRKB["max_text_chars"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    args = p.parse_args()

    # Resolve default paths from a single-dataset preset when not given.
    preset = DATASET_PATHS.get(args.dataset, {})
    args.train_data = args.train_data or preset.get("train_data")
    args.image_dir = args.image_dir or preset.get("image_dir")
    args.output = args.output or preset.get("hrkb_output")
    return args


def main() -> None:
    from hrkb.build import build
    from hrkb.utils import load_json

    args = parse_args()
    logger.info("=" * 60)
    logger.info("HRKB  ·  Stage 1: HRKB Construction")
    logger.info("=" * 60)
    logger.info("Dataset(s) : %s", args.datasets or args.dataset)
    logger.info("Train data : %s", args.train_data)
    logger.info("Image dir  : %s", args.image_dir)
    logger.info("Output     : %s", args.output)
    logger.info("Label split: %s", args.label_split)

    t0 = time.time()

    if args.datasets:
        # Unified multi-domain HRKB: pool training splits, then cluster jointly.
        pooled, image_dirs = [], []
        for ds in args.datasets:
            if ds not in DATASET_PATHS:
                raise ValueError(f"Unknown dataset: {ds}")
            cfg = DATASET_PATHS[ds]
            pooled += load_json(cfg["train_data"])
            image_dirs.append(cfg["image_dir"])
        # The unified builder clusters the pooled corpus as one label group set;
        # we write a temporary combined train file consumed by build().
        import json
        tmp = args.output + ".pooled.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pooled, f)
        # images may live in different folders; build() uses a single image_dir.
        # For unified runs, image files must be reachable from one of the dirs.
        args.train_data = tmp
        args.image_dir = image_dirs[0]
        args.label_split = "joint" if args.label_split == "separate" else args.label_split
        args.k_total = args.k_total or 60

    build(args)

    logger.info("=" * 60)
    logger.info("HRKB construction complete.  Time: %s", format_time(time.time() - t0))
    logger.info("Saved to: %s", args.output)


if __name__ == "__main__":
    main()
