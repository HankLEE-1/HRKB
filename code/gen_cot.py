#!/usr/bin/env python3
"""Stage 2 — Generate knowledge-augmented Chain-of-Thought (CoT).

For each meme, retrieves a label-agnostic knowledge prefix (prototype /
instance / medoid / none) and injects it into CoT steps 2-3. Generated chains
are stored per ``--variant`` so the retrieval-method ablation can share items.

Examples
--------
    python gen_cot.py --dataset FHM
    python gen_cot.py --dataset FHM --mode instance --top-k 5 --variant instance_k5
    python gen_cot.py --dataset FHM --mode none --variant none
"""

import argparse
import os
import time

from hrkb.config import COT, DATASET_PATHS, RETRIEVAL
from hrkb.cot import generate
from hrkb.utils import format_time, setup_logger

logger = setup_logger(__name__, log_file=None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HRKB Stage 2 — knowledge-augmented CoT generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset", default=os.environ.get("HRKB_DATASET", "FHM"))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--hrkb-path", default=None)

    p.add_argument("--mode", default=RETRIEVAL["mode"],
                   choices=["prototype", "instance", "medoid", "none"])
    p.add_argument("--top-k", type=int, default=COT["hrkb_top_k"])
    p.add_argument("--variant", default=None,
                   help="Storage key for this CoT variant (default: {mode}_k{top_k}).")
    p.add_argument("--inject-steps", nargs="+", type=int,
                   default=COT["inject_steps"],
                   help="1-indexed CoT steps that receive the knowledge prefix.")
    p.add_argument("--clip-model", default=COT["clip_model"])
    p.add_argument("--local-model", default=COT["local_model"])
    p.add_argument("--workers", type=int, default=1)

    args = p.parse_args()

    preset = DATASET_PATHS.get(args.dataset, {})
    args.data_dir = args.data_dir or os.path.dirname(preset.get("train_data", ""))
    args.image_dir = args.image_dir or preset.get("image_dir", "")
    args.hrkb_path = args.hrkb_path or preset.get("hrkb_output", "")
    if args.variant is None:
        args.variant = f"{args.mode}_k{args.top_k}"
    return args


def main() -> None:
    args = parse_args()
    logger.info("=" * 60)
    logger.info("HRKB  ·  Stage 2: Knowledge-Augmented CoT Generation")
    logger.info("=" * 60)
    logger.info("Dataset   : %s", args.dataset)
    logger.info("Mode      : %s (top-k=%d, variant=%s)",
                args.mode, args.top_k, args.variant)
    logger.info("Inject    : steps %s", args.inject_steps)

    t0 = time.time()
    generate(data_dir=args.data_dir, image_dir=args.image_dir,
             hrkb_path=args.hrkb_path, mode=args.mode, top_k=args.top_k,
             variant=args.variant, inject_steps=args.inject_steps,
             clip_model=args.clip_model, workers=args.workers)
    logger.info("=" * 60)
    logger.info("CoT generation complete.  Time: %s", format_time(time.time() - t0))


if __name__ == "__main__":
    main()
