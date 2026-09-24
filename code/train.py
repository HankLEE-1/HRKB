#!/usr/bin/env python3
"""Stage 3 — Train the gated multimodal classifier (reference detector).

Trains the detector over the CoT-enriched dataset, reporting macro-F1 / Acc /
AUC. The CoT variant (retrieval mode) and seed are selectable so the retrieval
ablation and five-seed reporting can reuse this entry point.
"""

import argparse
import math
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn import metrics
from tqdm import tqdm
from transformers import (
    AdamW, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel,
    XLMRobertaModel, get_cosine_schedule_with_warmup,
)

from hrkb.config import CLIP_MODEL, DATASET_PATHS, LANGUAGE_MODEL, NUM_LABELS, TRAIN
from hrkb.dataset import Collator, MemeDataset
from hrkb.model import GatedMultimodalClassifier
from hrkb.utils import format_time, set_seed, setup_logger

logger = setup_logger(__name__, log_file=None)


def _resolve_paths(dataset: str) -> dict:
    cfg = DATASET_PATHS[dataset]
    return {
        "train_data": cfg["train_data"],
        "val_data": cfg["val_data"],
        "test_data": cfg["test_data"],
        "image_dir": cfg["image_dir"],
        "save_name": f"data_{dataset}.pth",
    }


def build_model(device):
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    bert = XLMRobertaModel.from_pretrained(LANGUAGE_MODEL)
    vit = CLIPVisionModel.from_pretrained(CLIP_MODEL)

    for p in vit.parameters():
        p.requires_grad = False

    model = GatedMultimodalClassifier(vit, bert, NUM_LABELS,
                                      dropout=TRAIN["dropout"])
    # Freeze lower layers of the text backbone.
    for p in model.bert.parameters():
        p.requires_grad = False
    if getattr(model.bert, "pooler", None) is not None:
        for p in model.bert.pooler.parameters():
            p.requires_grad = True
    n = TRAIN["unfreeze_last_n_layers"]
    for layer in model.bert.encoder.layer[-n:]:
        for p in layer.parameters():
            p.requires_grad = True
    model.to(device)
    return model, tokenizer, processor


def evaluate(model, dl, device):
    model.eval()
    preds, golds, probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Eval"):
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["token_type_ids"].to(device),
            )
            preds.extend(logits.argmax(-1).cpu().numpy())
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            golds.extend(batch["labels"].cpu().numpy())

    acc = round(metrics.accuracy_score(golds, preds) * 100, 2)
    f1 = round(metrics.f1_score(golds, preds, average="macro") * 100, 2)
    try:
        auc = round(metrics.roc_auc_score(golds, probs) * 100, 2)
    except Exception:
        auc = 0.0
    return {"acc": acc, "f1": f1, "auc": auc}


def train_and_eval(dataset, variant, seed, device="cuda:0", epochs=None,
                   batch_size=None, lr=None, do_train=True):
    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    paths = _resolve_paths(dataset)
    epochs = epochs or TRAIN["epochs"]
    batch_size = batch_size or TRAIN["batch_size"]
    lr = lr or TRAIN["lr"]

    model, tokenizer, processor = build_model(device)

    train_ds = MemeDataset(paths["train_data"], processor, tokenizer,
                           paths["image_dir"], TRAIN["max_len"], "train", True, variant)
    val_ds = MemeDataset(paths["val_data"], processor, tokenizer,
                         paths["image_dir"], TRAIN["max_len"], "val", False, variant)
    test_ds = MemeDataset(paths["test_data"], processor, tokenizer,
                          paths["image_dir"], TRAIN["max_len"], "test", False, variant)

    coll = Collator(tokenizer)
    train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds),
                          batch_size=batch_size, collate_fn=coll)
    val_dl = DataLoader(val_ds, sampler=SequentialSampler(val_ds),
                        batch_size=batch_size, collate_fn=coll)
    test_dl = DataLoader(test_ds, sampler=SequentialSampler(test_ds),
                         batch_size=batch_size, collate_fn=coll)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=TRAIN["weight_decay"])
    total_steps = math.ceil(len(train_dl)) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * TRAIN["warmup_ratio"]), total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=TRAIN["label_smoothing"])

    save_path = os.path.join("trained_models", paths["save_name"])
    best_val = {"f1": -1}

    if do_train:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_dl, desc=f"Train e{epoch + 1}"):
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["pixel_values"].to(device),
                    batch["token_type_ids"].to(device),
                )
                loss = loss_fn(logits, batch["labels"].to(device).long())
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            val_rec = evaluate(model, val_dl, device)
            logger.info("Epoch %d  val %s", epoch + 1, val_rec)
            if val_rec["f1"] > best_val["f1"]:
                best_val = val_rec
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    test_rec = evaluate(model, test_dl, device)
    return {"test": test_rec, "best_val": best_val}


def parse_args():
    p = argparse.ArgumentParser(description="HRKB Stage 3 — train the detector.")
    p.add_argument("--dataset", default=os.environ.get("HRKB_DATASET", "FHM"),
                   choices=list(DATASET_PATHS.keys()))
    p.add_argument("--variant", default="prototype_k5",
                   help="CoT variant produced by gen_cot.py.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Multiple seeds for five-seed reporting.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds or [args.seed]
    results = []
    for s in seeds:
        logger.info("==== Seed %d ====", s)
        r = train_and_eval(args.dataset, args.variant, s, args.device,
                           args.epochs, args.batch_size, args.lr)
        results.append(r["test"])
        logger.info("Seed %d -> %s", s, r["test"])

    if len(results) > 1:
        import numpy as np
        agg = {k: (round(float(np.mean([r[k] for r in results])), 2),
                   round(float(np.std([r[k] for r in results])), 2))
               for k in ("f1", "acc", "auc")}
        logger.info("Mean (+- std) over %d seeds: %s", len(results), agg)


if __name__ == "__main__":
    main()
