"""Shared utilities: I/O, seeding, logging, data-field helpers."""

import datetime
import json
import logging
import os
import random

import numpy as np
import torch


# ── Reproducibility ─────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── I/O ─────────────────────────────────────────────────────────────────

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data, indent: int = 2, ensure_ascii: bool = False) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


# ── Logging ─────────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: str = "log",
                 log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def format_time(elapsed: float) -> str:
    return str(datetime.timedelta(seconds=int(round(elapsed))))


# ── Data-field helpers ──────────────────────────────────────────────────

def get_label(item: dict) -> int:
    """Binarized harmful-vs-benign label (HarMeme tiers merged upstream)."""
    for key in ("label", "labels", "class", "target", "metaphor occurrence"):
        if key in item:
            try:
                return int(item[key])
            except (ValueError, TypeError):
                v = str(item[key]).strip().lower()
                return 1 if v in ("hate", "harmful", "offensive",
                                  "yes", "true", "1") else 0
    return 0


def get_file_name(item: dict) -> str:
    for key in ("file_name", "image", "img", "image_name", "filename"):
        if key in item and item[key]:
            return str(item[key])
    return ""


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)
