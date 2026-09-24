"""Centralized configuration for the HRKB pipeline.

All hyperparameters mirror the paper:
  * prototype count K = K_h + K_b = 23 + 7 = 30
  * retrieval depth k = 5
  * knowledge injected into CoT steps 2-3 (context + mechanism)
  * label-aware offline construction, label-agnostic inference
"""

import os

# ── Encoders ─────────────────────────────────────────────────────────────
CLIP_MODEL = "openai/clip-vit-base-patch32"   # clustering, retrieval, frozen vision
LANGUAGE_MODEL = "xlm-roberta-large"           # trainable text encoder
NUM_LABELS = 2

# ── Dataset presets (paths relative to code/) ────────────────────────────
DATASET_PATHS = {
    "FHM": {
        "train_data": "../data_FHM/train_data_FHM.json",
        "val_data":   "../data_FHM/val_data_FHM.json",
        "test_data":  "../data_FHM/test_data_FHM.json",
        "image_dir":  "../data_FHM/FHM",
        "hrkb_output": "hrkb_FHM.json",
    },
    "MAMI": {
        "train_data": "../data_MAMI/train_data_MAMI.json",
        "val_data":   "../data_MAMI/val_data_MAMI.json",
        "test_data":  "../data_MAMI/test_data_MAMI.json",
        "image_dir":  "../data_MAMI/MAMI",
        "hrkb_output": "hrkb_MAMI.json",
    },
    "HarMeme": {
        "train_data": "../data_HarMeme/train_data_HarMeme.json",
        "val_data":   "../data_HarMeme/val_data_HarMeme.json",
        "test_data":  "../data_HarMeme/test_data_HarMeme.json",
        "image_dir":  "../data_HarMeme/HarMeme",
        "hrkb_output": "hrkb_HarMeme.json",
    },
    "PrideMM": {
        "train_data": "../data_PrideMM/train_data_PrideMM.json",
        "val_data":   "../data_PrideMM/val_data_PrideMM.json",
        "test_data":  "../data_PrideMM/test_data_PrideMM.json",
        "image_dir":  "../data_PrideMM/img",
        "hrkb_output": "hrkb_PrideMM.json",
    },
}
DEFAULT_DATASET = os.environ.get("HRKB_DATASET", "FHM")

# ── Stage 1: HRKB construction ───────────────────────────────────────────
HRKB = {
    "k_harmful": 23,        # informed by the 23-category rhetorical taxonomy
    "k_benign": 7,          # chosen by silhouette sweep over {3, ..., 10}
    "top_m": 5,             # centroid-nearest memes summarized per cluster
    "kmeans_restarts": 5,   # k-means++ with 5 restarts
    "batch_size": 16,       # CLIP encoding batch
    "max_text_chars": 512,
    "benign_sweep": list(range(3, 11)),  # silhouette-sweep candidates for K_b
    # Label-split construction variants (Table "label"):
    #   "separate" -> K_h=23 / K_b=7  (default, ours)
    #   "joint"    -> single K-Means over all samples, no label split
    #   "balanced" -> K_h=K_b=15
    "label_split": "separate",
}

# MLLM backend for prototype summarization (label-agnostic output only)
QWEN_API_URL   = os.environ.get("HRKB_API_URL",   "https://api.siliconflow.cn/v1/chat/completions")
QWEN_API_KEY   = os.environ.get("HRKB_API_KEY",   "")
QWEN_API_MODEL = os.environ.get("HRKB_API_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

PROTOTYPE_SUMM_PROMPT = (
    "The following are {n} memes from the same rhetorical cluster. "
    "Each meme is shown as an image paired with its OCR caption text. "
    "Identify and summarize the single core rhetorical pattern or propaganda "
    "technique they all share. Output exactly one concise sentence (max 30 words). "
    "Describe only the shared rhetorical mechanism; do NOT classify as harmful or "
    "benign and do NOT name any category or verdict."
)

# ── Stage 2: Retrieval + CoT generation ──────────────────────────────────
RETRIEVAL = {
    "top_k": 5,                          # retrieval depth
    "mode": "prototype",                 # prototype | instance | medoid | none
}

COT = {
    "hrkb_top_k": int(os.environ.get("HRKB_TOP_K", "5")),
    "clip_model": CLIP_MODEL,
    "local_model": os.environ.get("HRKB_LOCAL_MODEL", "Qwen/Qwen2.5-VL-14B-Instruct"),
    # Knowledge prefix is injected only into these CoT steps (1-indexed):
    #   1 = visual perception, 2 = context association,
    #   3 = mechanism reasoning, 4 = summary induction
    "inject_steps": [2, 3],
}

# ── Stage 3: detection pipeline training ─────────────────────────────────
TRAIN = {
    "lr": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "freeze_vision": True,          # frozen CLIP ViT-B/32
    "unfreeze_last_n_layers": 2,    # text backbone fine-tuning
    "batch_size": 16,
    "epochs": 10,
    "early_stop_patience": 4,
    "label_smoothing": 0.1,
    "dropout": 0.3,
    "max_len": 512,
    "seeds": [42, 2024, 2025, 7, 100],  # five-seed reporting
}
