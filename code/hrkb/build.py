"""Stage 1 — Build the Harmful Rhetorical Knowledge Base (HRKB).

Offline construction, decoupled from raw instances:

  Step 1  collect training samples (I_j, T_j, y_j) as knowledge items
  Step 2  embed by phi(j) = Norm(CLIP_img + CLIP_txt) and K-Means-cluster,
          separately for harmful vs. benign samples (or joint/balanced variants)
  Step 3  summarize each cluster into a one-sentence rhetorical prototype
          via an MLLM, over the top-m centroid-nearest memes

The result is a *label-free* prototype base B = {(P_k, mu_k)}: prototypes
describe rhetorical mechanisms, never verdicts or class names. Labels only
organize clustering offline; they are excluded from retrieval and detection.

Alongside the prototype base we persist two auxiliary artifacts used by the
controlled ablation (retrieval-method comparison):

  * an instance index  (all training embeddings) for `instance` retrieval
  * the medoid sample   (centroid-nearest meme) per cluster for `medoid` retrieval
"""

import argparse
import base64
import datetime
import io
import os
import time

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import (
    CLIP_MODEL, HRKB, PROTOTYPE_SUMM_PROMPT,
    QWEN_API_KEY, QWEN_API_MODEL, QWEN_API_URL,
)
from .utils import get_file_name, get_label, load_json, safe_text, save_json, set_seed

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Image helpers ────────────────────────────────────────────────────────

def _open_image(image_dir: str, fname: str) -> Image.Image:
    try:
        return Image.open(os.path.join(image_dir, fname)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="black")


def _encode_b64(image_dir: str, fname: str) -> str:
    path = os.path.join(image_dir, fname)
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), "black").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()


# ── MLLM summarization ───────────────────────────────────────────────────

def _summarize_cluster(samples, image_dir: str) -> str:
    fallback_texts = [safe_text(s.get("text")) for s in samples
                      if safe_text(s.get("text"))]
    fallback = (fallback_texts[0] if fallback_texts
                else "No rhetorical pattern available.")[:260]
    if not QWEN_API_KEY:
        return fallback

    content = [{"type": "text",
                "text": PROTOTYPE_SUMM_PROMPT.format(n=len(samples))}]
    for i, s in enumerate(samples, 1):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,"
                       f"{_encode_b64(image_dir, get_file_name(s))}"},
        })
        content.append({
            "type": "text",
            "text": f"Meme {i} OCR: {safe_text(s.get('text')) or '(no text)'}",
        })

    payload = {
        "model": QWEN_API_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 80,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}",
               "Content-Type": "application/json"}

    for attempt in range(3):
        try:
            r = requests.post(QWEN_API_URL, headers=headers,
                              json=payload, timeout=60)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return (text or "No explicit rhetorical pattern.")[:300]
        except Exception:
            if attempt < 2:
                time.sleep(3)
    return fallback


# ── CLIP encoding ────────────────────────────────────────────────────────

def encode_dataset(data, image_dir, model, processor,
                   device, batch_size, max_chars):
    embs = []
    model.eval()
    for start in tqdm(range(0, len(data), batch_size), desc="CLIP encoding"):
        batch = data[start:start + batch_size]
        images = [_open_image(image_dir, get_file_name(it)) for it in batch]
        texts  = [(safe_text(it.get("text"))[:max_chars]
                   or "empty meme text") for it in batch]

        inp = processor(text=texts, images=images, return_tensors="pt",
                        padding=True, truncation=True).to(device)
        with torch.no_grad():
            iv = F.normalize(model.get_image_features(
                pixel_values=inp["pixel_values"]), dim=-1)
            tv = F.normalize(model.get_text_features(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"]), dim=-1)
            embs.append(F.normalize(iv + tv, dim=-1).cpu().numpy())
        for im in images:
            im.close()
    return np.concatenate(embs)


# ── K-Means clustering + prototype extraction ────────────────────────────

def _kmeans(g_emb, n_clusters, seed, n_init):
    """k-means++ with `n_init` restarts; returns (assignments, normalized centers)."""
    if n_clusters == 1:
        assigns = np.zeros(len(g_emb), dtype=np.int64)
        centers = g_emb.mean(0, keepdims=True)
    else:
        km = KMeans(n_clusters=n_clusters, init="k-means++",
                    random_state=seed, n_init=n_init)
        assigns = km.fit_predict(g_emb)
        centers = km.cluster_centers_
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    return assigns, centers


def _pick_representatives(g_emb, centers, assigns, n_clusters, top_m):
    """For each cluster, return the global indices of the top_m centroid-nearest
    members, plus the single nearest member (medoid)."""
    reps, medoids = [], []
    for cid in range(n_clusters):
        members = np.where(assigns == cid)[0]
        if len(members) == 0:
            reps.append([])
            medoids.append(None)
            continue
        sims = g_emb[members] @ centers[cid]
        order = members[np.argsort(-sims)]
        reps.append(order[:top_m].tolist())
        medoids.append(order[0])
    return reps, medoids


def cluster_group(data, embeddings, idxs, group, n_clusters,
                  top_m, seed, n_init, image_dir, summarize=True):
    """Cluster one label group and emit label-free prototypes + medoids."""
    n_clusters = min(n_clusters, len(idxs))
    g_emb = embeddings[idxs]
    assigns, centers = _kmeans(g_emb, n_clusters, seed, n_init)
    reps_local, medoids_local = _pick_representatives(
        g_emb, centers, assigns, n_clusters, top_m)

    protos, medoid_records = [], []
    for cid in range(n_clusters):
        top_local = reps_local[cid]
        top_global = [idxs[int(x)] for x in top_local]
        medoid_global = idxs[int(medoids_local[cid])] if medoids_local[cid] is not None else None

        protos.append({
            "prototype_id": f"p{len(protos):03d}",
            "group": group,                       # offline-analysis only
            "centroid": centers[cid].astype(float).tolist(),
            "prototype_text": (_summarize_cluster(
                [data[i] for i in top_global], image_dir)
                if summarize else safe_text(data[top_global[0]].get("text"))),
            "representative_indices": top_global,
            "representative_samples": [
                {"file_name": get_file_name(data[i]),
                 "text": safe_text(data[i].get("text")),
                 "label": get_label(data[i])}
                for i in top_global
            ],
        })
        if medoid_global is not None:
            medoid_records.append({
                "prototype_id": protos[-1]["prototype_id"],
                "file_name": get_file_name(data[medoid_global]),
                "text": safe_text(data[medoid_global].get("text")),
                "embedding": embeddings[medoid_global].astype(float).tolist(),
            })
    return protos, medoid_records


# ── Benign K sweep (silhouette) ──────────────────────────────────────────

def pick_k_benign(g_emb, candidates, seed, n_init):
    best_k, best_s = candidates[0], -2.0
    for k in candidates:
        if k >= len(g_emb):
            continue
        assigns, _ = _kmeans(g_emb, k, seed, n_init)
        s = silhouette_score(g_emb, assigns, metric="cosine")
        if s > best_s:
            best_k, best_s = k, s
    return best_k


# ── Main pipeline ────────────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = load_json(args.train_data)
    if not data:
        raise ValueError(f"Empty dataset: {args.train_data}")

    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_proc  = CLIPProcessor.from_pretrained(args.clip_model)
    embs = encode_dataset(data, args.image_dir, clip_model, clip_proc,
                          device, args.batch_size, args.max_text_chars)

    labels = np.array([get_label(it) for it in data])
    harmful_idx = np.where(labels == 1)[0]
    benign_idx  = np.where(labels == 0)[0]

    split = args.label_split
    if split == "joint":
        n_clusters = args.k_total or (args.k_harmful + args.k_benign)
        protos, medoids = cluster_group(
            data, embs, np.arange(len(data)), "joint", n_clusters,
            args.top_m, args.seed, args.restarts, args.image_dir)
        k_h, k_b = 0, 0
    else:
        if split == "balanced":
            k_h = k_b = 15
        else:  # "separate" (default)
            k_h = args.k_harmful
            k_b = args.k_benign if not args.sweep_benign else pick_k_benign(
                embs[benign_idx], args.benign_sweep, args.seed, args.restarts)

        protos_h, medoids_h = cluster_group(
            data, embs, harmful_idx, "harmful", k_h,
            args.top_m, args.seed, args.restarts, args.image_dir)
        protos_b, medoids_b = cluster_group(
            data, embs, benign_idx, "benign", k_b,
            args.top_m, args.seed, args.restarts, args.image_dir)
        protos = protos_h + protos_b
        medoids = medoids_h + medoids_b

    # Instance index for `instance` retrieval ablation.
    instance_index = [
        {"file_name": get_file_name(it), "text": safe_text(it.get("text")),
         "label": int(labels[i]), "embedding": embs[i].astype(float).tolist()}
        for i, it in enumerate(data)
    ]

    output = {
        "dataset": args.dataset,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "clip_model": args.clip_model,
        "label_split": split,
        "k_harmful": k_h,
        "k_benign": k_b,
        "num_prototypes": len(protos),
        "prototypes": protos,
        "medoids": medoids,
        "instance_index": instance_index,
    }
    save_json(args.output, output)
    print(f"Saved {len(protos)} prototypes -> {args.output}")
