"""Label-agnostic retrieval over the HRKB (or its ablation counterparts).

Four strategies, selected by ``mode``, all sharing the same query embedding
phi(query) = Norm(CLIP_img + CLIP_txt) and cosine similarity:

  * ``prototype`` — top-k rhetorical prototypes from the whole label-free base
  * ``instance``  — top-k raw training memes (instance-level retrieval)
  * ``medoid``    — top-k cluster medoids (centroid-nearest meme per cluster)
  * ``none``      — no retrieved knowledge

Every strategy returns a knowledge prefix built from *text only*: no class
names, no labels, no label-conditioned routing. Harmful and benign candidates
compete on equal footing, so the prefix cannot act as a unidirectional prior.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor

from .utils import load_json

ImageFile.LOAD_TRUNCATED_IMAGES = True

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RetrievalIndex:
    def __init__(self, hrkb_path: str, mode: str = "prototype",
                 top_k: int = 5, clip_model: str = "openai/clip-vit-base-patch32",
                 device: str = None):
        self.mode = mode
        self.top_k = top_k
        self.device = device or _DEVICE

        self.hrkb = None
        if hrkb_path and os.path.exists(hrkb_path):
            self.hrkb = load_json(hrkb_path)

        self._clip_model = None
        self._clip_proc = None
        self._clip_name = clip_model
        self._cand_texts = []
        self._cand_embs = np.zeros((0, 512), dtype=np.float32)

        self._prepare_candidates()

    # ── CLIP singleton ──────────────────────────────────────────────────
    def _clip(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(
                self._clip_name).to(self.device)
            self._clip_proc = CLIPProcessor.from_pretrained(self._clip_name)
            self._clip_model.eval()
        return self._clip_model, self._clip_proc

    # ── candidate table ─────────────────────────────────────────────────
    def _prepare_candidates(self):
        if not self.hrkb:
            return
        protos = self.hrkb.get("prototypes", [])
        if self.mode == "prototype":
            self._cand_texts = [p.get("prototype_text", "") for p in protos]
            vecs = [np.asarray(p["centroid"], dtype=np.float32)
                    for p in protos if p.get("centroid")]
        elif self.mode == "medoid":
            medoids = self.hrkb.get("medoids", [])
            self._cand_texts = [m.get("text", "") for m in medoids]
            vecs = [np.asarray(m["embedding"], dtype=np.float32)
                    for m in medoids if m.get("embedding")]
        elif self.mode == "instance":
            inst = self.hrkb.get("instance_index", [])
            self._cand_texts = [i.get("text", "") for i in inst]
            vecs = [np.asarray(i["embedding"], dtype=np.float32)
                    for i in inst if i.get("embedding")]
        else:  # "none"
            self._cand_texts = []
            vecs = []

        if vecs:
            self._cand_embs = np.stack(vecs)
            self._cand_embs /= np.clip(
                np.linalg.norm(self._cand_embs, axis=1, keepdims=True),
                1e-12, None)

    # ── query embedding ─────────────────────────────────────────────────
    def build_query(self, image_path: str, text: str) -> np.ndarray:
        model, proc = self._clip()
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), "black")
        inp = proc(text=[text or "empty meme text"], images=[img],
                   return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            iv = F.normalize(model.get_image_features(
                pixel_values=inp["pixel_values"]), dim=-1)
            tv = F.normalize(model.get_text_features(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"]), dim=-1)
            q = F.normalize(iv + tv, dim=-1)
        img.close()
        return q.squeeze(0).cpu().numpy().astype(np.float32)

    # ── retrieval ───────────────────────────────────────────────────────
    def retrieve(self, image_path: str, text: str) -> str:
        if self.mode == "none" or len(self._cand_embs) == 0:
            return ""
        q = self.build_query(image_path, text)
        scores = self._cand_embs @ q
        top = int(min(self.top_k, len(self._cand_texts)))
        lines = []
        for idx in np.argsort(-scores)[:top]:
            t = self._cand_texts[int(idx)]
            if t:
                lines.append(f"{len(lines) + 1}. {t}")
        return "\n".join(lines)
