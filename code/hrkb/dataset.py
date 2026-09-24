"""Data loading, collation, and augmentation for HRKB detection training."""

import os

import torch
from PIL import Image, ImageFile
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from .utils import load_json

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_AUG = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333),
                        interpolation=InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomGrayscale(0.05),
])


def get_cot(item: dict, variant: str) -> dict:
    """Return the four CoT segments for a given variant, or empty defaults."""
    cot = item.get("cot_variants", {}).get(variant, {})
    return {
        "visual": cot.get("visual", ""),
        "context": cot.get("context", ""),
        "mechanism": cot.get("mechanism", ""),
        "summary": cot.get("summary", ""),
    }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        max_len = max(len(b[0]) for b in batch)
        input_ids, att_masks, ttids = [], [], []
        pixel_values, labels = [], []

        for ids, att, pv, lbl, tti in batch:
            pad = max_len - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad)
            att_masks.append(att + [0] * pad)
            ttids.append(tti + [0] * pad)
            pixel_values.append(pv)
            labels.append(lbl)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(att_masks),
            "pixel_values": torch.stack(pixel_values).squeeze(1),
            "labels": torch.tensor(labels),
            "token_type_ids": torch.tensor(ttids),
        }


class MemeDataset(torch.utils.data.Dataset):
    """Loads CoT-enriched JSON and builds ``[CLS] OCR [SEP] CoT [SEP]``
    token sequences together with CLIP pixel values."""

    def __init__(self, path, processor, tokenizer, image_dir,
                 max_len=512, usage="train", use_aug=True, variant="prototype_k5"):
        self.data = load_json(path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.max_len = max_len
        self.usage = usage
        self.use_aug = use_aug
        self.variant = variant

    def __len__(self):
        return len(self.data)

    def _tok(self, text: str, max_tok: int = 100):
        toks = self.tokenizer.tokenize(text)[:max_tok]
        return self.tokenizer.convert_tokens_to_ids(toks)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = int(item.get("label", item.get("metaphor occurrence", 0)))

        try:
            img = Image.open(
                os.path.join(self.image_dir, item["file_name"])).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color="black")
        if self.usage == "train" and self.use_aug:
            try:
                img = TRAIN_AUG(img)
            except Exception:
                pass
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"]
        img.close()

        cot = get_cot(item, self.variant)
        ocr_ids = self._tok(item.get("text", "") or "", 100)
        vis_ids = self._tok(cot["visual"] or "None", 50)
        ctx_ids = self._tok(cot["context"] or "None", 90)
        mech_ids = self._tok(cot["mechanism"] or "None", 100)
        sum_ids = self._tok(cot["summary"] or "None", 60)

        sep = self.tokenizer.sep_token_id
        ids = [self.tokenizer.cls_token_id]
        ids += ocr_ids + [sep]
        ids += vis_ids + [sep] + ctx_ids + [sep] + mech_ids + [sep] + sum_ids + [sep]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            ids[-1] = sep

        att = [1] * len(ids)
        tti = [0] * len(ids)
        return ids, att, pv, label, tti
