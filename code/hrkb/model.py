"""Gated multimodal classifier: the reference detector instantiated with the HRKB.

Deliberately standard, so performance differences can be attributed to the
retrieved knowledge rather than to architectural novelty:

  * text  : XLM-RoBERTa-large over "[CLS] OCR [SEP] CoT [SEP]"
  * vision: frozen CLIP ViT-B/32
  * fusion: bidirectional cross-attention + confidence gate (g = 1 - s)
  * head  : two-layer MLP
"""

import torch
from torch import nn

from .modules import ConfidenceGate, CrossModalCoAttention


class GatedMultimodalClassifier(nn.Module):
    def __init__(self, vit, bert, num_labels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vit = vit
        self.bert = bert

        self.img_dim = vit.config.hidden_size      # 768  (CLIP ViT-B/32)
        self.txt_dim = bert.config.hidden_size      # 1024 (XLM-R large)
        self.hidden_dim = 1024

        self.cross_attn = CrossModalCoAttention(
            self.txt_dim, self.img_dim, self.hidden_dim, dropout=dropout)

        self.v_cls_proj = nn.Linear(self.img_dim, self.hidden_dim)
        self.t_cls_proj = nn.Linear(self.txt_dim, self.hidden_dim)

        self.gate = ConfidenceGate(self.hidden_dim, num_labels, dropout=dropout)

    def forward(self, input_ids, attention_mask, pixel_values,
                token_type_ids=None, use_gate=True):
        # Text stream
        txt_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_seq = txt_out.last_hidden_state            # [B, L, 1024]
        t_cls = txt_out.pooler_output                   # [B, 1024]

        # Vision stream (frozen)
        img_out = self.vit(pixel_values=pixel_values)
        img_seq = img_out.last_hidden_state            # [B, 50, 768]
        v_cls = img_out.pooler_output                   # [B, 768]

        h_vt, h_tv = self.cross_attn(
            txt_seq, img_seq,
            txt_mask=attention_mask,
            img_mask=torch.ones(img_seq.shape[:2], device=img_seq.device),
        )

        logits, s = self.gate(
            h_vt, h_tv,
            self.t_cls_proj(t_cls),
            self.v_cls_proj(v_cls),
            txt_mask=attention_mask,
            use_gate=use_gate,
        )
        return logits, s
