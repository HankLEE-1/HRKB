"""Neural-network sub-modules for the gated multimodal classifier.

  CrossModalCoAttention : bidirectional text <-> image attention
  ConfidenceGate        : similarity-guided fusion (g = 1 - s)
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _masked_mean(seq: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        return torch.sum(seq * m, dim=1) / m.sum(dim=1).clamp(min=1e-9)
    return seq.mean(dim=1)


class CrossModalCoAttention(nn.Module):
    """Bidirectional cross-attention between the text and vision streams.

    .. math::
        H_{vt} &= \\mathrm{MHA}(Q{=}T,\\, K{=}I,\\, V{=}I) \\\\
        H_{tv} &= \\mathrm{MHA}(Q{=}I,\\, K{=}T,\\, V{=}T)
    """

    def __init__(self, txt_dim: int, img_dim: int, hidden_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.t2i = nn.MultiheadAttention(hidden_dim, num_heads,
                                         dropout=dropout, batch_first=True)
        self.i2t = nn.MultiheadAttention(hidden_dim, num_heads,
                                         dropout=dropout, batch_first=True)
        self.txt_proj = (nn.Linear(txt_dim, hidden_dim)
                         if txt_dim != hidden_dim else nn.Identity())
        self.img_proj = (nn.Linear(img_dim, hidden_dim)
                         if img_dim != hidden_dim else nn.Identity())
        self.ln_t = nn.LayerNorm(hidden_dim)
        self.ln_v = nn.LayerNorm(hidden_dim)

    def forward(self, txt_emb, img_emb, txt_mask=None, img_mask=None):
        t_h = self.txt_proj(txt_emb)
        i_h = self.img_proj(img_emb)

        kp_i = (img_mask == 0) if img_mask is not None else None
        kp_t = (txt_mask == 0) if txt_mask is not None else None

        h_vt, _ = self.t2i(query=t_h, key=i_h, value=i_h,
                           key_padding_mask=kp_i)
        h_vt = self.ln_t(h_vt + t_h)

        h_tv, _ = self.i2t(query=i_h, key=t_h, value=t_h,
                           key_padding_mask=kp_t)
        h_tv = self.ln_v(h_tv + i_h)
        return h_vt, h_tv


class ConfidenceGate(nn.Module):
    """Fuse text-only and cross-modal representations via a confidence gate.

    .. math::
        s &= \\cos(t_{cls},\\, \\mathrm{proj}(v_{cls})),\\quad g = 1 - s \\\\
        h_{fused} &= g \\cdot h_{cross} + (1 - g) \\cdot h_T

    When text and vision disagree (low s), g rises and the model leans on the
    cross-modal interaction; when they agree, it trusts the text representation.
    """

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.cross_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, h_vt, h_tv, t_cls, v_cls, txt_mask=None, img_mask=None,
                use_gate=True):
        cos = torch.sum(
            F.normalize(t_cls, dim=1) * F.normalize(v_cls, dim=1),
            dim=1, keepdim=True,
        ).clamp(-1.0, 1.0)
        s = cos                                  # gate similarity
        g = 1.0 - s

        h_cross = self.cross_proj(torch.cat(
            [_masked_mean(h_vt, txt_mask), _masked_mean(h_tv, img_mask)], dim=1))
        h_T = t_cls

        if use_gate:
            h_fused = g * h_cross + (1.0 - g) * h_T
        else:
            h_fused = h_cross                    # w/o gating: always cross-modal
        return self.classifier(h_fused), s.squeeze(1)
