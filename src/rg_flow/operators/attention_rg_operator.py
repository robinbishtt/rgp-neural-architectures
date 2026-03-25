from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
class AttentionRGOperator(nn.Module):
    def __init__(
        self,
        d_model:     int,
        n_heads:     int   = 4,
        dropout:     float = 0.0,
        bias:        bool  = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.W_q.weight, std=1.0 / self.d_head ** 0.5)
        nn.init.normal_(self.W_k.weight, std=1.0 / self.d_head ** 0.5)
        nn.init.normal_(self.W_v.weight, std=1.0 / d_model ** 0.5)
        nn.init.zeros_(self.W_o.weight)
        if bias:
            nn.init.zeros_(self.W_q.bias)
            nn.init.zeros_(self.W_k.bias)
            nn.init.zeros_(self.W_v.bias)
            nn.init.zeros_(self.W_o.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, d = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale  = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn   = self.dropout(F.softmax(scores, dim=-1))
        out    = torch.matmul(attn, V)
        out    = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out    = self.W_o(out)
        out = x + out
        if len(batch) == 1:
            out = out.squeeze(1)  
        return out