from __future__ import annotations
import torch
import torch.nn as nn
class WaveletRGOperator(nn.Module):
    def __init__(
        self,
        in_dim:      int,
        out_dim:     int,
        n_scales:    int   = 2,
        use_detail:  bool  = True,
    ) -> None:
        super().__init__()
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.n_scales   = n_scales
        self.use_detail = use_detail
        self.lowpass = nn.Linear(in_dim, out_dim)
        if use_detail:
            self.highpass = nn.ModuleList([
                nn.Linear(in_dim, out_dim) for _ in range(n_scales)
            ])
            self.detail_mix = nn.Parameter(
                torch.zeros(n_scales) * 0.1
            )
        nn.init.normal_(self.lowpass.weight, std=1.0 / in_dim ** 0.5)
        nn.init.zeros_(self.lowpass.bias)
        self.activation = nn.Tanh()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.lowpass(x))
        if self.use_detail:
            mixing = torch.sigmoid(self.detail_mix)
            for i, hp in enumerate(self.highpass):
                detail = self.activation(hp(x))
                h = h + mixing[i] * detail
        return h