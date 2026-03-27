from __future__ import annotations
import torch
import torch.nn as nn
class LearnedRGOperator(nn.Module):
    def __init__(
        self,
        in_dim:       int,
        out_dim:      int,
        hyper_hidden: int  = 32,
    ) -> None:
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        n_stats = 3
        n_params = out_dim * in_dim + out_dim  
        self.hyper_net = nn.Sequential(
            nn.Linear(n_stats, hyper_hidden),
            nn.Tanh(),
            nn.Linear(hyper_hidden, n_params),
        )
        self.baseline = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()
        nn.init.normal_(self.baseline.weight, std=1.0 / in_dim ** 0.5)
        nn.init.zeros_(self.baseline.bias)
        nn.init.zeros_(self.hyper_net[-1].weight)
        nn.init.zeros_(self.hyper_net[-1].bias)
    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        mu     = x.mean(dim=-1, keepdim=True)
        sigma  = x.std(dim=-1, keepdim=True) + 1e-8
        skew   = ((x - mu) ** 3).mean(dim=-1, keepdim=True) / sigma ** 3
        return torch.cat([mu, sigma, skew], dim=-1)  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats  = self._compute_stats(x)           
        params = self.hyper_net(stats)             
        W_delta = params[:, :self.out_dim * self.in_dim].view(
            -1, self.out_dim, self.in_dim
        )  
        b_delta = params[:, self.out_dim * self.in_dim:]  
        h_base = self.baseline(x)  
        h_adapt = torch.bmm(W_delta, x.unsqueeze(-1)).squeeze(-1) + b_delta
        return self.activation(h_base + h_adapt)