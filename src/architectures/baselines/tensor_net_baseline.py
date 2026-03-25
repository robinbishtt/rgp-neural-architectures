from __future__ import annotations
import math
import torch
import torch.nn as nn
class TensorTrainLayer(nn.Module):
    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bond_dim:     int = 32,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.bond_dim     = bond_dim
        rank = min(bond_dim, min(in_features, out_features))
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._init_parameters()
    def _init_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.normal_(self.U, std=std)
        nn.init.normal_(self.V, std=std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.U @ self.V).T + self.bias
    @property
    def n_parameters(self) -> int:
        rank = self.U.shape[1]
        return self.out_features * rank + rank * self.in_features
class TensorNetBaseline(nn.Module):
    def __init__(
        self,
        input_dim:   int   = 784,
        hidden_dim:  int   = 512,
        output_dim:  int   = 10,
        depth:       int   = 4,
        bond_dim:    int   = 32,
    ) -> None:
        super().__init__()
        self.embed  = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TensorTrainLayer(hidden_dim, hidden_dim, bond_dim=bond_dim)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        self.head   = nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.embed(x))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(torch.tanh(layer(h)) + h)  
        return self.head(h)