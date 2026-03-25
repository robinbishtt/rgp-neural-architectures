from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_out = out_channels // 4
        self.branch1x1 = nn.Sequential(
            nn.Linear(in_channels, branch_out), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Linear(in_channels, branch_out), nn.ReLU(),
            nn.Linear(branch_out, branch_out), nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Linear(in_channels, branch_out), nn.ReLU(),
            nn.Linear(branch_out, branch_out), nn.ReLU(),
            nn.Linear(branch_out, branch_out), nn.ReLU()
        )
        self.branch_pool = nn.Sequential(
            nn.Linear(in_channels, branch_out), nn.ReLU()
        )
        self.total_out = branch_out * 4
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1x1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], dim=-1)
class InceptionBaseline(nn.Module):
    def __init__(
        self,
        input_dim:  int,
        n_classes:  int,
        n_blocks:   int   = 3,
        d_model:    int   = 128,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks     = nn.ModuleList()
        self.norms      = nn.ModuleList()
        in_dim          = d_model
        for _ in range(n_blocks):
            block    = InceptionBlock(in_dim, d_model)
            out_dim  = block.total_out
            proj     = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()
            self.blocks.append(nn.ModuleDict({"inc": block, "proj": proj}))
            self.norms.append(nn.LayerNorm(d_model))
            in_dim = d_model
        self.classifier = nn.Linear(d_model, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        for block_dict, norm in zip(self.blocks, self.norms):
            h = block_dict["inc"](x)
            h = block_dict["proj"](h)
            x = norm(x + h)
        return self.classifier(x)