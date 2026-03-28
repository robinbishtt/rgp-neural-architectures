from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class RotationEquivariantConv(nn.Module):
    def __init(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_rotations: int = 4,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations
        self.stride = stride
        self.padding = padding
        self.base_kernel = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )
        self.rotation_angles = torch.linspace(0, 2 * math.pi, num_rotations + 1)[:-1]
        self.rotation_gates = nn.Parameter(torch.ones(num_rotations) / num_rotations)
    def _rotate_kernel(
        self,
        kernel: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
        ], device=kernel.device).unsqueeze(0)
        grid = F.affine_grid(
            theta,
            kernel.unsqueeze(0).shape,
            align_corners=False,
        )
        rotated = F.grid_sample(
            kernel.unsqueeze(0),
            grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros',
        )
        return rotated.squeeze(0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotation_weights = F.softmax(self.rotation_gates, dim=0)
        outputs = []
        for idx, angle in enumerate(self.rotation_angles):
            rotated_kernel = self._rotate_kernel(self.base_kernel, angle.to(x.device))
            out = F.conv2d(
                x,
                rotated_kernel,
                stride=self.stride,
                padding=self.padding,
            )
            outputs.append(out * rotation_weights[idx])
        return sum(outputs)
class TranslationEquivariantPool(nn.Module):
    def __init(
        self,
        kernel_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
class ScaleEquivariantProjection(nn.Module):
    def __init(
        self,
        in_features: int,
        out_features: int,
        num_scales: int = 4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_scales = num_scales
        self.scale_projections = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for _ in range(num_scales)
        ])
        self.scale_gates = nn.Parameter(torch.ones(num_scales) / num_scales)
        self.xi_encoder = nn.Sequential(
            nn.Linear(1, num_scales),
            nn.Softmax(dim=-1),
        )
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if xi_scale is not None:
            scale_weights = self.xi_encoder(xi_scale.view(-1, 1) if xi_scale.dim() == 1 else xi_scale)
        else:
            scale_weights = F.softmax(self.scale_gates, dim=0).unsqueeze(0)
        outputs = []
        for idx, proj in enumerate(self.scale_projections):
            proj_out = proj(x)
            weight = scale_weights[:, idx].view(-1, 1)
            outputs.append(proj_out * weight)
        return sum(outputs)
class SymmetryEquivariantRGBlock(nn.Module):
    def __init(
        self,
        in_channels: int,
        out_channels: int,
        xi_depth: float = 7.2,
        num_rotations: int = 4,
        enable_rotation: bool = True,
        enable_scale: bool = True,
        enable_translation: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.xi_depth = xi_depth
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.enable_translation = enable_translation
        if enable_rotation:
            self.rot_conv = RotationEquivariantConv(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                num_rotations=num_rotations,
            )
        self.standard_conv = nn.Conv2d(
            in_channels,
            out_channels // 2 if enable_rotation else out_channels,
            kernel_size=3,
            padding=1,
        )
        if enable_scale:
            self.scale_proj = ScaleEquivariantProjection(
                in_features=out_channels,
                out_features=out_channels,
            )
        if enable_translation:
            self.trans_pool = TranslationEquivariantPool()
        self.bn = nn.BatchNorm2d(out_channels)
        self.symmetry_gates = nn.Parameter(torch.zeros(3))
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gates = torch.sigmoid(self.symmetry_gates)
        outputs = []
        if self.enable_rotation:
            rot_out = self.rot_conv(x)
            outputs.append(rot_out * gates[0])
        std_out = self.standard_conv(x)
        outputs.append(std_out * (1 - gates[0]))
        combined = torch.cat(outputs, dim=1)
        combined = self.bn(combined)
        combined = F.relu(combined, inplace=True)
        if self.enable_scale and xi_scale is not None:
            batch_size, channels, height, width = combined.shape
            flat = combined.view(batch_size, channels, -1).transpose(1, 2)
            scaled = self.scale_proj(flat, xi_scale)
            combined = scaled.transpose(1, 2).view(batch_size, channels, height, width)
        if self.enable_translation:
            combined = self.trans_pool(combined)
        info = {
            : gates[0].item(),
            : gates[1].item() if self.enable_scale else 0.0,
            : gates[2].item() if self.enable_translation else 0.0,
        }
        return combined, info
class GroupEquivariantAttention(nn.Module):
    def __init(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_group_elements: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_group_elements = num_group_elements
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.group_embeddings = nn.Parameter(
            torch.randn(num_group_elements, embed_dim) * 0.02
        )
        self.equivariance_gates = nn.Parameter(torch.zeros(num_group_elements))
    def apply_group_action(
        self,
        x: torch.Tensor,
        group_idx: int,
    ) -> torch.Tensor:
        embedding = self.group_embeddings[group_idx]
        return x + embedding.unsqueeze(0).unsqueeze(0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        group_weights = F.softmax(self.equivariance_gates, dim=0)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        group_qs = []
        for g_idx in range(self.num_group_elements):
            g_x = self.apply_group_action(x, g_idx)
            g_q = self.q_proj(g_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            group_qs.append(g_q * group_weights[g_idx])
        q_equivariant = sum(group_qs)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q_equivariant, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output
class SymmetryEquivarianceEngine(nn.Module):
    def __init(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        xi_depth: float = 7.2,
        num_rotations: int = 4,
        img_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.equivariant_blocks = nn.ModuleList([
            GroupEquivariantAttention(
                embed_dim=embed_dim,
                num_heads=12,
                num_group_elements=num_rotations,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.xi_invariant = nn.Parameter(torch.ones(1) * xi_depth)
        self._init_parameters()
    def _init_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)
        if xi_scale is None:
            xi_scale = torch.sigmoid(self.xi_invariant)
        for block in self.equivariant_blocks:
            x = x + block(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits
    def test_equivariance(
        self,
        x: torch.Tensor,
        rotation_angle: float = 90.0,
    ) -> Dict[str, float]:
        with torch.no_grad():
            output_original = self.forward(x)
            angle_rad = rotation_angle * math.pi / 180.0
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
            ], device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x_rotated = F.grid_sample(x, grid, align_corners=False)
            output_rotated = self.forward(x_rotated)
            difference = (output_original - output_rotated).norm()
            relative_error = difference / (output_original.norm() + 1e-8)
        return {
            : relative_error.item(),
            : difference.item(),
            : rotation_angle,
        }