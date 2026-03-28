from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class ScaleInvariantAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        scaling_factor_init: float = 1.0,
        enable_scale_adaptation: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling_factor_init = scaling_factor_init
        self.enable_scale_adaptation = enable_scale_adaptation
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if enable_scale_adaptation:
            self.xi_scale_encoder = nn.Sequential(
                nn.Linear(1, self.head_dim // 2),
                nn.LayerNorm(self.head_dim // 2),
                nn.GELU(),
                nn.Linear(self.head_dim // 2, self.head_dim),
            )
            self.scale_gating = nn.Parameter(torch.ones(num_heads))
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()
    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        xi_scale: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            key = query
        if value is None:
            value = query
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        if self.enable_scale_adaptation and xi_scale is not None:
            scale_adaptation = self._compute_scale_adaptation(xi_scale, batch_size)
            q = q * scale_adaptation
            k = k * scale_adaptation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, tgt_len, self.embed_dim)
        output = self.out_proj(attn_output)
        if need_weights:
            return output, attn_weights
        return output, None
    def _compute_scale_adaptation(
        self,
        xi_scale: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if xi_scale.dim() == 0:
            xi_scale = xi_scale.unsqueeze(0)
        if xi_scale.dim() == 1:
            xi_scale = xi_scale.unsqueeze(-1)
        xi_encoded = self.xi_scale_encoder(xi_scale)
        gating = torch.sigmoid(self.scale_gating)
        adaptation = 1.0 + gating.view(1, self.num_heads, 1) * xi_encoded.unsqueeze(1)
        return adaptation.unsqueeze(2)
class RGMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_scale_levels: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scale_levels = num_scale_levels
        self.head_dim = embed_dim // num_heads
        assert num_heads % num_scale_levels == 0
        self.heads_per_scale = num_heads // num_scale_levels
        self.scale_heads = nn.ModuleList([
            ScaleInvariantAttention(
                embed_dim=self.head_dim * self.heads_per_scale,
                num_heads=self.heads_per_scale,
                dropout=dropout,
                bias=bias,
                scaling_factor_init=1.0 / (2 ** scale_idx),
            )
            for scale_idx in range(num_scale_levels)
        ])
        self.scale_projections = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim * self.heads_per_scale)
            for _ in range(num_scale_levels)
        ])
        self.scale_aggregations = nn.ModuleList([
            nn.Linear(self.head_dim * self.heads_per_scale, embed_dim)
            for _ in range(num_scale_levels)
        ])
        self.out_projection = nn.Linear(embed_dim * num_scale_levels, embed_dim)
        self.xi_router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_scale_levels),
        )
        self._init_parameters()
    def _init_parameters(self) -> None:
        for proj in self.scale_projections:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        for agg in self.scale_aggregations:
            nn.init.xavier_uniform_(agg.weight)
            if agg.bias is not None:
                nn.init.zeros_(agg.bias)
        nn.init.xavier_uniform_(self.out_projection.weight)
        if self.out_projection.bias is not None:
            nn.init.zeros_(self.out_projection.bias)
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if xi_scale is None:
            xi_scale = self._estimate_xi_from_features(x)
        routing_weights = F.softmax(self.xi_router(x.mean(dim=1)), dim=-1)
        scale_outputs = []
        for scale_idx, (scale_head, scale_proj, scale_agg) in enumerate(
            zip(self.scale_heads, self.scale_projections, self.scale_aggregations)
        ):
            scale_input = scale_proj(x)
            scale_xi = xi_scale * (2.0 ** scale_idx)
            scale_out, _ = scale_head(
                query=scale_input,
                xi_scale=scale_xi,
                attn_mask=attn_mask,
            )
            scale_out = scale_agg(scale_out)
            scale_outputs.append(scale_out * routing_weights[:, scale_idx:scale_idx+1, None])
        concatenated = torch.cat(scale_outputs, dim=-1)
        output = self.out_projection(concatenated)
        return output
    def _estimate_xi_from_features(self, x: torch.Tensor) -> torch.Tensor:
        feature_variance = x.var(dim=-1, keepdim=True)
        xi_estimate = torch.log1p(feature_variance.mean(dim=1, keepdim=True))
        return xi_estimate.squeeze(-1)
class RGAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_scale_levels: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = RGMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_scale_levels=num_scale_levels,
            dropout=attn_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.xi_tracker = nn.Parameter(torch.zeros(1))
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if xi_scale is None:
            xi_scale = torch.sigmoid(self.xi_tracker)
        attn_out = self.attn(self.norm1(x), xi_scale=xi_scale, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, xi_scale
class RGAttentionFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_scale_levels: int = 4,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_size: int = 16,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scale_levels = num_scale_levels
        self.depth = depth
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            RGAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_scale_levels=num_scale_levels,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.xi_depth_controller = nn.Parameter(torch.linspace(1.0, 0.1, depth))
        self._init_parameters()
    def _init_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
    def forward(
        self,
        x: torch.Tensor,
        xi_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if xi_input is None:
            xi_input = self._compute_input_complexity(x)
        xi_values = []
        for block_idx, block in enumerate(self.blocks):
            depth_scale = torch.sigmoid(self.xi_depth_controller[block_idx])
            layer_xi = xi_input * depth_scale
            x, xi_out = block(x, xi_scale=layer_xi)
            xi_values.append(xi_out.detach())
        x = self.norm(x)
        return x, xi_values
    def _compute_input_complexity(self, x: torch.Tensor) -> torch.Tensor:
        spatial_variance = x[:, 1:, :].var(dim=1)
        complexity = torch.log1p(spatial_variance.mean(dim=-1, keepdim=True))
        return complexity.squeeze(-1)
    def get_attention_maps(self) -> list:
        return []