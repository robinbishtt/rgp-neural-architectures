from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class RGExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: str = 'gelu',
        dropout: float = 0.0,
        expert_type: str = 'texture',
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.expert_type = expert_type
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()
    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class RGRouter(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_experts: int,
        xi_scale_dim: int = 1,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.scale_encoder = nn.Sequential(
            nn.Linear(xi_scale_dim, in_features // 4),
            nn.LayerNorm(in_features // 4),
            nn.GELU(),
            nn.Linear(in_features // 4, in_features // 2),
        )
        self.feature_processor = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.GELU(),
        )
        self.router_head = nn.Linear(in_features, num_experts)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self._init_parameters()
    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.router_head.weight, gain=0.01)
        nn.init.zeros_(self.router_head.bias)
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = x.shape[0]
        feature_repr = self.feature_processor(x.mean(dim=1) if x.dim() > 2 else x)
        if xi_scale is not None:
            scale_repr = self.scale_encoder(xi_scale.view(-1, 1) if xi_scale.dim() == 1 else xi_scale)
            combined_repr = torch.cat([feature_repr, scale_repr], dim=-1)
        else:
            combined_repr = feature_repr
        router_logits = self.router_head(combined_repr)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        temperature = F.softplus(self.temperature) + 0.1
        routing_weights = F.softmax(router_logits / temperature, dim=-1)
        aux_info = {
            : router_logits,
            : temperature,
        }
        return routing_weights, aux_info
class RGMoELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = 'gelu',
        dropout: float = 0.0,
        load_balance_coef: float = 0.01,
        importance_loss_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.load_balance_coef = load_balance_coef
        self.importance_loss_coef = importance_loss_coef
        expert_types = ['texture', 'shape', 'global', 'local', 'edge', 'color', 'frequency', 'semantic']
        self.experts = nn.ModuleList([
            RGExpert(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                activation=activation,
                dropout=dropout,
                expert_type=expert_types[i % len(expert_types)],
            )
            for i in range(num_experts)
        ])
        self.router = RGRouter(
            in_features=in_features,
            num_experts=num_experts,
        )
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('load_balance_history', torch.zeros(1000))
        self.history_ptr = 0
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)
        routing_weights, router_info = self.router(x, xi_scale)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-10)
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)
                expert_weight = topk_weights[mask, (topk_indices[mask] == expert_idx).nonzero(as_tuple=True)[1]]
                output[mask] = output[mask] + expert_weight.unsqueeze(-1) * expert_output
        output = output.view(batch_size, seq_len, self.out_features)
        aux_loss = self._compute_auxiliary_loss(routing_weights, x)
        load_balance = self._compute_load_balance(routing_weights)
        self.expert_usage += routing_weights.sum(dim=0).detach()
        aux_info = {
            : routing_weights,
            : topk_indices,
            : aux_loss,
            : load_balance,
            : self.expert_usage.clone(),
        }
        aux_info.update(router_info)
        return output, aux_info
    def _compute_auxiliary_loss(
        self,
        routing_weights: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        router_prob = routing_weights.mean(dim=0)
        uniform_prob = torch.ones_like(router_prob) / self.num_experts
        load_balance_loss = self.load_balance_coef * self.num_experts * (
            router_prob * router_prob
        ).sum()
        importance = routing_weights.sum(dim=0)
        cv_squared = (importance.var() / (importance.mean() ** 2 + 1e-10))
        importance_loss = self.importance_loss_coef * cv_squared
        total_aux_loss = load_balance_loss + importance_loss
        return total_aux_loss
    def _compute_load_balance(self, routing_weights: torch.Tensor) -> torch.Tensor:
        expert_fraction = routing_weights.mean(dim=0)
        target_fraction = 1.0 / self.num_experts
        balance = 1.0 - (expert_fraction - target_fraction).abs().sum() / 2.0
        if self.history_ptr < len(self.load_balance_history):
            self.load_balance_history[self.history_ptr] = balance
            self.history_ptr += 1
        return balance
    def get_expert_statistics(self) -> Dict[str, torch.Tensor]:
        usage_normalized = self.expert_usage / (self.expert_usage.sum() + 1e-10)
        return {
            : self.expert_usage.clone(),
            : usage_normalized,
            : -(usage_normalized * (usage_normalized + 1e-10).log()).sum(),
            : usage_normalized.max() / (usage_normalized.mean() + 1e-10),
        }
class RGMoEBlock(nn.Module):
    def __init(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        load_balance_coef: float = 0.01,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.moe = RGMoELayer(
            in_features=embed_dim,
            hidden_features=hidden_dim,
            out_features=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            load_balance_coef=load_balance_coef,
        )
        self.xi_adapter = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if xi_scale is not None:
            xi_scale = self.xi_adapter(xi_scale.view(-1, 1) if xi_scale.dim() == 1 else xi_scale)
        moe_out, aux_info = self.moe(self.norm1(x), xi_scale)
        x = x + moe_out
        return x, aux_info
class RGMoETransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_size: int = 16,
        img_size: int = 224,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_experts = num_experts
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            RGMoEBlock(
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
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
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)
        aux_info_list = []
        for block in self.blocks:
            x, aux_info = block(x, xi_scale)
            aux_info_list.append(aux_info)
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits, aux_info_list
    def get_load_balance_statistics(self) -> Dict[str, float]:
        all_balances = []
        for block in self.blocks:
            if hasattr(block.moe, 'load_balance_history'):
                valid_history = block.moe.load_balance_history[block.moe.load_balance_history > 0]
                if len(valid_history) > 0:
                    all_balances.append(valid_history.mean().item())
        if not all_balances:
            return {'mean_load_balance': 0.0, 'min_load_balance': 0.0, 'max_load_balance': 0.0}
        return {
            : sum(all_balances) / len(all_balances),
            : min(all_balances),
            : max(all_balances),
        }