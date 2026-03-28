from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class RenormalizedNorm(nn.Module):
    def __init(
        self,
        num_features: int,
        xi_scale_dim: int = 1,
        eps: float = 1e-6,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.xi_encoder = nn.Sequential(
            nn.Linear(xi_scale_dim, num_features // 4),
            nn.LayerNorm(num_features // 4),
            nn.GELU(),
            nn.Linear(num_features // 4, num_features),
            nn.Sigmoid(),
        )
        self.scale_gating = nn.Parameter(torch.zeros(num_features))
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        if self.training or not self.track_running_stats:
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if xi_scale is not None:
            scale_factor = self.xi_encoder(xi_scale.view(-1, 1) if xi_scale.dim() == 1 else xi_scale)
            gating = torch.sigmoid(self.scale_gating)
            x_normalized = x_normalized * (1 + gating * scale_factor)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized
class ScaleInvariantBatchNorm(nn.Module):
    def __init(
        self,
        num_features: int,
        num_scale_levels: int = 4,
        eps: float = 1e-6,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_scale_levels = num_scale_levels
        self.eps = eps
        self.momentum = momentum
        self.scale_norms = nn.ModuleList([
            RenormalizedNorm(
                num_features=num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
            )
            for _ in range(num_scale_levels)
        ])
        self.scale_router = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.LayerNorm(num_features // 2),
            nn.GELU(),
            nn.Linear(num_features // 2, num_scale_levels),
        )
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 4:
            batch_size, channels, height, width = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
        else:
            x_flat = x
        routing_weights = F.softmax(self.scale_router(x_flat.mean(dim=0, keepdim=True)), dim=-1)
        outputs = []
        for scale_idx, norm in enumerate(self.scale_norms):
            if xi_scale is not None:
                scale_xi = xi_scale * (2.0 ** scale_idx)
            else:
                scale_xi = None
            scale_output = norm(x_flat, xi_scale=scale_xi)
            outputs.append(scale_output * routing_weights[0, scale_idx])
        output = sum(outputs)
        if x.dim() == 4:
            output = output.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return output
class RGGroupNorm(nn.Module):
    def __init(
        self,
        num_channels: int,
        num_groups: int = 32,
        xi_scale_dim: int = 1,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        assert num_channels % num_groups == 0
        self.channels_per_group = num_channels // num_groups
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.xi_scale_encoder = nn.Sequential(
            nn.Linear(xi_scale_dim, num_groups),
            nn.Sigmoid(),
        )
        self.group_gating = nn.Parameter(torch.ones(num_groups))
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, self.num_groups, self.channels_per_group, height, width)
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if xi_scale is not None:
            scale_factors = self.xi_scale_encoder(xi_scale.view(1, -1) if xi_scale.dim() == 1 else xi_scale)
            gating = torch.sigmoid(self.group_gating)
            scale_factors = scale_factors.view(1, self.num_groups, 1, 1, 1)
            gating = gating.view(1, self.num_groups, 1, 1, 1)
            x_normalized = x_normalized * (1 + gating * scale_factors)
        x_normalized = x_normalized.view(batch_size, channels, height, width)
        if self.weight is not None and self.bias is not None:
            x_normalized = x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_normalized
class FisherWeightedNorm(nn.Module):
    def __init(
        self,
        num_features: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('fisher_weights', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.fisher_estimator = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.LayerNorm(num_features // 2),
            nn.GELU(),
            nn.Linear(num_features // 2, num_features),
            nn.Sigmoid(),
        )
    def forward(
        self,
        x: torch.Tensor,
        compute_fisher: bool = True,
    ) -> torch.Tensor:
        if compute_fisher:
            with torch.no_grad():
                fisher_estimate = self.fisher_estimator(x.mean(dim=0, keepdim=True)).squeeze()
                self.fisher_weights = 0.9 * self.fisher_weights + 0.1 * fisher_estimate
        if self.training:
            self.num_batches_tracked += 1
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        if self.training:
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_normalized = x_normalized * self.fisher_weights
        if self.weight is not None and self.bias is not None:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized
class MultiScaleNorm(nn.Module):
    def __init(
        self,
        num_features: int,
        num_scales: int = 3,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_scales = num_scales
        self.eps = eps
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)
            for _ in range(num_scales)
        ])
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        self.xi_to_scale = nn.Sequential(
            nn.Linear(1, num_scales),
            nn.Softmax(dim=-1),
        )
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if xi_scale is not None:
            scale_probs = self.xi_to_scale(xi_scale.view(-1, 1) if xi_scale.dim() == 1 else xi_scale)
        else:
            scale_probs = F.softmax(self.scale_weights, dim=-1)
        outputs = []
        for scale_idx, norm in enumerate(self.scale_norms):
            scale_output = norm(x)
            weight = scale_probs[0, scale_idx] if scale_probs.dim() > 1 else scale_probs[scale_idx]
            outputs.append(scale_output * weight)
        output = sum(outputs)
        return output
class RGLayerNorm(nn.Module):
    def __init(
        self,
        normalized_shape: int or Tuple[int, ...],
        xi_scale_dim: int = 1,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.xi_encoder = nn.Sequential(
            nn.Linear(xi_scale_dim, max(1, self.normalized_shape[0] // 4)),
            nn.GELU(),
            nn.Linear(max(1, self.normalized_shape[0] // 4), self.normalized_shape[0]),
            nn.Sigmoid(),
        )
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if xi_scale is not None:
            scale_factor = self.xi_encoder(xi_scale.view(-1, 1) if xi_scale.dim() == 1 else xi_scale)
            for _ in range(x_normalized.dim() - 2):
                scale_factor = scale_factor.unsqueeze(1)
            x_normalized = x_normalized * (1 + scale_factor)
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized