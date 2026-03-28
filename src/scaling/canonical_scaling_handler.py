from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class CorrelationLengthEstimator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        estimation_method: str = 'eigenvalue',
        damping: float = 1e-8,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.estimation_method = estimation_method
        self.damping = damping
        self.register_buffer('covariance_running', torch.zeros(feature_dim, feature_dim))
        self.register_buffer('num_updates', torch.zeros(1, dtype=torch.long))
        self.momentum = 0.9
    def compute_covariance_spectrum(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = features.shape[0]
        features_centered = features - features.mean(dim=0, keepdim=True)
        covariance = torch.matmul(features_centered.t(), features_centered) / (batch_size - 1)
        covariance = covariance + self.damping * torch.eye(
            self.feature_dim,
            device=covariance.device,
        )
        return covariance
    def estimate_xi_from_spectrum(
        self,
        covariance: torch.Tensor,
    ) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(covariance)
        eigenvalues = torch.clamp(eigenvalues, min=self.damping)
        if self.estimation_method == 'eigenvalue':
            xi = 1.0 / eigenvalues.sqrt().mean()
        elif self.estimation_method == 'max_eigenvalue':
            xi = 1.0 / eigenvalues.max().sqrt()
        elif self.estimation_method == 'geometric_mean':
            log_eig = eigenvalues.log()
            xi = 1.0 / (log_eig.mean() / 2).exp()
        else:
            xi = 1.0 / eigenvalues.sqrt().mean()
        return xi
    def forward(
        self,
        features: torch.Tensor,
        update_running: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        covariance = self.compute_covariance_spectrum(features)
        if update_running:
            self.covariance_running = self.momentum * self.covariance_running + (1 - self.momentum) * covariance
            self.num_updates += 1
        xi = self.estimate_xi_from_spectrum(covariance)
        return xi, covariance
    def get_running_xi(self) -> torch.Tensor:
        return self.estimate_xi_from_spectrum(self.covariance_running)
class CanonicalScalingHandler(nn.Module):
    def __init(
        self,
        xi_data_init: float = 50.0,
        xi_target: float = 1.0,
        chi_susceptibility: float = 0.870,
        max_depth: int = 50,
        min_depth: int = 1,
        enable_dynamic_depth: bool = True,
    ) -> None:
        super().__init__()
        self.xi_target = xi_target
        self.chi_susceptibility = chi_susceptibility
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.enable_dynamic_depth = enable_dynamic_depth
        self.xi_depth = -1.0 / math.log(max(chi_susceptibility, 1e-6))
        self.xi_data = nn.Parameter(torch.tensor(xi_data_init))
        self.depth_controller = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.register_buffer('depth_history', torch.zeros(10000))
        self.register_buffer('xi_history', torch.zeros(10000))
        self.history_ptr = 0
        self.current_depth = min_depth
    def compute_l_min(self, xi_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        if xi_data is None:
            xi_data = F.softplus(self.xi_data) + 1.0
        ratio = xi_data / self.xi_target
        ratio = torch.clamp(ratio, min=1.0)
        l_min = self.xi_depth * torch.log(ratio)
        return l_min
    def compute_depth_from_complexity(
        self,
        xi_data: torch.Tensor,
    ) -> int:
        l_min = self.compute_l_min(xi_data)
        if self.enable_dynamic_depth:
            depth_input = l_min.view(1, 1)
            depth_adjustment = self.depth_controller(depth_input).squeeze()
            adjusted_depth = l_min + depth_adjustment
        else:
            adjusted_depth = l_min
        depth = int(torch.clamp(
            adjusted_depth.round(),
            min=self.min_depth,
            max=self.max_depth,
        ).item())
        return depth
    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        batch_xi = self.estimate_batch_xi(features)
        depth = self.compute_depth_from_complexity(batch_xi)
        self.current_depth = depth
        if self.history_ptr < len(self.depth_history):
            self.depth_history[self.history_ptr] = depth
            self.xi_history[self.history_ptr] = batch_xi.item()
            self.history_ptr += 1
        info = {
            : batch_xi,
            : torch.tensor(self.xi_depth),
            : self.compute_l_min(batch_xi),
            : depth,
            : torch.tensor(self.xi_target),
        }
        return depth, info
    def estimate_batch_xi(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        feature_variance = features.var(dim=-1).mean()
        xi_estimate = torch.log1p(feature_variance * 10)
        return xi_estimate
    def get_scaling_statistics(self) -> Dict[str, float]:
        valid_depths = self.depth_history[self.depth_history > 0]
        valid_xis = self.xi_history[self.xi_history > 0]
        if len(valid_depths) == 0:
            return {
                : self.min_depth,
                : 0.0,
                : self.xi_target,
                : self.xi_depth,
            }
        return {
            : valid_depths.mean().item(),
            : valid_depths.std().item(),
            : valid_depths.min().item(),
            : valid_depths.max().item(),
            : valid_xis.mean().item(),
            : valid_xis.std().item(),
            : self.xi_depth,
        }
class StochasticDepthController(nn.Module):
    def __init(
        self,
        max_depth: int,
        survival_prob_init: float = 1.0,
        survival_prob_min: float = 0.5,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.survival_prob_init = survival_prob_init
        self.survival_prob_min = survival_prob_min
        self.layer_survival_probs = nn.Parameter(
            torch.linspace(survival_prob_init, survival_prob_min, max_depth),
            requires_grad=False,
        )
    def forward(
        self,
        layer_idx: int,
        training: bool = True,
    ) -> torch.Tensor:
        if not training:
            return torch.tensor(1.0)
        survival_prob = torch.sigmoid(self.layer_survival_probs[layer_idx])
        survival = torch.bernoulli(survival_prob)
        return survival
    def get_expected_depth(self) -> torch.Tensor:
        return self.layer_survival_probs.sigmoid().sum()
class AdaptiveRGDepth(nn.Module):
    def __init(
        self,
        base_depth: int = 18,
        min_depth: int = 3,
        max_depth: int = 50,
        xi_data_default: float = 50.0,
        xi_target: float = 1.0,
        chi: float = 0.870,
    ) -> None:
        super().__init__()
        self.base_depth = base_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scaling_handler = CanonicalScalingHandler(
            xi_data_init=xi_data_default,
            xi_target=xi_target,
            chi_susceptibility=chi,
            max_depth=max_depth,
            min_depth=min_depth,
        )
        self.stochastic_controller = StochasticDepthController(max_depth)
        self.complexity_estimator = CorrelationLengthEstimator(
            feature_dim=512,
            estimation_method='eigenvalue',
        )
        self.depth_cache: Dict[int, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    def compute_adaptive_depth(
        self,
        x: torch.Tensor,
        use_cache: bool = True,
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        if use_cache:
            cache_key = self._compute_cache_key(x)
            if cache_key in self.depth_cache:
                self.cache_hits += 1
                depth = self.depth_cache[cache_key]
                info = {
                    : torch.tensor(0.0),
                    : torch.tensor(True),
                    : depth,
                }
                return depth, info
            self.cache_misses += 1
        xi, _ = self.complexity_estimator(x, update_running=True)
        depth, info = self.scaling_handler(x)
        if use_cache:
            self.depth_cache[cache_key] = depth
        info['cached'] = torch.tensor(False)
        return depth, info
    def _compute_cache_key(self, x: torch.Tensor) -> int:
        mean_val = x.mean().item()
        std_val = x.std().item()
        return hash((round(mean_val, 4), round(std_val, 4)))
    def get_cache_statistics(self) -> Dict[str, float]:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return {'hit_rate': 0.0, 'cache_size': len(self.depth_cache)}
        return {
            : self.cache_hits / total,
            : len(self.depth_cache),
            : total,
        }
    def forward(self, x: torch.Tensor) -> Tuple[int, Dict[str, torch.Tensor]]:
        return self.compute_adaptive_depth(x, use_cache=True)
class DepthScheduler:
    def __init(
        self,
        initial_depth: int = 10,
        final_depth: int = 50,
        warmup_epochs: int = 10,
        schedule_type: str = 'linear',
    ) -> None:
        self.initial_depth = initial_depth
        self.final_depth = final_depth
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self.current_depth = initial_depth
    def step(self) -> int:
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            if self.schedule_type == 'linear':
                self.current_depth = int(
                    self.initial_depth + (self.final_depth - self.initial_depth) * progress
                )
            elif self.schedule_type == 'cosine':
                cosine_prog = (1 - math.cos(progress * math.pi)) / 2
                self.current_depth = int(
                    self.initial_depth + (self.final_depth - self.initial_depth) * cosine_prog
                )
            elif self.schedule_type == 'exponential':
                exp_prog = (self.final_depth / self.initial_depth) ** progress
                self.current_depth = int(self.initial_depth * exp_prog)
        else:
            self.current_depth = self.final_depth
        return self.current_depth
    def get_state(self) -> Dict[str, int]:
        return {
            : self.current_epoch,
            : self.current_depth,
            : self.initial_depth,
            : self.final_depth,
        }