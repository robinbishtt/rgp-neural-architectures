from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class FisherInformationEstimator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_samples: int = 100,
        damping: float = 1e-5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.damping = damping
        self.gradient_buffer: List[torch.Tensor] = []
        self.fisher_matrix: Optional[torch.Tensor] = None
        self.register_buffer('running_fisher', torch.zeros(feature_dim, feature_dim))
        self.register_buffer('num_updates', torch.zeros(1, dtype=torch.long))
        self.diagonal_approximation = True
    def compute_gradient_covariance(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        batch_size = features.shape[0]
        features.requires_grad_(True)
        logits = features
        loss = loss_fn(logits, labels)
        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=features,
            create_graph=False,
            retain_graph=False,
        )[0]
        features.requires_grad_(False)
        return gradients
    def update_fisher_information(
        self,
        gradients: torch.Tensor,
        momentum: float = 0.9,
    ) -> torch.Tensor:
        batch_size = gradients.shape[0]
        if self.diagonal_approximation:
            fisher_diag = (gradients ** 2).mean(dim=0)
            self.running_fisher = momentum * self.running_fisher + (1 - momentum) * fisher_diag
        else:
            gradients_flat = gradients.view(batch_size, -1)
            fisher_matrix = torch.matmul(gradients_flat.t(), gradients_flat) / batch_size
            fisher_matrix = fisher_matrix + self.damping * torch.eye(
                fisher_matrix.shape[0],
                device=fisher_matrix.device,
            )
            self.running_fisher = momentum * self.running_fisher + (1 - momentum) * fisher_matrix
        self.num_updates += 1
        return self.running_fisher
    def get_neuron_importance(self) -> torch.Tensor:
        if self.diagonal_approximation:
            importance = self.running_fisher.sqrt()
        else:
            importance = self.running_fisher.diagonal().sqrt()
        return importance / (importance.max() + 1e-8)
    def compute_correlation_length(self) -> torch.Tensor:
        if self.diagonal_approximation:
            eigenvalues = self.running_fisher + self.damping
        else:
            eigenvalues = torch.linalg.eigvalsh(self.running_fisher)
        eigenvalues = torch.clamp(eigenvalues, min=self.damping)
        xi = 1.0 / eigenvalues.sqrt().mean()
        return xi
    def reset_buffer(self) -> None:
        self.gradient_buffer = []
class FisherDynamicRouter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 8,
        sparsity_threshold: float = 0.1,
        fisher_weight: float = 1.0,
        enable_masking: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.sparsity_threshold = sparsity_threshold
        self.fisher_weight = fisher_weight
        self.enable_masking = enable_masking
        self.expert_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            for _ in range(num_experts)
        ])
        self.expert_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features))
            for _ in range(num_experts)
        ])
        self.routing_network = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, num_experts),
        )
        self.fisher_estimator = FisherInformationEstimator(in_features)
        self.neuron_mask = nn.Parameter(torch.ones(in_features), requires_grad=False)
        self.register_buffer('fisher_history', torch.zeros(1000))
        self.register_buffer('mask_history', torch.zeros(1000, in_features))
        self.history_ptr = 0
    def forward(
        self,
        x: torch.Tensor,
        return_routing_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size = x.shape[0]
        masked_x = x * self.neuron_mask.unsqueeze(0)
        routing_logits = self.routing_network(masked_x.mean(dim=0, keepdim=True))
        routing_weights = F.softmax(routing_logits, dim=-1)
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_out = F.linear(
                masked_x,
                self.expert_weights[expert_idx],
                self.expert_biases[expert_idx],
            )
            expert_outputs.append(expert_out * routing_weights[0, expert_idx])
        output = sum(expert_outputs)
        aux_info = {
            : routing_weights,
            : self.neuron_mask.clone(),
            : (self.neuron_mask > 0).sum().item(),
        }
        if return_routing_weights:
            return output, routing_weights, aux_info
        return output, None, aux_info
    def update_masks_from_fisher(self, threshold: Optional[float] = None) -> torch.Tensor:
        if threshold is None:
            threshold = self.sparsity_threshold
        neuron_importance = self.fisher_estimator.get_neuron_importance()
        if neuron_importance.shape != self.neuron_mask.shape:
            if neuron_importance.numel() == self.neuron_mask.numel():
                neuron_importance = neuron_importance.view(self.neuron_mask.shape)
            else:
                return self.neuron_mask
        new_mask = (neuron_importance > threshold).float()
        self.neuron_mask.data = new_mask
        self.mask_history[self.history_ptr] = new_mask
        return new_mask
    def compute_fisher_loss(self) -> torch.Tensor:
        neuron_importance = self.fisher_estimator.get_neuron_importance()
        if neuron_importance.shape != self.neuron_mask.shape:
            return torch.tensor(0.0, device=neuron_importance.device)
        masked_importance = neuron_importance * self.neuron_mask
        fisher_loss = -masked_importance.mean() * self.fisher_weight
        return fisher_loss
    def estimate_and_update_fisher(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        gradients = self.fisher_estimator.compute_gradient_covariance(
            features, labels, loss_fn
        )
        self.fisher_estimator.update_fisher_information(gradients)
        fisher_info = self.fisher_estimator.get_neuron_importance()
        if self.history_ptr < len(self.fisher_history):
            self.fisher_history[self.history_ptr] = fisher_info.mean()
        return fisher_info
class FisherRegularizedLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        fisher_routers: List[FisherDynamicRouter],
        fisher_lambda: float = 0.1,
        warmup_epochs: int = 5,
        scheduler_type: str = 'linear',
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.fisher_routers = fisher_routers
        self.fisher_lambda_init = fisher_lambda
        self.fisher_lambda = 0.0
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type
        self.current_epoch = 0
        self.fisher_loss_history: List[float] = []
        self.base_loss_history: List[float] = []
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_loss_value = self.base_loss(predictions, targets)
        fisher_loss = torch.tensor(0.0, device=predictions.device)
        for router in self.fisher_routers:
            fisher_loss = fisher_loss + router.compute_fisher_loss()
        total_loss = base_loss_value + self.fisher_lambda * fisher_loss
        self.fisher_loss_history.append(fisher_loss.item())
        self.base_loss_history.append(base_loss_value.item())
        loss_info = {
            : total_loss,
            : base_loss_value,
            : fisher_loss,
            : torch.tensor(self.fisher_lambda),
        }
        return total_loss, loss_info
    def step_epoch(self) -> None:
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            if self.scheduler_type == 'linear':
                self.fisher_lambda = self.fisher_lambda_init * progress
            elif self.scheduler_type == 'cosine':
                self.fisher_lambda = self.fisher_lambda_init * (1 - math.cos(progress * math.pi / 2))
            elif self.scheduler_type == 'exponential':
                self.fisher_lambda = self.fisher_lambda_init * (0.1 ** (1 - progress))
        else:
            self.fisher_lambda = self.fisher_lambda_init
    def get_fisher_statistics(self) -> Dict[str, float]:
        if not self.fisher_loss_history:
            return {'mean_fisher_loss': 0.0, 'fisher_ratio': 0.0}
        mean_fisher = sum(self.fisher_loss_history[-100:]) / min(len(self.fisher_loss_history), 100)
        mean_base = sum(self.base_loss_history[-100:]) / min(len(self.base_loss_history), 100)
        fisher_ratio = mean_fisher / (mean_base + 1e-8)
        return {
            : mean_fisher,
            : mean_base,
            : fisher_ratio,
            : self.fisher_lambda,
        }
import math