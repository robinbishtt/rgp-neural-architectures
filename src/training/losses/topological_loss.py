from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class RGFlowConsistencyLoss(nn.Module):
    def __init(
        self,
        chi_susceptibility: float = 0.870,
        xi_depth: Optional[float] = None,
        contraction_weight: float = 1.0,
        fixed_point_weight: float = 1.0,
        correlation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.chi_susceptibility = chi_susceptibility
        if xi_depth is None:
            self.xi_depth = -1.0 / math.log(max(chi_susceptibility, 1e-6))
        else:
            self.xi_depth = xi_depth
        self.contraction_weight = contraction_weight
        self.fixed_point_weight = fixed_point_weight
        self.correlation_weight = correlation_weight
        self.expected_decay_rate = chi_susceptibility ** 2
    def compute_metric_contraction_loss(
        self,
        layer_features: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(layer_features) < 2:
            return torch.tensor(0.0, device=layer_features[0].device)
        contraction_losses = []
        for layer_idx in range(len(layer_features) - 1):
            current_features = layer_features[layer_idx]
            next_features = layer_features[layer_idx + 1]
            current_cov = self._compute_covariance(current_features)
            next_cov = self._compute_covariance(next_features)
            current_eig = torch.linalg.eigvalsh(current_cov)
            next_eig = torch.linalg.eigvalsh(next_cov)
            current_eig = torch.clamp(current_eig, min=1e-8)
            next_eig = torch.clamp(next_eig, min=1e-8)
            expected_next = current_eig * self.expected_decay_rate
            layer_contraction = F.mse_loss(
                next_eig.log(),
                expected_next.log(),
            )
            contraction_losses.append(layer_contraction)
        return torch.stack(contraction_losses).mean()
    def compute_fixed_point_convergence_loss(
        self,
        layer_features: List[torch.Tensor],
        target_scale: float = 1.0,
    ) -> torch.Tensor:
        if len(layer_features) < 2:
            return torch.tensor(0.0, device=layer_features[0].device)
        final_features = layer_features[-1]
        final_cov = self._compute_covariance(final_features)
        final_eig = torch.linalg.eigvalsh(final_cov)
        final_eig = torch.clamp(final_eig, min=1e-8)
        target_variance = torch.tensor(target_scale, device=final_eig.device)
        convergence_loss = F.mse_loss(
            final_eig,
            target_variance * torch.ones_like(final_eig),
        )
        return convergence_loss
    def compute_correlation_decay_loss(
        self,
        layer_features: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(layer_features) < 2:
            return torch.tensor(0.0, device=layer_features[0].device)
        correlation_lengths = []
        for features in layer_features:
            cov = self._compute_covariance(features)
            xi = self._estimate_correlation_length(cov)
            correlation_lengths.append(xi)
        xi_tensor = torch.stack(correlation_lengths)
        layer_indices = torch.arange(len(layer_features), device=xi_tensor.device, dtype=torch.float32)
        expected_xi = correlation_lengths[0] * (self.chi_susceptibility ** layer_indices)
        expected_xi = torch.clamp(expected_xi, min=1e-6)
        decay_loss = F.mse_loss(xi_tensor.log(), expected_xi.log())
        return decay_loss
    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        features_centered = features - features.mean(dim=0, keepdim=True)
        batch_size = features.shape[0]
        covariance = torch.matmul(features_centered.t(), features_centered) / (batch_size - 1)
        return covariance + 1e-6 * torch.eye(covariance.shape[0], device=covariance.device)
    def _estimate_correlation_length(self, covariance: torch.Tensor) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(covariance)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        xi = 1.0 / eigenvalues.sqrt().mean()
        return xi
    def forward(
        self,
        layer_features: List[torch.Tensor],
        target_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if target_scale is None:
            target_scale = 1.0
        contraction_loss = self.compute_metric_contraction_loss(layer_features)
        fixed_point_loss = self.compute_fixed_point_convergence_loss(
            layer_features, target_scale
        )
        correlation_loss = self.compute_correlation_decay_loss(layer_features)
        total_loss = (
            self.contraction_weight * contraction_loss +
            self.fixed_point_weight * fixed_point_loss +
            self.correlation_weight * correlation_loss
        )
        loss_info = {
            : total_loss,
            : contraction_loss,
            : fixed_point_loss,
            : correlation_loss,
            : torch.tensor(self.xi_depth),
        }
        return total_loss, loss_info
class InformationBottleneckLoss(nn.Module):
    def __init(
        self,
        beta: float = 1.0,
        target_information: float = 0.5,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.target_information = target_information
    def compute_mutual_information(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        x_norm = F.normalize(x_flat, dim=-1)
        y_norm = F.normalize(y_flat, dim=-1)
        similarity_matrix = torch.matmul(x_norm, y_norm.t())
        pos_sim = similarity_matrix.diag()
        neg_sim = similarity_matrix.sum(dim=-1) - pos_sim
        mi_estimate = (pos_sim / (pos_sim + neg_sim + 1e-8)).mean()
        return mi_estimate
    def forward(
        self,
        input_features: torch.Tensor,
        output_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mi = self.compute_mutual_information(input_features, output_features)
        compression_loss = F.mse_loss(mi, torch.tensor(self.target_information, device=mi.device))
        total_loss = self.beta * compression_loss
        info = {
            : total_loss,
            : mi,
            : mi / (input_features.numel() / output_features.numel() + 1e-8),
        }
        return total_loss, info
class TopologicalRegularizationLoss(nn.Module):
    def __init(
        self,
        persistence_weight: float = 0.1,
        connectivity_weight: float = 0.1,
        num_landmarks: int = 100,
    ) -> None:
        super().__init__()
        self.persistence_weight = persistence_weight
        self.connectivity_weight = connectivity_weight
        self.num_landmarks = num_landmarks
    def compute_persistence_loss(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        batch_size = features.shape[0]
        num_samples = min(self.num_landmarks, batch_size)
        indices = torch.randperm(batch_size)[:num_samples]
        landmarks = features[indices]
        distances = torch.cdist(features, landmarks)
        persistence = distances.std(dim=0).mean()
        target_persistence = 1.0
        loss = F.mse_loss(persistence, torch.tensor(target_persistence, device=persistence.device))
        return loss
    def compute_connectivity_loss(
        self,
        features: torch.Tensor,
        epsilon: float = 0.5,
    ) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        features_norm = F.normalize(features, dim=-1)
        similarity = torch.matmul(features_norm, features_norm.t())
        adjacency = (similarity > epsilon).float()
        degree = adjacency.sum(dim=-1)
        target_degree = torch.log(torch.tensor(features.shape[0], dtype=torch.float32, device=degree.device))
        loss = F.mse_loss(degree.mean(), target_degree)
        return loss
    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        persistence_loss = self.compute_persistence_loss(features)
        connectivity_loss = self.compute_connectivity_loss(features)
        total_loss = (
            self.persistence_weight * persistence_loss +
            self.connectivity_weight * connectivity_loss
        )
        info = {
            : total_loss,
            : persistence_loss,
            : connectivity_loss,
        }
        return total_loss, info
class CombinedRGLoss(nn.Module):
    def __init(
        self,
        base_loss: nn.Module,
        rg_flow_loss: Optional[RGFlowConsistencyLoss] = None,
        ib_loss: Optional[InformationBottleneckLoss] = None,
        topo_loss: Optional[TopologicalRegularizationLoss] = None,
        rg_lambda: float = 0.1,
        ib_lambda: float = 0.05,
        topo_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.rg_flow_loss = rg_flow_loss or RGFlowConsistencyLoss()
        self.ib_loss = ib_loss or InformationBottleneckLoss()
        self.topo_loss = topo_loss or TopologicalRegularizationLoss()
        self.rg_lambda = rg_lambda
        self.ib_lambda = ib_lambda
        self.topo_lambda = topo_lambda
        self.loss_history: Dict[str, List[float]] = {
            : [],
            : [],
            : [],
            : [],
            : [],
        }
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        layer_features: Optional[List[torch.Tensor]] = None,
        input_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_loss = self.base_loss(predictions, targets)
        rg_loss = torch.tensor(0.0, device=predictions.device)
        if layer_features is not None and self.rg_lambda > 0:
            rg_loss, rg_info = self.rg_flow_loss(layer_features)
            rg_loss = self.rg_lambda * rg_loss
        ib_loss = torch.tensor(0.0, device=predictions.device)
        if input_features is not None and layer_features is not None and self.ib_lambda > 0:
            ib_loss, ib_info = self.ib_loss(input_features, layer_features[-1])
            ib_loss = self.ib_lambda * ib_loss
        topo_loss = torch.tensor(0.0, device=predictions.device)
        if layer_features is not None and self.topo_lambda > 0:
            topo_loss, topo_info = self.topo_loss(layer_features[-1])
            topo_loss = self.topo_lambda * topo_loss
        total_loss = base_loss + rg_loss + ib_loss + topo_loss
        self.loss_history['base'].append(base_loss.item())
        self.loss_history['rg_flow'].append(rg_loss.item())
        self.loss_history['ib'].append(ib_loss.item())
        self.loss_history['topo'].append(topo_loss.item())
        self.loss_history['total'].append(total_loss.item())
        loss_info = {
            : total_loss,
            : base_loss,
            : rg_loss,
            : ib_loss,
            : topo_loss,
        }
        return total_loss, loss_info
    def get_loss_statistics(self) -> Dict[str, float]:
        stats = {}
        for key, values in self.loss_history.items():
            if values:
                recent_values = values[-100:]
                stats[f'{key}_mean'] = sum(recent_values) / len(recent_values)
                stats[f'{key}_std'] = (sum((v - stats[f'{key}_mean']) ** 2 for v in recent_values) / len(recent_values)) ** 0.5
            else:
                stats[f'{key}_mean'] = 0.0
                stats[f'{key}_std'] = 0.0
        return stats