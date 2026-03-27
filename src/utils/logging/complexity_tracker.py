from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class FisherInformationMonitor(nn.Module):
    def __init(
        self,
        feature_dim: int,
        window_size: int = 100,
        damping: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.damping = damping
        self.register_buffer('fisher_accumulator', torch.zeros(feature_dim, feature_dim))
        self.register_buffer('gradient_buffer', torch.zeros(window_size, feature_dim))
        self.register_buffer('buffer_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('num_updates', torch.zeros(1, dtype=torch.long))
        self.momentum = 0.99
    def update(self, gradients: torch.Tensor) -> torch.Tensor:
        batch_size = gradients.shape[0]
        for i in range(batch_size):
            ptr = self.buffer_ptr.item()
            self.gradient_buffer[ptr] = gradients[i]
            self.buffer_ptr[0] = (ptr + 1) % self.window_size
        if self.num_updates >= self.window_size:
            fisher_estimate = torch.matmul(
                self.gradient_buffer.t(),
                self.gradient_buffer,
            ) / self.window_size
            fisher_estimate = fisher_estimate + self.damping * torch.eye(
                self.feature_dim,
                device=fisher_estimate.device,
            )
            self.fisher_accumulator = self.momentum * self.fisher_accumulator + (1 - self.momentum) * fisher_estimate
        self.num_updates += batch_size
        return self.fisher_accumulator
    def get_effective_dimension(self) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(self.fisher_accumulator)
        eigenvalues = torch.clamp(eigenvalues, min=self.damping)
        effective_dim = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        return effective_dim
    def get_neuron_importance(self) -> torch.Tensor:
        importance = self.fisher_accumulator.diagonal().sqrt()
        importance = importance / (importance.max() + 1e-8)
        return importance
class CorrelationLengthMonitor(nn.Module):
    def __init(
        self,
        max_history: int = 1000,
        estimation_method: str = 'spectral',
    ) -> None:
        super().__init__()
        self.max_history = max_history
        self.estimation_method = estimation_method
        self.register_buffer('xi_history', torch.zeros(max_history))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.layer_xi_values: Dict[int, List[float]] = {}
    def estimate_xi(
        self,
        features: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        if method is None:
            method = self.estimation_method
        if features.dim() == 3:
            features = features.mean(dim=1)
        features_centered = features - features.mean(dim=0, keepdim=True)
        batch_size = features.shape[0]
        covariance = torch.matmul(features_centered.t(), features_centered) / (batch_size - 1)
        covariance = covariance + 1e-8 * torch.eye(covariance.shape[0], device=covariance.device)
        if method == 'spectral':
            eigenvalues = torch.linalg.eigvalsh(covariance)
            eigenvalues = torch.clamp(eigenvalues, min=1e-8)
            xi = 1.0 / eigenvalues.sqrt().mean()
        elif method == 'trace':
            xi = covariance.trace().sqrt()
        elif method == 'frobenius':
            xi = covariance.norm('fro')
        else:
            eigenvalues = torch.linalg.eigvalsh(covariance)
            eigenvalues = torch.clamp(eigenvalues, min=1e-8)
            xi = 1.0 / eigenvalues.sqrt().mean()
        return xi
    def update_history(self, xi: torch.Tensor, layer_idx: Optional[int] = None) -> None:
        ptr = self.history_ptr.item()
        if ptr < self.max_history:
            self.xi_history[ptr] = xi.item()
            self.history_ptr[0] = ptr + 1
        if layer_idx is not None:
            if layer_idx not in self.layer_xi_values:
                self.layer_xi_values[layer_idx] = []
            self.layer_xi_values[layer_idx].append(xi.item())
    def get_xi_statistics(self) -> Dict[str, float]:
        valid_history = self.xi_history[self.xi_history > 0]
        if len(valid_history) == 0:
            return {
                : 0.0,
                : 0.0,
                : 0.0,
                : 0.0,
            }
        return {
            : valid_history.mean().item(),
            : valid_history.std().item(),
            : valid_history.min().item(),
            : valid_history.max().item(),
        }
    def get_layer_xi_decay(self) -> List[Tuple[int, float]]:
        layer_stats = []
        for layer_idx, values in sorted(self.layer_xi_values.items()):
            if values:
                mean_xi = sum(values[-100:]) / min(len(values), 100)
                layer_stats.append((layer_idx, mean_xi))
        return layer_stats
class ComplexityTracker(nn.Module):
    def __init(
        self,
        num_layers: int = 18,
        feature_dim: int = 512,
        window_size: int = 100,
        enable_fisher: bool = True,
        enable_correlation: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.enable_fisher = enable_fisher
        self.enable_correlation = enable_correlation
        if enable_fisher:
            self.fisher_monitors = nn.ModuleList([
                FisherInformationMonitor(feature_dim, window_size)
                for _ in range(num_layers)
            ])
        if enable_correlation:
            self.correlation_monitor = CorrelationLengthMonitor()
        self.register_buffer('global_step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('layer_complexity', torch.zeros(num_layers))
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    def register_layer_hooks(self, model: nn.Module, layer_names: List[str]) -> None:
        def create_hook(layer_idx: int):
            def hook(module, input, output):
                self._layer_forward_hook(layer_idx, output)
                return output
            return hook
        for idx, (name, module) in enumerate(model.named_modules()):
            if any(ln in name for ln in layer_names):
                if idx < self.num_layers:
                    hook = module.register_forward_hook(create_hook(idx))
                    self.hooks.append(hook)
    def _layer_forward_hook(
        self,
        layer_idx: int,
        output: torch.Tensor,
    ) -> None:
        if output.requires_grad:
            output.register_hook(lambda grad: self._layer_backward_hook(layer_idx, grad))
        if self.enable_correlation:
            with torch.no_grad():
                xi = self.correlation_monitor.estimate_xi(output)
                self.correlation_monitor.update_history(xi, layer_idx)
    def _layer_backward_hook(
        self,
        layer_idx: int,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        if self.enable_fisher and layer_idx < len(self.fisher_monitors):
            if gradients.dim() > 2:
                grad_flat = gradients.view(gradients.shape[0], -1)
                if grad_flat.shape[-1] >= self.feature_dim:
                    grad_sample = grad_flat[:, :self.feature_dim]
                    self.fisher_monitors[layer_idx].update(grad_sample)
        return gradients
    def get_complexity_report(self) -> Dict[str, any]:
        report = {
            : self.global_step.item(),
            : self.num_layers,
        }
        if self.enable_correlation:
            xi_stats = self.correlation_monitor.get_xi_statistics()
            report['xi_statistics'] = xi_stats
            layer_decay = self.correlation_monitor.get_layer_xi_decay()
            report['layer_xi_decay'] = layer_decay
            if len(layer_decay) >= 2:
                xi_0 = layer_decay[0][1]
                xi_final = layer_decay[-1][1]
                if xi_0 > 0 and xi_final > 0:
                    effective_depth = math.log(xi_final / xi_0) / math.log(self._estimate_chi())
                    report['effective_depth'] = effective_depth
        if self.enable_fisher:
            fisher_stats = []
            for idx, monitor in enumerate(self.fisher_monitors):
                effective_dim = monitor.get_effective_dimension().item()
                neuron_importance = monitor.get_neuron_importance()
                fisher_stats.append({
                    : idx,
                    : effective_dim,
                    : neuron_importance.mean().item(),
                    : neuron_importance.max().item(),
                })
            report['fisher_statistics'] = fisher_stats
        return report
    def _estimate_chi(self) -> float:
        layer_decay = self.correlation_monitor.get_layer_xi_decay()
        if len(layer_decay) < 2:
            return 0.870
        xi_values = [xi for _, xi in layer_decay]
        decay_rates = []
        for i in range(len(xi_values) - 1):
            if xi_values[i] > 0:
                rate = xi_values[i + 1] / xi_values[i]
                decay_rates.append(rate)
        if decay_rates:
            return sum(decay_rates) / len(decay_rates)
        return 0.870
    def step(self) -> None:
        self.global_step += 1
    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
class RealTimeVisualizer:
    def __init(
        self,
        tracker: ComplexityTracker,
        update_interval: int = 100,
    ) -> None:
        self.tracker = tracker
        self.update_interval = update_interval
        self.xi_traces: List[List[float]] = [[] for _ in range(tracker.num_layers)]
        self.fisher_traces: List[List[float]] = [[] for _ in range(tracker.num_layers)]
    def update(self) -> Dict[str, any]:
        step = self.tracker.global_step.item()
        if step % self.update_interval != 0:
            return {}
        report = self.tracker.get_complexity_report()
        if 'layer_xi_decay' in report:
            for layer_idx, xi in report['layer_xi_decay']:
                if layer_idx < len(self.xi_traces):
                    self.xi_traces[layer_idx].append(xi)
        if 'fisher_statistics' in report:
            for stat in report['fisher_statistics']:
                layer_idx = stat['layer']
                if layer_idx < len(self.fisher_traces):
                    self.fisher_traces[layer_idx].append(stat['mean_importance'])
        return report
    def get_traces(self) -> Dict[str, List[List[float]]]:
        return {
            : self.xi_traces,
            : self.fisher_traces,
        }
class RGFlowValidator:
    def __init(
        self,
        chi_expected: float = 0.870,
        tolerance: float = 0.05,
    ) -> None:
        self.chi_expected = chi_expected
        self.tolerance = tolerance
        self.validation_history: List[Dict[str, float]] = []
    def validate_flow(
        self,
        tracker: ComplexityTracker,
    ) -> Dict[str, any]:
        report = tracker.get_complexity_report()
        validation = {
            : True,
            : [],
        }
        if 'layer_xi_decay' in report:
            layer_decay = report['layer_xi_decay']
            if len(layer_decay) >= 2:
                xi_values = [xi for _, xi in layer_decay]
                decay_rates = []
                for i in range(len(xi_values) - 1):
                    if xi_values[i] > 0:
                        rate = xi_values[i + 1] / xi_values[i]
                        decay_rates.append(rate)
                if decay_rates:
                    chi_observed = sum(decay_rates) / len(decay_rates)
                    chi_error = abs(chi_observed - self.chi_expected) / self.chi_expected
                    validation['chi_observed'] = chi_observed
                    validation['chi_expected'] = self.chi_expected
                    validation['chi_error'] = chi_error
                    if chi_error > self.tolerance:
                        validation['is_valid'] = False
                        validation['violations'].append(
                            f'Chi deviation: {chi_error:.4f} > {self.tolerance}'
                        )
        if 'xi_statistics' in report:
            xi_stats = report['xi_statistics']
            if xi_stats['std_xi'] / (xi_stats['mean_xi'] + 1e-8) > 0.5:
                validation['violations'].append('High XI variance detected')
        self.validation_history.append(validation)
        return validation
    def get_validation_summary(self) -> Dict[str, float]:
        if not self.validation_history:
            return {'valid_ratio': 0.0, 'mean_chi_error': 0.0}
        valid_count = sum(1 for v in self.validation_history if v['is_valid'])
        chi_errors = [
            v.get('chi_error', 0.0)
            for v in self.validation_history
            if 'chi_error' in v
        ]
        return {
            : valid_count / len(self.validation_history),
            : sum(chi_errors) / len(chi_errors) if chi_errors else 0.0,
            : len(self.validation_history),
        }