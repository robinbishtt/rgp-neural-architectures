from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)
@dataclass
class EvalResult:
    loss:            float
    accuracy:        float
    top5_accuracy:   Optional[float]      = None
    per_class_acc:   Optional[Dict[int, float]] = None
    correlation_lengths: Optional[List[float]] = None   
    n_samples:       int = 0
    elapsed_s:       float = 0.0
class Evaluator:
    def __init__(
        self,
        device:                       Optional[torch.device] = None,
        compute_top5:                 bool = False,
        compute_per_class:            bool = False,
        compute_correlation_lengths:  bool = False,
    ) -> None:
        if device is None:
            from src.utils.device_manager import DeviceManager
            device = DeviceManager.get_device()
        self.device                      = device
        self.compute_top5                = compute_top5
        self.compute_per_class           = compute_per_class
        self.compute_correlation_lengths = compute_correlation_lengths
    @torch.no_grad()
    def evaluate(
        self,
        model:  nn.Module,
        loader: DataLoader,
    ) -> EvalResult:
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        t0 = time.perf_counter()
        total_loss    = 0.0
        total_correct = 0
        top5_correct  = 0
        n_samples     = 0
        per_class_correct: Dict[int, int] = {}
        per_class_total:   Dict[int, int] = {}
        xi_per_layer: List[List[float]] = []
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.compute_correlation_lengths and hasattr(model, "forward_with_activations"):
                logits, activations = model.forward_with_activations(x)
                xi_layer = self._compute_correlation_lengths(activations, x)
                xi_per_layer.append(xi_layer)
            else:
                logits = model(x)
            total_loss    += criterion(logits, y).item()
            preds          = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            n_samples     += y.size(0)
            if self.compute_top5 and logits.size(1) >= 5:
                top5 = logits.topk(5, dim=1).indices
                top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            if self.compute_per_class:
                for cls_id in y.unique().tolist():
                    mask = (y == cls_id)
                    per_class_correct[cls_id] = per_class_correct.get(cls_id, 0) + (preds[mask] == y[mask]).sum().item()
                    per_class_total[cls_id]   = per_class_total.get(cls_id, 0) + mask.sum().item()
        loss     = total_loss / max(n_samples, 1)
        accuracy = total_correct / max(n_samples, 1)
        per_class: Optional[Dict[int, float]] = None
        if self.compute_per_class:
            per_class = {c: per_class_correct[c] / per_class_total[c]
                         for c in per_class_total}
        xi_mean: Optional[List[float]] = None
        if xi_per_layer:
            arr = np.array(xi_per_layer)  
            xi_mean = arr.mean(axis=0).tolist()
        return EvalResult(
            loss              = loss,
            accuracy          = accuracy,
            top5_accuracy     = top5_correct / max(n_samples, 1) if self.compute_top5 else None,
            per_class_acc     = per_class,
            correlation_lengths = xi_mean,
            n_samples         = n_samples,
            elapsed_s         = time.perf_counter() - t0,
        )
    def evaluate_ood(
        self,
        model:      nn.Module,
        id_loader:  DataLoader,
        ood_loader: DataLoader,
    ) -> Tuple[EvalResult, EvalResult]:
        id_result  = self.evaluate(model, id_loader)
        ood_result = self.evaluate(model, ood_loader)
        logger.info(
            ,
            id_result.accuracy, ood_result.accuracy,
            id_result.accuracy - ood_result.accuracy,
        )
        return id_result, ood_result
    def _compute_correlation_lengths(
        self,
        activations: List[torch.Tensor],
        x: torch.Tensor,
    ) -> List[float]:
        xi_vals = []
        for h_k in activations:
            h_k_cpu = h_k.float().cpu()
            h_mu = h_k_cpu.mean(dim=0, keepdim=True)
            h_c  = h_k_cpu - h_mu
            B = h_c.shape[0]
            F = (h_c.T @ h_c) / max(B - 1, 1)  
            eigenvalues = torch.linalg.eigvalsh(F).numpy()
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) == 0:
                xi_vals.append(float("nan"))
            else:
                xi_k = 1.0 / np.sqrt(np.mean(1.0 / eigenvalues))
                xi_vals.append(float(xi_k))
        return xi_vals