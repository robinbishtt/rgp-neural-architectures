"""
src/training/training_utils.py

Shared training utility functions used across Trainer, DistributedTrainer,
MixedPrecisionTrainer, and CurriculumTrainer.
"""
from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    top_k:    int = 1,
) -> float:
    """
    Compute top-k classification accuracy.

    Args:
        model:  trained classifier
        loader: validation DataLoader
        device: computation device
        top_k:  k for top-k accuracy (default 1 = standard accuracy)

    Returns:
        Accuracy in [0, 1]
    """
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            if top_k == 1:
                pred     = out.argmax(dim=-1)
                correct += (pred == y).sum().item()
            else:
                _, top_k_pred = out.topk(top_k, dim=-1)
                correct += top_k_pred.eq(y.unsqueeze(1)).any(dim=-1).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Compute the L{norm_type} norm of all gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type)


def clip_gradients(
    model:     nn.Module,
    max_norm:  float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradients by global norm. Returns the gradient norm before clipping.
    """
    return float(nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type))


def freeze_layers(
    model:      nn.Module,
    n_freeze:   int,
) -> int:
    """
    Freeze the first n_freeze linear layers (set requires_grad=False).

    Used in progressive training where lower layers are frozen first
    to stabilize the critical initialization before upper layers are trained.

    Returns:
        Number of layers actually frozen.
    """
    frozen = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if frozen < n_freeze:
                for p in m.parameters():
                    p.requires_grad_(False)
                frozen += 1
            else:
                break
    return frozen


def cosine_similarity_layers(
    model_a: nn.Module,
    model_b: nn.Module,
) -> List[float]:
    """
    Compute per-layer cosine similarity between two models' weight matrices.

    Used to measure how much two training trajectories diverge, as a
    reproducibility diagnostic.

    Returns:
        List of cosine similarities per linear layer.
    """
    similarities = []
    params_a = [p for p in model_a.parameters() if p.dim() >= 2]
    params_b = [p for p in model_b.parameters() if p.dim() >= 2]
    for pa, pb in zip(params_a, params_b):
        flat_a = pa.detach().view(-1)
        flat_b = pb.detach().view(-1)
        sim    = torch.dot(flat_a, flat_b) / (flat_a.norm() * flat_b.norm() + 1e-12)
        similarities.append(float(sim.item()))
    return similarities
 