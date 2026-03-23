"""
src/architectures/rg_net/rg_net_shallow.py

RGNetShallow - L=10-50 configuration for rapid prototyping.
"""
from __future__ import annotations
from src.architectures.rg_net.rg_net import RGNetStandard as RGNet


class RGNetShallow(RGNet):
    """
    Shallow RG-Net for fast prototyping and hyperparameter search.
    Depth range: L = 10 to 50. Width: N = 64 to 256.
    """
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 depth: int = 20, width: int = 128, **kwargs):
        if not (10 <= depth <= 50):
            raise ValueError(f"RGNetShallow expects depth 10-50, got {depth}")
        super().__init__(in_features=in_features, n_classes=n_classes,
                         depth=depth, width=width, **kwargs)
 