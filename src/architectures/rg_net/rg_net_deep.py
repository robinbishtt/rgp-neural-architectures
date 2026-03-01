"""
src/architectures/rg_net/rg_net_deep.py

RGNetDeep  L=500 configuration for depth scaling studies.
"""
from __future__ import annotations
from src.architectures.rg_net.rg_net import RGNet


class RGNetDeep(RGNet):
    """
    Deep RG-Net for scaling studies. L=500, N=512.
    Requires gradient checkpointing to fit in 24GB VRAM.
    """
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 depth: int = 500, width: int = 512, **kwargs):
        kwargs.setdefault("use_gradient_checkpointing", True)
        super().__init__(in_features=in_features, n_classes=n_classes,
                         depth=depth, width=width, **kwargs)
