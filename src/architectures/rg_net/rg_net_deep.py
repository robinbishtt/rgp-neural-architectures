from __future__ import annotations
from src.architectures.rg_net.rg_net import RGNetStandard as RGNet
class RGNetDeep(RGNet):
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 depth: int = 500, width: int = 512, **kwargs):
        kwargs.setdefault("use_gradient_checkpointing", True)
        super().__init__(in_features=in_features, n_classes=n_classes,
                         depth=depth, width=width, **kwargs)