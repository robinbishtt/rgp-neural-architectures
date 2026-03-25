from __future__ import annotations
from src.architectures.rg_net.rg_net import RGNetStandard as RGNet
class RGNetUltraDeep(RGNet):
    def __init__(self, in_features: int = 784, n_classes: int = 10,
                 depth: int = 1000, width: int = 1024, **kwargs):
        if depth < 500:
            raise ValueError(f"RGNetUltraDeep expects depth >= 500, got {depth}")
        kwargs["use_gradient_checkpointing"] = True
        super().__init__(in_features=in_features, n_classes=n_classes,
                         depth=depth, width=width, **kwargs)