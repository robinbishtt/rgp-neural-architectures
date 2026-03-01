"""src/architectures/rg_net — RG-Net architecture family."""

from src.architectures.rg_net.rg_net import RGNet
from src.architectures.rg_net.rg_net_shallow import RGNetShallow
from src.architectures.rg_net.rg_net_standard import RGNetStandard
from src.architectures.rg_net.rg_net_deep import RGNetDeep
from src.architectures.rg_net.rg_net_ultra_deep import RGNetUltraDeep
from src.architectures.rg_net.rg_net_variable_width import RGNetVariableWidth
from src.architectures.rg_net.rg_net_multiscale import RGNetMultiScale

__all__ = [
    "RGNet", "RGNetShallow", "RGNetStandard", "RGNetDeep",
    "RGNetUltraDeep", "RGNetVariableWidth", "RGNetMultiScale",
]
