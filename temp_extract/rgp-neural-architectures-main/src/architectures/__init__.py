"""
src/architectures — Neural architecture families.

    rg_net/    : RGNet, RGNetShallow, RGNetStandard, RGNetDeep,
                 RGNetUltraDeep, RGNetVariableWidth, RGNetMultiScale
    baselines/ : MLPBaseline, ResNetBaseline, DenseNetBaseline,
                 VGGBaseline, TransformerBaseline, AttentionBaseline,
                 InceptionBaseline
"""
from src.architectures.rg_net import (
    RGNet, RGNetShallow, RGNetStandard, RGNetDeep,
    RGNetUltraDeep, RGNetVariableWidth, RGNetMultiScale,
)
from src.architectures.baselines import (
    MLPBaseline, ResNetBaseline, DenseNetBaseline, VGGBaseline,
    TransformerBaseline, AttentionBaseline, InceptionBaseline,
)

__all__ = [
    "RGNet", "RGNetShallow", "RGNetStandard", "RGNetDeep",
    "RGNetUltraDeep", "RGNetVariableWidth", "RGNetMultiScale",
    "MLPBaseline", "ResNetBaseline", "DenseNetBaseline", "VGGBaseline",
    "TransformerBaseline", "AttentionBaseline", "InceptionBaseline",
]
