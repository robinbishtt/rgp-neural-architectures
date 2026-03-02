"""tests/ablation/test_depth_ablation.py"""
import torch
from src.architectures.rg_net.rg_net_shallow import RGNetShallow
from src.architectures.rg_net.rg_net_standard import RGNetStandard


def test_shallow_fewer_params_than_standard():
    shallow  = RGNetShallow(in_features=32, n_classes=4, depth=10, width=64)
    standard = RGNetStandard(in_features=32, n_classes=4, depth=100, width=64)
    p_s  = sum(p.numel() for p in shallow.parameters())
    p_st = sum(p.numel() for p in standard.parameters())
    assert p_s < p_st
