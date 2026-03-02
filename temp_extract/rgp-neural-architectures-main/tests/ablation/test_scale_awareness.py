"""tests/ablation/test_scale_awareness.py"""
import torch
from src.architectures.rg_net.rg_net_multiscale import RGNetMultiScale


def test_multiscale_output_shape():
    m   = RGNetMultiScale(in_features=32, n_classes=4, depth=8, width=32)
    out = m(torch.randn(4, 32))
    assert out.shape == (4, 4)
