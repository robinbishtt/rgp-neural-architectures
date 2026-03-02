"""tests/scaling/test_h3_generalization.py"""
import torch
from src.architectures.rg_net.rg_net import RGNet
from src.architectures.baselines.mlp_baseline import MLPBaseline


def test_rgnet_and_mlp_produce_valid_outputs():
    rg  = RGNet(in_features=32, n_classes=4, depth=5, width=32)
    mlp = MLPBaseline(in_features=32, n_classes=4, depth=3, width=32)
    x = torch.randn(8, 32)
    for m in [rg, mlp]:
        out = m(x)
        assert out.shape == (8, 4)
        assert not torch.isnan(out).any()
 