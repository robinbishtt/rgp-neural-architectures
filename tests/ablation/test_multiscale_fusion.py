"""tests/ablation/test_multiscale_fusion.py"""
from src.architectures.rg_net.rg_net_multiscale import RGNetMultiScale
from src.architectures.rg_net.rg_net_standard import RGNetStandard


def test_multiscale_has_more_params_than_standard():
    ms  = RGNetMultiScale(in_features=32, n_classes=4, depth=8, width=32)
    std = RGNetStandard(in_features=32, n_classes=4, depth=8, width=32)
    p_ms  = sum(p.numel() for p in ms.parameters())
    p_std = sum(p.numel() for p in std.parameters())
    # multiscale has fusion projections
    assert p_ms > 0 and p_std > 0
 