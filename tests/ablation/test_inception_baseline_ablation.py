"""tests/ablation/test_inception_baseline_ablation.py"""
import torch


class TestInceptionBaselineAblation:
    def test_inception_output_shape(self):
        from src.architectures.baselines.inception_baseline import InceptionBaseline
        model = InceptionBaseline(input_dim=16, n_classes=4, n_blocks=2, d_model=32)
        out = model(torch.randn(4, 16))
        assert out.shape == (4, 4)

    def test_multi_branch_gradient(self):
        import torch.nn as nn
        from src.architectures.baselines.inception_baseline import InceptionBaseline
        torch.manual_seed(4)
        model = InceptionBaseline(input_dim=16, n_classes=4)
        x = torch.randn(4, 16)
        nn.CrossEntropyLoss()(model(x), torch.randint(0, 4, (4,))).backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()
 