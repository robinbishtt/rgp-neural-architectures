import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import tempfile
import os
class TestCheckpointRobustness:
    def test_save_load_produces_identical_output(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(6)
        model = RGNetStandard(input_dim=16, n_classes=4)
        x = torch.randn(4, 16)
        with torch.no_grad():
            out_before = model(x).clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.pt")
            torch.save(model.state_dict(), path)
            model2 = RGNetStandard(input_dim=16, n_classes=4)
            model2.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            out_after = model2(x)
        assert torch.allclose(out_before, out_after, atol=1e-6),            "Checkpoint save/load changed model outputs"
    def test_partial_state_dict_handled_gracefully(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(7)
        model = RGNetStandard(input_dim=16, n_classes=4)
        full_state = model.state_dict()
        partial = {k: v for i, (k, v) in enumerate(full_state.items()) if i % 2 == 0}
        try:
            model.load_state_dict(partial, strict=False)
        except Exception as e:
            pytest.fail(f"strict=False should not raise: {e}")