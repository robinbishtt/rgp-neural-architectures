import pytest
torch = pytest.importorskip("torch", reason="torch not installed")  
import torch
class TestSeedRegistry:
    def test_singleton(self):
        from src.utils.seed_registry import SeedRegistry
        assert SeedRegistry.get_instance() is SeedRegistry.get_instance()
    def test_reproducibility(self):
        from src.utils.seed_registry import SeedRegistry
        r = SeedRegistry.get_instance()
        r.set_master_seed(42)
        v1 = torch.randn(8)
        r.set_master_seed(42)
        v2 = torch.randn(8)
        assert torch.equal(v1, v2)
    def test_worker_seeds_deterministic(self):
        from src.utils.seed_registry import SeedRegistry
        r = SeedRegistry.get_instance()
        r.set_master_seed(0)
        assert r.get_worker_seed(0) == r.get_worker_seed(0)
    def test_worker_seeds_differ(self):
        from src.utils.seed_registry import SeedRegistry
        r = SeedRegistry.get_instance()
        r.set_master_seed(0)
        assert r.get_worker_seed(0) != r.get_worker_seed(1)
    def test_snapshot_restore(self):
        from src.utils.seed_registry import SeedRegistry
        r = SeedRegistry.get_instance()
        r.set_master_seed(99)
        s = r.snapshot_state()
        v1 = torch.randn(4)
        r.restore_state(s)
        v2 = torch.randn(4)
        assert torch.equal(v1, v2)