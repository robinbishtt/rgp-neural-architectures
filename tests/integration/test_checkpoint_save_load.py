import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
import tempfile
def test_checkpoint_save_and_load():
    from src.checkpoint.checkpoint_manager import CheckpointManager
    with tempfile.TemporaryDirectory() as tmpdir:
        cm    = CheckpointManager(checkpoint_dir=tmpdir, save_every_n_steps=1)
        model = nn.Linear(4, 2)
        cm.save(step=1, model=model, metrics={"loss": 0.5})
        w_before = model.weight.data.clone()
        nn.init.zeros_(model.weight)
        cm.load(model=model)
        assert torch.allclose(model.weight.data, w_before, atol=1e-6)