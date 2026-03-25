import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
def test_model_loss_decreases():
    model = nn.Linear(8, 2)
    opt   = torch.optim.SGD(model.parameters(), lr=0.1)
    x     = torch.randn(32, 8); y = torch.randint(0, 2, (32,))
    losses = []
    for _ in range(20):
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 1.5  