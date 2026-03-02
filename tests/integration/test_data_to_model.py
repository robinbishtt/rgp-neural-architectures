"""tests/integration/test_data_to_model.py"""
import torch
import torch.nn as nn


def test_synthetic_data_loads_and_trains():
    from src.datasets.synthetic_hierarchy import SyntheticHierarchy
    from torch.utils.data import DataLoader
    ds     = SyntheticHierarchy(n_samples=64, n_features=32, seed=42)
    loader = DataLoader(ds, batch_size=16)
    model  = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 4))
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    for x, y in loader:
        logits = model(x)
        loss   = nn.CrossEntropyLoss()(logits, y)
        loss.backward(); opt.step(); opt.zero_grad()
        assert not torch.isnan(loss)
        break
 