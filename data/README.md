# Synthetic Hierarchical Datasets

Small synthetic datasets for fast-track validation (N=500 each).

## Files

| File | xi_data | Samples | Classes | Use |
|---|---|---|---|---|
| hier1_xi5_n500.npz | 5 | 500 | 4 | Hier-1 (shallow) |
| hier2_xi15_n500.npz | 15 | 500 | 4 | Hier-2 |
| hier3_xi50_n500.npz | 50 | 500 | 4 | Hier-3 (main) |
| hier4_xi100_n500.npz | 100 | 500 | 4 | Hier-4 |
| hier5_xi200_n500.npz | 200 | 500 | 4 | Hier-5 (deep) |

## Format

Each `.npz` file contains:
- `X`: float32 array (n_samples, 64) - input features
- `y`: int64 array (n_samples,) - class labels
- `xi_data`: float - correlation length
- `n_classes`: int - number of classes

## Loading

```python
import numpy as np
data = np.load("data/hier3_xi50_n500.npz")
X, y = data["X"], data["y"]
xi_data = float(data["xi_data"])
```

## Full Datasets

For full reproduction (N=50000, paper scale), generate via:
```bash
python -c "from src.datasets.hierarchical_dataset import HierarchicalGaussianDataset, DatasetConfig; ..."
```
See `docs/DATASETS.md` for full generation instructions.

## Notes

These files are for reviewer fast-track only.
Full paper experiments used N=50000 samples per dataset.
