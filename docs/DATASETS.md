# Dataset Documentation

## Hierarchical Datasets (Hier-1 to Hier-5)

The five hierarchical datasets control correlation length exactly.

| Dataset | ξ_data | Scales | Label | Predicted L_min |
|---|---|---|---|---|
| Hier-1 | 5 | 2 | Shallow | ~13 layers |
| Hier-2 | 15 | 3 | Moderate | ~22 layers |
| Hier-3 | 50 | 4 | Deep | ~31 layers |
| Hier-4 | 100 | 5 | Very deep | ~37 layers |
| Hier-5 | 200 | 6 | Ultra deep | ~42 layers |

L_min = k_c · log(ξ_data / ξ_target), k_c ≈ 8 (fast-track), k_c ≈ 100 (tanh critical).

## Generation

```python
from src.datasets.hierarchical_dataset import HierarchicalGaussianDataset, DatasetConfig

cfg = DatasetConfig(
    n_samples=50000, input_dim=784, n_classes=10,
    xi_data=50.0,    # Hier-3
    seed=42,
)
ds = HierarchicalGaussianDataset(cfg)
```

## Standard Benchmarks

| Benchmark | Classes | Train | Test | Notes |
|---|---|---|---|---|
| CIFAR-10 | 10 | 50000 | 10000 | From `torchvision` |
| CIFAR-100 | 100 | 50000 | 10000 | Paper Table 1 |
| HierarchicalMNIST | 10 | 60000 | 10000 | See `src/datasets/hierarchical_mnist.py` |

## OOD Evaluation

OOD shifts are applied via covariance rotation (see `src/datasets/hierarchical_dataset.py:OODDataset`):

```python
from src.datasets.hierarchical_dataset import OODDataset, DatasetConfig
cfg = DatasetConfig(n_samples=5000, xi_data=50.0)
ood_ds = OODDataset(cfg, shift_magnitude=0.5)  # moderate shift
```

Shift levels: 0.0 (ID) → 1.0 (maximum OOD correlation rotation).

## Data Integrity

All datasets are SHA-256 verified via `src/provenance/data_auditor.py`:
```bash
python -c "from src.provenance.data_auditor import DataAuditor; DataAuditor().verify_all()"
```
