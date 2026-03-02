# Datasets

This document describes every dataset used in the RGP Neural Architectures experiments: how each dataset is generated, what correlation structure it carries, how provenance is verified, and where the data is expected to be located on disk.

---

## Design Rationale

All datasets in this codebase carry explicit **hierarchical correlation structure**. This is a deliberate design choice: to test H3 (multi-scale generalization), it must be the case that the data actually contains multi-scale correlations that an architecture with RG-inspired depth structure could in principle exploit. Standard MNIST and CIFAR-10, without modification, have correlation structure that is largely determined by the image formation process and does not systematically test whether an architecture can leverage scale separation. The hierarchical datasets defined here add controlled multi-scale correlation on top of existing benchmarks (or generate it entirely synthetically), enabling clean experimental comparisons.

---

## Dataset Overview

| Name | Class | Source | Correlation Control | H1 | H2 | H3 |
|---|---|---|:---:|:---:|:---:|:---:|
| Synthetic Hierarchy | `SyntheticHierarchy` | Generated | Exact | ✅ | ✅ | ✅ |
| Hierarchical MNIST | `HierarchicalMNIST` | MNIST + augmentation | Approximate |  |  | ✅ |
| Hierarchical CIFAR | `HierarchicalCIFAR` | CIFAR-10 + augmentation | Approximate |  |  | ✅ |
| Medical Hierarchy | `MedicalHierarchy` | External (see below) | Structural |  |  | ✅ |

---

## Synthetic Hierarchy  `src/datasets/synthetic_hierarchy.py`

`SyntheticHierarchy` is the primary dataset for H1 and H2 validation because it provides **exact, configurable** correlation structure. The data-generating process is as follows.

A target correlation length `xi_data` is specified. A covariance matrix is constructed as C[i,j] = exp(−|i−j| / xi_data), which defines an exponentially decaying correlation function. Input vectors of dimension `input_dim` are sampled from a zero-mean Gaussian with this covariance. Labels are assigned by a fixed two-layer teacher network whose weights are drawn once at dataset construction time and then frozen. The teacher uses `tanh` activation and critical initialization.

The complete dataset is generated in a single call with a seed from `SeedRegistry`. The seed is set before any NumPy or PyTorch calls in the generation function, ensuring that the same master seed always produces the same dataset. After generation, `DataAuditor.compute_checksum()` is called and the result is compared against `MASTER_HASHES["synthetic_hierarchy"]` in `src/provenance/master_hashes.py`.

**Configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `n_samples` | 50,000 | Total dataset size |
| `input_dim` | 128 | Input vector dimension |
| `xi_data` | 8.0 | Target correlation length |
| `n_classes` | 4 | Number of label classes |
| `teacher_depth` | 2 | Teacher network depth |
| `teacher_width` | 64 | Teacher network width |
| `seed` | from `SeedRegistry` | Generation seed |

**Fast-track override:** `n_samples=100`, `input_dim=32`. Generated data is functionally identical in structure but smaller for rapid pipeline verification.

---

## Hierarchical MNIST  `src/datasets/hierarchical_mnist.py`

`HierarchicalMNIST` extends standard MNIST by adding multi-scale spatial correlation via a structured noise augmentation. The augmentation process samples a hierarchy of Gaussian blobs at three scales  coarse (σ=16), medium (σ=8), and fine (σ=2)  and adds them to the original image with configurable scale weights. The three scales correspond to the three levels of the hierarchical label structure (digit family, digit identity, stroke style).

The dataset downloads and caches standard MNIST via `torchvision.datasets.MNIST`. The hierarchical augmentation is applied deterministically at construction time using the seed from `SeedRegistry`. Labels are the standard ten MNIST classes.

**Usage note:** `HierarchicalMNIST` is used only for H3 experiments. For H1 and H2, which require precise control over the input correlation length `xi_data`, use `SyntheticHierarchy` instead.

---

## Hierarchical CIFAR  `src/datasets/hierarchical_cifar.py`

`HierarchicalCIFAR` applies the same hierarchical augmentation strategy to CIFAR-10. Three spatial scales are used: coarse (σ=8), medium (σ=4), and fine (σ=1). The scale weights are set to emphasize the coarse and fine scales equally, with a moderate medium-scale contribution, producing a bimodal frequency spectrum that challenges architectures without multi-scale processing.

CIFAR-10 is downloaded and cached via `torchvision.datasets.CIFAR10`. Labels are the standard ten CIFAR-10 classes.

---

## Medical Hierarchy  `src/datasets/medical_hierarchy.py`

`MedicalHierarchy` provides a real-world validation dataset with **structural** rather than synthetic hierarchical organisation. The hierarchy arises naturally from anatomy: images contain structures at three scales (organ, tissue, cell). This dataset is used to test whether the performance advantage observed in H3 on synthetic data transfers to a domain where the hierarchy is not explicitly constructed.

**Data source:** This dataset is not bundled in the repository due to size and licensing constraints. The expected dataset must be downloaded separately and placed at `data/medical_hierarchy/`. The `MedicalHierarchy` class will raise a `FileNotFoundError` with instructions if the data directory is absent. Contact the dataset provider for access; details are included in the paper's data availability statement.

**Checksum verification:** Once the data is placed in `data/medical_hierarchy/`, its SHA-256 manifest is verified against `MASTER_HASHES["medical_hierarchy"]` at construction time.

---

## Data Loaders  `src/datasets/loaders/`

Four loader implementations are provided, each suited to a different use case.

`DeterministicDataLoader` wraps `torch.utils.data.DataLoader` with `worker_init_fn=SeedRegistry.get_instance().worker_init_fn` and a fixed generator seed. This guarantees that the same sequence of batches is produced in every training run with the same master seed. This is the default loader for all single-GPU experiments.

`DistributedDataLoader` uses `DistributedSampler` with `shuffle=True` and coordinates across workers via the rank and world size. Worker seeds are derived from `SeedRegistry.get_worker_seed(worker_id)` to maintain cross-worker determinism.

`CachedDataLoader` loads the entire dataset into CPU memory at construction time, then serves batches from the in-memory copy. This eliminates I/O overhead for small datasets (up to approximately 4 GB). Used by default for `SyntheticHierarchy` with `n_samples ≤ 50,000`.

`StreamingDataLoader` processes large datasets that do not fit in memory. It reads files in a fixed, deterministic order determined by the master seed and processes them in chunks. Used for `MedicalHierarchy` on machines with less than 32 GB RAM.

---

## Directory Structure

All data lives under `data/` at the project root. This directory is excluded from version control via `.gitignore`.

```
data/
├── synthetic_hierarchy/        Generated by SyntheticHierarchy.__init__()
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   └── manifest.json           SHA-256 checksums for all three splits
│
├── hierarchical_mnist/         Downloaded + augmented by HierarchicalMNIST
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   └── manifest.json
│
├── hierarchical_cifar/         Downloaded + augmented by HierarchicalCIFAR
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   └── manifest.json
│
└── medical_hierarchy/          External — must be downloaded separately
    ├── train/
    ├── val/
    ├── test/
    └── manifest.json
```

---

## Checksum Verification

All checksums are stored in `src/provenance/master_hashes.py`. To verify your local data matches the official datasets:

```python
from src.provenance.data_auditor import DataAuditor
from src.provenance.master_hashes import MASTER_HASHES

for name, expected in MASTER_HASHES.items():
    result = DataAuditor.verify_checksum(f"data/{name}", expected)
    print(f"{'OK' if result else 'MISMATCH'}: {name}")
```

If a checksum does not match, delete the dataset directory and regenerate it with the correct master seed (`SEED=42`). All datasets except `MedicalHierarchy` can be regenerated automatically by calling the respective class constructor.

---

## Generating All Datasets

All datasets (except `MedicalHierarchy`) can be generated in a single step:

```bash
python3 -c "
from src.utils.seed_registry import SeedRegistry
SeedRegistry.get_instance().set_master_seed(42)

from src.datasets.synthetic_hierarchy import SyntheticHierarchy
from src.datasets.hierarchical_mnist import HierarchicalMNIST
from src.datasets.hierarchical_cifar import HierarchicalCIFAR

SyntheticHierarchy(root='data/synthetic_hierarchy', generate=True)
HierarchicalMNIST(root='data/hierarchical_mnist', download=True)
HierarchicalCIFAR(root='data/hierarchical_cifar', download=True)
print('All datasets generated and verified.')
"
```

Estimated disk space: synthetic hierarchy 180 MB, hierarchical MNIST 90 MB, hierarchical CIFAR 620 MB.
 