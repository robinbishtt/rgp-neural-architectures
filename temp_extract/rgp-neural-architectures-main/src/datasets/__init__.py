"""
src/datasets — Hierarchical dataset collection.

All dataset classes produce (input, label, hierarchy_depth) tuples.
The hierarchy_depth label is the key distinguishing feature — it encodes
the level in the semantic tree where a given sample was drawn, which is
exactly the quantity that H3 (multiscale generalization) is tested on.
"""
from src.datasets.hierarchical_dataset  import (
    HierarchicalGaussianDataset,
    IIDGaussianDataset,
    DatasetConfig,
)
from src.datasets.hierarchical_mnist    import HierarchicalMNIST
from src.datasets.hierarchical_cifar    import HierarchicalCIFAR
from src.datasets.synthetic_hierarchy   import SyntheticHierarchy
from src.datasets.imagenet_hierarchy    import ImageNetHierarchy
from src.datasets.medical_hierarchy     import MedicalHierarchy

__all__ = [
    "HierarchicalGaussianDataset",
    "IIDGaussianDataset",
    "DatasetConfig",
    "HierarchicalMNIST",
    "HierarchicalCIFAR",
    "SyntheticHierarchy",
    "ImageNetHierarchy",
    "MedicalHierarchy",
]
