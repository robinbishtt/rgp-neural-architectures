"""
src/provenance/master_hashes.py

Official SHA-256 checksums for all datasets used in the paper.

This file is version-controlled and represents the ground truth for data
integrity. Any dataset not matching these hashes is considered corrupted
or generated with different parameters.

HOW TO UPDATE:
    from src.provenance.data_auditor import DataAuditor
    h = DataAuditor().compute_checksum("data/hierarchical_mnist_xi5/")
    # Then update the entry below
"""
from __future__ import annotations
from typing import Dict


# Official dataset checksums (SHA-256 hex digests).
# Populated after data generation in the reference environment.
# Set to None for datasets generated locally (will skip verification).
MASTER_HASHES: Dict[str, str] = {
    # Synthetic hierarchy datasets
    "synthetic_xi5_n10000_seed42":     None,
    "synthetic_xi10_n10000_seed42":    None,
    "synthetic_xi20_n10000_seed42":    None,
    "synthetic_xi50_n10000_seed42":    None,

    # MNIST with correlation structure
    "hierarchical_mnist_xi5_train":    None,
    "hierarchical_mnist_xi5_test":     None,
    "hierarchical_mnist_xi10_train":   None,
    "hierarchical_mnist_xi10_test":    None,

    # CIFAR with hierarchy
    "hierarchical_cifar_train":        None,
    "hierarchical_cifar_test":         None,
}


def get_expected_hash(dataset_name: str) -> str | None:
    """Return expected SHA-256 hash for dataset_name, or None if not registered."""
    return MASTER_HASHES.get(dataset_name)


def is_registered(dataset_name: str) -> bool:
    return dataset_name in MASTER_HASHES
 