"""src/proofs  Symbolic and numerical theorem verification."""

from src.proofs.theorem1_fisher_transform import run_all_verifications as verify_theorem1
from src.proofs.theorem2_exponential_decay import run_all_verifications as verify_theorem2
from src.proofs.theorem3_depth_scaling import run_all_verifications as verify_theorem3
from src.proofs.lemma_critical_init import run_all_verifications as verify_critical_init

__all__ = [
    "verify_theorem1", "verify_theorem2",
    "verify_theorem3", "verify_critical_init",
]
