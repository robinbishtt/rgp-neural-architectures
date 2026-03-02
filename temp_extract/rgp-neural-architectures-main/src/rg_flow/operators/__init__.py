"""
src/rg_flow/operators — RG Flow Operator registry.

Base operators (operators.py):
    StandardRGOperator   – tanh MLP with critical σ_w initialization
    ResidualRGOperator   – residual variant with skip connection

Advanced standalone operators (canonical implementations):
    AttentionRGOperator  – multi-head self-attention coarse-graining
    WaveletRGOperator    – Haar wavelet multi-resolution coarse-graining
    LearnedRGOperator    – data-adaptive hyper-network coarse-graining
"""
from src.rg_flow.operators.operators import (
    StandardRGOperator,
    ResidualRGOperator,
)
from src.rg_flow.operators.attention_rg_operator import AttentionRGOperator
from src.rg_flow.operators.wavelet_rg_operator   import WaveletRGOperator
from src.rg_flow.operators.learned_rg_operator   import LearnedRGOperator

__all__ = [
    "StandardRGOperator",
    "ResidualRGOperator",
    "AttentionRGOperator",
    "WaveletRGOperator",
    "LearnedRGOperator",
]
