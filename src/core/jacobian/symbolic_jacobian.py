"""
src/core/jacobian/symbolic_jacobian.py

SymPy-based symbolic Jacobian computation for small networks.
Used to generate analytic reference values for unit test verification.
"""
from __future__ import annotations
import numpy as np

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class SymbolicJacobian:
    """
    Symbolic Jacobian via SymPy for analytic cross-verification.

    Intended for small networks (N ≤ 10) where exact symbolic computation
    is tractable. Use AutogradJacobian for production-scale networks.
    """

    def __init__(self) -> None:
        if not SYMPY_AVAILABLE:
            raise ImportError("sympy is required for SymbolicJacobian. pip install sympy")

    def compute(
        self,
        W: np.ndarray,
        b: np.ndarray,
        x_val: np.ndarray,
        activation: str = "tanh",
    ) -> np.ndarray:
        """
        Compute exact Jacobian of σ(Wx + b) w.r.t. x symbolically.

        Returns (N_out, N_in) array of float values evaluated at x_val.
        """
        n_out, n_in = W.shape
        x_sym = sp.Matrix([sp.Symbol(f"x{i}") for i in range(n_in)])
        W_sym = sp.Matrix(W.tolist())
        b_sym = sp.Matrix(b.tolist())

        pre = W_sym * x_sym + b_sym
        act_map = {"tanh": sp.tanh, "relu": lambda z: sp.Piecewise((z, z > 0), (0, True))}
        act = act_map.get(activation, sp.tanh)
        h   = pre.applyfunc(act)

        J_sym = h.jacobian(x_sym)
        subs  = {x_sym[i]: float(x_val[i]) for i in range(n_in)}
        J_num = np.array(J_sym.subs(subs).tolist(), dtype=float)
        return J_num

    def verify_autograd(
        self,
        J_symbolic: np.ndarray,
        J_autograd: np.ndarray,
        rtol: float = 1e-5,
    ) -> bool:
        """Return True if symbolic and autograd Jacobians agree within rtol."""
        return bool(np.allclose(J_symbolic, J_autograd, rtol=rtol, atol=1e-7))
 