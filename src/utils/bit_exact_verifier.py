from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union
import hashlib
import numpy as np
import torch
@dataclass
class BitExactReport:
    passed:          bool
    n_tensors_checked: int
    n_mismatches:    int
    mismatch_details: List[Dict[str, Any]]  = field(default_factory=list)
    max_abs_diff:    float                  = 0.0
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines  = [
            f"[{status}] BitExact Verification",
            f"  Tensors checked: {self.n_tensors_checked}",
            f"  Mismatches:      {self.n_mismatches}",
            f"  Max |diff|:      {self.max_abs_diff:.2e}",
        ]
        for d in self.mismatch_details[:5]:  
            lines.append(f"  → {d}")
        return "\n".join(lines)
class BitExactVerifier:
    def __init__(self, strict: bool = True) -> None:
        self.strict  = strict
        self._run_a: Dict[str, Any] = {}
        self._run_b: Dict[str, Any] = {}
    def register_run_a(self, key: str, value: Any) -> None:
        self._run_a[key] = self._to_comparable(value)
    def register_run_b(self, key: str, value: Any) -> None:
        self._run_b[key] = self._to_comparable(value)
    def verify(self) -> BitExactReport:
        keys       = set(self._run_a.keys()) | set(self._run_b.keys())
        mismatches = []
        max_diff   = 0.0
        for key in sorted(keys):
            if key not in self._run_a:
                mismatches.append({"key": key, "reason": "missing in run A"})
                continue
            if key not in self._run_b:
                mismatches.append({"key": key, "reason": "missing in run B"})
                continue
            a, b = self._run_a[key], self._run_b[key]
            if isinstance(a, np.ndarray):
                if self.strict:
                    match = np.array_equal(a, b)
                else:
                    match = bool(np.allclose(a, b, atol=1e-7))
                diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
                max_diff = max(max_diff, diff)
            elif isinstance(a, float):
                diff  = abs(a - b)
                match = (a == b) if self.strict else (diff < 1e-7)
                max_diff = max(max_diff, diff)
            else:
                match = (a == b)
            if not match:
                mismatches.append({
                    : key, "reason": "value mismatch",
                    : diff if isinstance(a, (np.ndarray, float)) else "N/A"
                })
        return BitExactReport(
            passed=(len(mismatches) == 0),
            n_tensors_checked=len(keys),
            n_mismatches=len(mismatches),
            mismatch_details=mismatches,
            max_abs_diff=max_diff,
        )
    @staticmethod
    def checkpoint_sha256(path: Union[str, Path]) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    @staticmethod
    def compare_state_dicts(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
    ) -> BitExactReport:
        verifier = BitExactVerifier(strict=True)
        for k, v in state_a.items():
            verifier.register_run_a(k, v.cpu().numpy())
        for k, v in state_b.items():
            verifier.register_run_b(k, v.cpu().numpy())
        return verifier.verify()
    @staticmethod
    def _to_comparable(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (int, float)):
            return float(value)
        return value