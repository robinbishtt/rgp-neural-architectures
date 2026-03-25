from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import torch
@dataclass
class ValidationResult:
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    def add(self, name: str, ok: bool, msg: str = "") -> None:
        self.checks[name] = ok
        if not ok:
            self.passed = False
            self.messages.append(f"FAIL [{name}]: {msg}")
class FastTrackValidator:
    def validate_loss_trajectory(
        self,
        losses: List[float],
        max_nan_fraction: float = 0.0,
        must_decrease: bool = True,
    ) -> ValidationResult:
        result = ValidationResult(passed=True)
        nan_count = sum(1 for l in losses if not np.isfinite(l))
        nan_frac  = nan_count / max(len(losses), 1)
        result.add("no_nan_loss", nan_frac <= max_nan_fraction,
                   f"NaN fraction={nan_frac:.2%}")
        if must_decrease and len(losses) >= 2:
            decreased = losses[-1] < losses[0]
            result.add("loss_decreases", decreased,
                       f"start={losses[0]:.4f} end={losses[-1]:.4f}")
        return result
    def validate_model_output(
        self,
        output: torch.Tensor,
        expected_shape: Optional[tuple] = None,
    ) -> ValidationResult:
        result = ValidationResult(passed=True)
        result.add("no_nan_output", not torch.isnan(output).any().item(),
                   )
        result.add("no_inf_output", not torch.isinf(output).any().item(),
                   )
        if expected_shape is not None:
            result.add("correct_shape", tuple(output.shape) == expected_shape,
                       f"got {tuple(output.shape)}, expected {expected_shape}")
        return result
    def validate_correlation_length(
        self,
        xi_values: np.ndarray,
        min_r2: float = 0.90,
    ) -> ValidationResult:
        from scipy.optimize import curve_fit
        result = ValidationResult(passed=True)
        k = np.arange(len(xi_values))
        result.add("positive_xi", (xi_values > 0).all(),
                   )
        try:
            def _exp(k, xi0, kc): return xi0 * np.exp(-k / kc)
            popt, _ = curve_fit(_exp, k, xi_values, p0=[xi_values[0], 5.0],
                                bounds=([0, 0.1], [np.inf, np.inf]), maxfev=2000)
            residuals = xi_values - _exp(k, *popt)
            ss_res = (residuals ** 2).sum()
            ss_tot = ((xi_values - xi_values.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            result.add("r2_threshold", r2 >= min_r2, f"R²={r2:.3f} < {min_r2}")
        except Exception as exc:
            result.add("exponential_fit", False, str(exc))
        return result
    def validate_pipeline(self, results: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(passed=True)
        if "losses" in results:
            sub = self.validate_loss_trajectory(results["losses"])
            result.checks.update(sub.checks)
            result.messages.extend(sub.messages)
            if not sub.passed:
                result.passed = False
        if "xi_values" in results:
            sub = self.validate_correlation_length(
                np.array(results["xi_values"])
            )
            result.checks.update(sub.checks)
            result.messages.extend(sub.messages)
            if not sub.passed:
                result.passed = False
        return result