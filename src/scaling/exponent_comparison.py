from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import t as t_dist
@dataclass
class ExponentComparisonResult:
    exponent_name:  str
    measured:       float
    predicted:      float
    std_error:      float
    t_statistic:    float
    p_value:        float
    consistent:     bool      
    effect_size:    float     
class ExponentComparison:
    def compare(
        self,
        name:           str,
        measured:       float,
        std_error:      float,
        predicted:      float,
        n_measurements: int = 1,
    ) -> ExponentComparisonResult:
        t_stat = (measured - predicted) / (std_error + 1e-12)
        df     = max(n_measurements - 1, 1)
        p_val  = 2.0 * float(t_dist.sf(abs(t_stat), df=df))
        return ExponentComparisonResult(
            exponent_name=name,
            measured=measured,
            predicted=predicted,
            std_error=std_error,
            t_statistic=float(t_stat),
            p_value=p_val,
            consistent=(abs(t_stat) < 2.0),
            effect_size=float(abs(measured - predicted) / (std_error + 1e-12)),
        )
    def compare_all(
        self,
        measurements: Dict[str, Tuple[float, float]],
        predictions:  Dict[str, float],
        n_measurements: int = 1,
    ) -> List[ExponentComparisonResult]:
        results = []
        for name, (measured, se) in measurements.items():
            pred = predictions.get(name, 0.0)
            results.append(self.compare(name, measured, se, pred, n_measurements))
        return results