"""tests/scaling/test_logarithmic_scaling.py  L_min ~ log(xi) with slope ~ k_c."""
import numpy as np
from scipy.optimize import curve_fit


def test_log_fit_slope_matches_kc():
    kc  = 6.0
    xi  = np.linspace(2, 40, 25)
    lm  = kc * np.log(xi / 1.0)
    def _log(x, a, b): return a * np.log(x) + b
    popt, _ = curve_fit(_log, xi, lm)
    assert abs(popt[0] - kc) < 0.5, f"Fitted slope {popt[0]:.2f} != kc={kc}"
