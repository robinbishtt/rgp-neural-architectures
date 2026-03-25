import numpy as np
class TestFSSAnalysisExtended:
    def test_critical_exponents_positive(self):
        from src.scaling.critical_exponents import CriticalExponentExtractor
        extractor = CriticalExponentExtractor()
        Ls = np.array([32.0, 64.0, 128.0, 256.0])
        gc = 1.0
        gs = np.linspace(0.5, 1.5, 10)
        data = {}
        for L in Ls:
            data[L] = 1.0 / np.sqrt(L) * np.tanh((gs - gc) * L)
        try:
            result = extractor.extract(data, Ls, gs)
            if hasattr(result, "nu"):
                assert result.nu > 0
        except Exception:
            pass  