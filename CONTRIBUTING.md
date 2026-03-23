# Contributing to RGP Neural Architectures

Thank you for considering a contribution. This document describes how to set up a development environment, what standards the codebase enforces, and how to submit changes correctly.

---

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

## Getting Started

Fork the repository on GitHub, then clone your fork locally.

```bash
git clone https://github.com/<your-username>/rgp-neural-architectures.git
cd rgp-neural-architectures
git remote add upstream https://anonymous.4open.science/r/rgp-neural-architectures-BB30
```

Create a dedicated branch for your change. Use descriptive names that reflect the scope of the work.

```bash
git checkout -b fix/correlation-length-estimator
git checkout -b feat/parallel-qr-algorithm
git checkout -b docs/reproducibility-guide
```

Install the development environment.

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

---

## Development Standards

### Code Style

This project uses `black` for formatting and `isort` for import ordering. Both are enforced automatically via the pre-commit hooks. Run them manually at any time.

```bash
make format       # black + isort
make lint         # flake8 + isort check
make typecheck    # mypy
```

All public functions and classes must have docstrings. The preferred format is NumPy style.

```python
def fit(self, xi_values: np.ndarray) -> CorrelationLengthResult:
    """
    Fit exponential decay ξ(k) = ξ₀ · exp(−k / k_c) to measured values.

    Parameters
    ----------
    xi_values : np.ndarray
        Per-layer correlation lengths of shape (n_layers,).

    Returns
    -------
    CorrelationLengthResult
        Fitted parameters, goodness-of-fit R², and criticality parameter χ₁.
    """
```

### Determinism

Every new module that involves randomness **must** obtain its seed from `SeedRegistry`, not from `random.seed()`, `np.random.seed()`, or `torch.manual_seed()` called directly.

```python
# Correct
from src.utils.seed_registry import SeedRegistry
rng = np.random.default_rng(SeedRegistry.get_instance().get_worker_seed(worker_id=0))

# Incorrect - breaks global determinism
np.random.seed(42)
```

### Device Placement

No `.cuda()`, `.cpu()`, or `device="cuda"` strings are permitted in source code. All tensor placement goes through `DeviceManager`.

```python
from src.utils.device_manager import DeviceManager
device = DeviceManager.get_instance().get_device()
tensor = tensor.to(device)
```

### Mathematical Correctness

Changes to `src/core/` must include or update the corresponding unit test in `tests/unit/`. If the change implements a theorem, lemma, or corollary from the paper, a corresponding symbolic verification should be added to `src/proofs/`.

---

## Testing

Run the full test suite before submitting a pull request.

```bash
make test_unit          # Must pass with zero failures
make test_integration   # Must pass with zero failures
make validate           # Must pass determinism checks
```

For changes to mathematical core modules, also run the spectral and stability tests.

```bash
make test_spectral
make test_stability
```

New tests should follow the existing pattern: one test file per class or module, named `test_<module_name>.py`, placed in the appropriate subdirectory of `tests/`.

---

## Submitting a Pull Request

Before opening a pull request, confirm that all of the following are true.

- `make lint` passes with zero warnings.
- `make test_unit` and `make test_integration` pass.
- `make validate` passes.
- New or modified code has docstrings.
- The `CHANGELOG.md` has an entry under `[Unreleased]` describing the change.
- No hardcoded random seeds, device strings, or branding appear in any file.

When you open the pull request, describe the problem being solved, the approach taken, and any trade-offs considered. Reference any related issues.

---

## Reporting Issues

Open a GitHub Issue with a minimal reproducible example. Include the Python version, PyTorch version, hardware configuration, and the exact error output. For correctness issues in the mathematical core, include the input values that produce the incorrect output.

---

## Questions

For questions about the theoretical framework or experimental design, open a Discussion on the GitHub repository.
 