# Module Reference

This document provides an API reference for all public classes and functions in the `src/` directory. It is organized by tier and submodule, matching the directory structure of the repository.

---

## Tier 1 — Nervous System

### `src.core.fisher`

**`FisherMetric`**

Primary class for computing the layer-wise Fisher information metric.

```python
class FisherMetric:
    def pushforward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute G^(k) = J_k^T G^(k-1) J_k for each specified layer.

        Parameters
        ----------
        model : nn.Module
            The neural network. Must be an RG-Net variant or any module
            with named Linear or Conv layers.
        x : torch.Tensor
            Input batch of shape (B, input_dim).
        layer_indices : list of int, optional
            Which layers to compute the metric for. If None, computes for
            all layers.

        Returns
        -------
        dict mapping layer index to metric tensor of shape (N_k, N_k).
        """

    def compute_from_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        n_batches: int = 10,
    ) -> Dict[int, torch.Tensor]:
        """Estimate Fisher metric by averaging over multiple batches."""
```

**`FisherEigenvalueAnalyzer`**

```python
class FisherEigenvalueAnalyzer:
    def analyze(
        self,
        metric: torch.Tensor,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Decompose a Fisher metric tensor.

        Returns
        -------
        eigenvalues : np.ndarray
            Sorted eigenvalues in descending order.
        effective_dimension : float
            d_eff = (Tr G)^2 / Tr(G^2).
        condition_number : float
            Ratio of largest to smallest non-zero eigenvalue.
        """
```

**`FisherMonteCarloEstimator`**

```python
class FisherMonteCarloEstimator:
    def estimate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        n_samples: int = 1000,
    ) -> Dict[int, torch.Tensor]:
        """Sampling-based Fisher metric estimation via random projections."""
```

---

### `src.core.jacobian`

**`AutogradJacobian`**

```python
class AutogradJacobian:
    def compute(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute the Jacobian dh^(k) / dh^(0) using repeated backward passes.

        Memory complexity O(N^2) where N is the layer width.
        Exact for all activation functions supported by autograd.
        """

class CumulativeJacobian:
    def log_singular_values(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute log singular values of the cumulative Jacobian.
        Used for Lyapunov exponent estimation.
        """
```

**`JVPJacobian`**, **`VJPJacobian`**, **`FiniteDifferenceJacobian`**, **`SymbolicJacobian`**

All four classes expose the same `compute(model, x)` interface as `AutogradJacobian`. `FiniteDifferenceJacobian` additionally accepts an `eps` parameter (default `1e-5`) for the finite difference step size. `SymbolicJacobian` only supports networks with small integer weights expressible as SymPy symbols; it is used exclusively in unit tests.

---

### `src.core.spectral`

**`MarchenkoPasturDistribution`**

```python
class MarchenkoPasturDistribution:
    def __init__(self, beta: float, sigma2: float = 1.0):
        """
        Parameters
        ----------
        beta : float
            Aspect ratio N/M where N is the number of columns and M
            is the number of rows of the random matrix.
        sigma2 : float
            Variance of the matrix entries.
        """

    def pdf(self, lam: np.ndarray) -> np.ndarray: ...
    def cdf(self, lam: np.ndarray) -> np.ndarray: ...
    def ks_test(self, empirical: np.ndarray) -> Tuple[float, float]:
        """Returns (KS statistic, p-value)."""
    def sample_wishart(
        self,
        n: int,
        m: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample eigenvalues from a Wishart matrix for testing."""
```

`WignerSemicircleDistribution`, `TracyWidomDistribution`, and `LevelSpacingDistribution` expose the same `pdf`, `cdf`, and `ks_test` interface.

**`empirical_spectral_density`**

```python
def empirical_spectral_density(
    eigenvalues: np.ndarray,
    bandwidth: Optional[float] = None,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kernel density estimate of the empirical spectral density.

    Returns
    -------
    x : np.ndarray  — evaluation points
    density : np.ndarray  — estimated density at each point
    """
```

---

### `src.core.correlation`

**`TwoPointCorrelation`**

```python
class TwoPointCorrelation:
    def propagate(
        self,
        q11: float,
        q12: float,
        q22: float,
        sigma_w: float,
        sigma_b: float,
        activation: str = "tanh",
    ) -> Tuple[float, float, float]:
        """One step of the mean-field recursion. Returns (q11', q12', q22')."""

    def run(
        self,
        depth: int,
        sigma_w: float,
        sigma_b: float,
        activation: str = "tanh",
    ) -> np.ndarray:
        """
        Run the recursion for `depth` layers starting from
        q11=q22=1.0, q12=0.99.

        Returns
        -------
        np.ndarray of shape (depth,) — correlation coefficient c^(k) per layer.
        """
```

**`chi1_gauss_hermite`**, **`critical_sigma_w2`**

```python
def chi1_gauss_hermite(
    sigma_w: float,
    sigma_b: float,
    activation: str = "tanh",
    n_points: int = 100,
) -> float:
    """
    Compute χ₁ = σ_w² · E[φ'(z)²] via Gauss-Hermite quadrature.
    χ₁ = 1 defines the critical point (edge of chaos).
    """

def critical_sigma_w2(
    sigma_b: float = 0.0,
    activation: str = "tanh",
    tol: float = 1e-8,
) -> float:
    """Find σ_w² such that χ₁(σ_w², σ_b) = 1 by bisection."""
```

**`CorrelationLengthResult`**

```python
@dataclass
class CorrelationLengthResult:
    xi_0: float          # Pre-exponential factor
    k_c: float           # Decay constant
    r2: float            # Goodness-of-fit R²
    chi1: float          # χ₁ at the fitted parameters
    xi_0_ci: Tuple[float, float]   # 95% confidence interval on xi_0
    k_c_ci: Tuple[float, float]    # 95% confidence interval on k_c
```

`ExponentialDecayFitter`, `FisherSpectrumMethod`, `MaximumLikelihoodEstimator`, and `TransferMatrixMethod` all expose:

```python
def fit(self, xi_values: np.ndarray) -> CorrelationLengthResult: ...
# or
def estimate(self, eigenvalues: np.ndarray) -> CorrelationLengthResult: ...
```

---

### `src.core.lyapunov`

**`LyapunovResult`**

```python
@dataclass
class LyapunovResult:
    exponents: np.ndarray       # Full Lyapunov spectrum, descending order
    mle: float                  # Maximum Lyapunov exponent
    lyapunov_sum: float         # Sum of all exponents
    kaplan_yorke_dim: float     # Kaplan-Yorke attractor dimension
    regime: str                 # "ordered" | "critical" | "chaotic"
```

**`StandardQRAlgorithm`**

```python
class StandardQRAlgorithm:
    def __init__(
        self,
        reortho_interval: int = 10,
        n_warmup: int = 5,
    ): ...

    def compute(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        n_steps: int = 100,
    ) -> LyapunovResult: ...
```

`AdaptiveQRAlgorithm` and `ParallelQRAlgorithm` expose the same `compute` interface. `AdaptiveQRAlgorithm` additionally accepts `condition_threshold` (default `1e6`) to trigger early re-orthogonalization. `ParallelQRAlgorithm` distributes layer batches across available GPUs.

**`analyze_lyapunov`** — convenience function

```python
def analyze_lyapunov(
    model: nn.Module,
    dataloader: DataLoader,
    algorithm: str = "adaptive",
    n_steps: int = 100,
) -> LyapunovResult:
    """Selects and runs the appropriate QR algorithm. Use this in experiments."""
```

---

## Tier 2 — Engine Room

### `src.architectures.rg_net`

All six RG-Net variants accept the same constructor signature and expose the same interface.

```python
class RGNetStandard(nn.Module):
    def __init__(
        self,
        depth: int = 100,
        width: int = 512,
        input_dim: int = 128,
        n_classes: int = 4,
        operator_type: str = "standard",  # standard|residual|attention|wavelet|learned
        activation: str = "tanh",
        sigma_w: float = 1.0,
        sigma_b: float = 0.05,
        gradient_checkpointing: bool = False,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        return_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Returns (logits, {layer_idx: hidden_state})."""
```

### `src.training`

**`Trainer`**

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[DictConfig] = None,
    ): ...

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ) -> Dict[str, List[float]]:
        """Returns training history dict with keys: train_loss, val_loss, val_acc."""

    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...
```

`DistributedTrainer`, `MixedPrecisionTrainer`, `GradientCheckpointingTrainer`, and `ProgressiveTrainer` all extend `Trainer` and accept the same constructor. Each overrides `fit()` with the appropriate training strategy while preserving the same return interface.

### `src.scaling`

**`FSSFitter`**

```python
class FSSFitter:
    def fit(
        self,
        depths: np.ndarray,
        widths: np.ndarray,
        accuracies: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fit the FSS ansatz f(L, N) = N^(-β/ν) · g((L - L_c) · N^(1/ν)).

        Returns
        -------
        dict with keys: nu, beta, gamma, L_c, chi2, p_value
        """
```

---

## Infrastructure Cross-Layer

### `src.utils.seed_registry`

```python
class SeedRegistry:
    @classmethod
    def get_instance(cls) -> "SeedRegistry": ...

    def set_master_seed(self, seed: int) -> None: ...
    def get_worker_seed(self, worker_id: int) -> int: ...
    def snapshot_state(self) -> Dict[str, Any]: ...
    def restore_state(self, state_dict: Dict[str, Any]) -> None: ...
    def worker_init_fn(self, worker_id: int) -> None:
        """Pass as worker_init_fn to DataLoader."""
```

### `src.utils.device_manager`

```python
class DeviceManager:
    @classmethod
    def get_instance(cls) -> "DeviceManager": ...

    def get_device(self) -> torch.device: ...
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor: ...
    def model_to_device(self, model: nn.Module) -> nn.Module: ...
    def get_device_info(self) -> Dict[str, Any]: ...
    def available_memory_gb(self) -> float: ...
    def empty_cache(self) -> None: ...
```

### `src.utils.determinism`

```python
@dataclass
class DeterminismConfig:
    use_deterministic_algorithms: bool = True
    set_cublas_workspace: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

def apply_global_determinism(config: DeterminismConfig = DeterminismConfig()) -> None:
    """Apply all determinism settings. Call once at the start of every training script."""
```

### `src.telemetry.telemetry_logger`

```python
class TelemetryLogger:
    def __init__(
        self,
        backends: List[str],   # ["tensorboard", "wandb", "mlflow", "jsonl"]
        log_dir: str = "logs/",
        project_name: str = "rgp-neural",
    ): ...

    def log_scalar(self, name: str, value: float, step: int) -> None: ...
    def log_histogram(self, name: str, values: np.ndarray, step: int) -> None: ...
    def log_fisher_metric(self, layer_id: int, metric: torch.Tensor, step: int) -> None: ...
    def log_jacobian_spectrum(self, layer_id: int, svs: np.ndarray, step: int) -> None: ...
    def log_checkpoint(self, path: str, metadata: Dict) -> None: ...
    def close(self) -> None: ...
```

### `src.checkpoint.checkpoint_manager`

```python
class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        save_every_n_epochs: int = 10,
        keep_last_n: int = 3,
        async_write: bool = True,
    ): ...

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Any],
        config: Optional[DictConfig] = None,
    ) -> str:
        """Returns the path of the saved checkpoint directory."""

    def load(self, path: str, model: nn.Module, optimizer: Optional[Any] = None) -> Dict:
        """Loads checkpoint and returns metrics dict."""

    def latest(self) -> Optional[str]:
        """Returns path of the most recent checkpoint, or None."""
```

### `src.provenance.data_auditor`

```python
class DataAuditor:
    @staticmethod
    def compute_checksum(path: Union[str, Path]) -> str:
        """SHA-256 of a file, or of the sorted concatenation of all file hashes in a directory."""

    @staticmethod
    def verify_checksum(path: Union[str, Path], expected: str) -> bool:
        """Raises DataIntegrityError on mismatch. Returns True on match."""

    @staticmethod
    def generate_manifest(directory: Union[str, Path]) -> Dict[str, str]:
        """Returns {relative_filepath: sha256_hash} for all files in directory."""

    @staticmethod
    def save_manifest(directory: Union[str, Path], output_path: str) -> None: ...

    @staticmethod
    def verify_manifest(manifest_path: str) -> bool: ...
```

---

## Exceptions

```python
class DataIntegrityError(Exception):
    """Raised when a SHA-256 checksum does not match the expected value."""

class DeterminismViolationError(Exception):
    """Raised when a non-deterministic operation is called in strict mode."""
```
