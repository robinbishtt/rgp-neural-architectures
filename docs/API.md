# RGP Neural Architectures — API Reference

## Overview

This document provides the public API reference for the `rgp_neural_architectures` package. All public classes and functions are organized by module tier as described in the system architecture document.

---

## Tier 1: Core Mathematics (`src/core/`)

### `src/core/fisher/`

| Class | Description |
|---|---|
| `FisherMetricBase` | Abstract base class for all Fisher metric estimators |
| `FisherMetric` | Layer-wise metric computation via g^(k) = J_k^T g^(k-1) J_k |
| `FisherEigenvalueAnalyzer` | Spectral decomposition and eigenvalue density |
| `FisherConditionTracker` | Condition number monitoring across depth |
| `FisherEffectiveDimension` | Effective dimension from eigenvalue spectrum |
| `FisherMonteCarloEstimator` | Sampling-based estimation for large networks |
| `FisherAnalyticCalculator` | Closed-form calculations for specific architectures |

### `src/core/jacobian/`

| Class | Description |
|---|---|
| `AutogradJacobian` | Full Jacobian via repeated backward passes |
| `JVPJacobian` | Forward-mode Jacobian via Jacobian-Vector Products |
| `VJPJacobian` | Reverse-mode Jacobian via Vector-Jacobian Products |
| `FiniteDifferenceJacobian` | Numerical Jacobian for verification (central differences) |
| `SymbolicJacobian` | SymPy-based symbolic Jacobian for small networks |

### `src/core/spectral/`

| Class | Description |
|---|---|
| `MarchenkoPasturDistribution` | MP law for Wishart matrix eigenvalue density |
| `WignerSemicircleDistribution` | GOE/GUE universality class distribution |
| `TracyWidomDistribution` | Edge fluctuation statistics at spectral bulk boundary |
| `LevelSpacingDistribution` | Nearest-neighbor level spacing analysis |

### `src/core/correlation/`

| Class | Description |
|---|---|
| `FisherSpectrumMethod` | ξ(k) via Fisher spectral integral |
| `ExponentialDecayFitter` | Fits ξ(k) = ξ_0 exp(−k/k_c) |
| `TransferMatrixMethod` | ξ from singular value ratios of layer weight matrices |
| `MaximumLikelihoodEstimator` | MLE with confidence intervals |

### `src/core/lyapunov/`

| Class | Description |
|---|---|
| `StandardQRAlgorithm` | Benettin et al. periodic QR re-orthogonalization |
| `AdaptiveQRAlgorithm` | Dynamic QR triggering based on condition number |
| `ParallelQRAlgorithm` | Distributed computation for ultra-deep networks |

---

## Tier 2: Engine Room (`src/architectures/`, `src/training/`, `src/scaling/`)

### `src/architectures/rg_net/`

| Class | Constructor Key Arguments | Description |
|---|---|---|
| `RGNetShallow` | `input_dim, n_classes` | L=10–50 prototype |
| `RGNetStandard` | `input_dim, n_classes` | L=100 main configuration |
| `RGNetDeep` | `input_dim, n_classes` | L=500 scaling studies |
| `RGNetUltraDeep` | `input_dim, n_classes` | L=1000+ with memory optimization |
| `RGNetVariableWidth` | `input_dim, n_classes, width_schedule` | Non-uniform width |
| `RGNetMultiScale` | `input_dim, n_classes, n_scales` | Explicit multi-scale fusion |

### `src/training/`

| Class | Description |
|---|---|
| `Trainer` | Main training loop with checkpointing and logging |
| `DistributedTrainer` | Multi-GPU synchronous distributed training |
| `MixedPrecisionTrainer` | FP16/BF16 training with FP32 master weights |
| `GradientCheckpointingTrainer` | Memory-efficient ultra-deep training |
| `ProgressiveTrainer` | Curriculum learning with layer-wise training |
| `CurriculumTrainer` | Data difficulty-based curriculum learning |

### `src/training/optimizers/`

| Class | Description |
|---|---|
| `AdamVariants` | Adam, AMSGrad, AdaBound |
| `SGDMomentum` | Nesterov momentum with adaptive scheduling |
| `SecondOrder` | L-BFGS and natural gradient approximations |
| `LayerWiseOptimizers` | Per-layer learning rates |
| `DiagonalNaturalGradient` | Diagonal Fisher-preconditioned optimizer |
| `FisherOptimizer` | Kronecker-factored approximate curvature (K-FAC) |
| `LearningRateFinder` | Smith (2017) LR range test |
| `LinearWarmupScheduler` | Linear warmup with configurable decay |
| `CosineAnnealingWithRestarts` | SGDR with warm restarts |

### `src/scaling/`

| Class | Description |
|---|---|
| `FSSFitter` | Fits FSS scaling ansatz to finite-size data |
| `CriticalExponentExtractor` | Extracts ν, β, γ from scaling collapse |
| `DataCollapseVerifier` | Tests collapse quality with statistical metrics |
| `BootstrapConfidence` | Bootstrap confidence intervals for exponents |
| `ScalingLawFitter` | Logarithmic, power-law, and linear scaling laws |
| `WidthScalingAnalyzer` | Observable dependence on network width N |
| `DepthWidthAnalyzer` | Joint (L, N) phase surface analysis |
| `PhaseDiagramMapper` | σ_w × σ_b phase diagram via mean-field theory |
| `CollapseQualityMetrics` | Chi-squared and Q-value collapse assessment |
| `ExponentComparison` | Statistical tests against theoretical predictions |
| `SpectralScalingAnalyzer` | Jacobian spectral statistics per layer |

---

## ICL: Infrastructure Cross-Layer (`src/utils/`, `src/telemetry/`, `src/checkpoint/`, `src/orchestration/`, `src/provenance/`)

### `src/utils/`

| Class / Function | Description |
|---|---|
| `SeedRegistry.get_instance()` | Singleton RNG manager |
| `SeedRegistry.set_master_seed(seed)` | Propagate seed to all RNGs |
| `SeedRegistry.snapshot_state()` | Capture full RNG state |
| `SeedRegistry.restore_state(state)` | Restore RNG for exact reproduction |
| `DeviceManager.get_device()` | Auto-detect best device (CUDA/MPS/CPU) |
| `DeviceManager.to_device(tensor)` | Move tensor to best device |
| `BitExactVerifier` | Bit-exact comparison of two training runs |
| `DeterminismAuditor` | Detect unsanctioned direct RNG access |
| `FastTrackValidator` | Validate fast-track pipeline outputs |

### `src/provenance/`

| Class / Method | Description |
|---|---|
| `DataAuditor.compute_checksum(path)` | SHA-256 of dataset |
| `DataAuditor.verify_checksum(path, expected)` | Raises on mismatch |
| `DataAuditor.generate_manifest(dir)` | Create manifest.json |
| `DataAuditor.verify_manifest(path)` | Verify all files in manifest |

---

## Quick-Reference: Key Entry Points

```python
# 1. Set global seed (must be called first)
from src.utils.seed_registry import SeedRegistry
SeedRegistry.get_instance().set_master_seed(42)

# 2. Auto-detect device
from src.utils.device_manager import DeviceManager
device = DeviceManager().get_device()

# 3. Build and train an RG-Net
from src.architectures.rg_net.rg_net_standard import RGNetStandard
from src.training.trainer import Trainer
model = RGNetStandard(input_dim=784, n_classes=10)

# 4. Verify data integrity before training
from src.provenance.data_auditor import DataAuditor
DataAuditor().verify_checksum("data/hierarchical_mnist.pt", expected_hash="<sha256>")

# 5. Run fast-track verification
# make reproduce_fast      (3–5 minutes)
# make verify_pipeline     (<1 minute smoke test)
```
