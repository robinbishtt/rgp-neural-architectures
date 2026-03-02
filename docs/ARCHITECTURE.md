# Architecture

This document describes the complete six-tier system design of the RGP Neural Architectures codebase and the rationale behind every structural decision. It is intended for contributors who want to understand how the 250-file repository is organized, and for reviewers who want to verify that every experimental claim has a corresponding implementation.

---

## Design Philosophy

The architecture follows three principles. First, **separation of concerns**: mathematical theory lives in one place, neural architectures in another, tests in a third, and configuration management in a fourth. No tier has responsibilities that belong to another. Second, **reproducibility by construction**: every tier that involves randomness, hardware, or data must go through the Infrastructure Cross-Layer rather than managing its own state. Third, **reviewer accessibility**: any person with a laptop and no GPU must be able to verify that the pipeline is functional within five minutes using the fast-track system.

---

## Six-Tier Overview

| Tier | Name | Location | Files | Purpose |
|:---:|---|---|:---:|---|
| 1 | Nervous System | `src/core/`, `src/rg_flow/`, `src/proofs/` | ~30 | Mathematical foundations and symbolic proofs |
| 2 | Engine Room | `src/architectures/`, `src/training/`, `src/scaling/` | ~30 | Neural architectures, training, and scaling analysis |
| 3 | Audit Bureau | `tests/` | 100+ | Comprehensive correctness and robustness verification |
| 4 | Command Center | `config/`, `src/datasets/`, `containers/` | ~40 | Configuration management, data, and containerization |
| 5 | Publication Machine | `experiments/`, `figures/`, `scripts/` | ~50 | Experimental pipelines, figure generation, automation |
| ICL | Infrastructure Cross-Layer | `src/utils/`, `src/telemetry/`, `src/checkpoint/`, `src/orchestration/`, `src/provenance/` | ~35 | Determinism, telemetry, checkpointing, device management, data provenance |

---

## Tier 1  Nervous System

**Location:** `src/core/`, `src/rg_flow/operators/`, `src/proofs/`

The Nervous System is the mathematical bedrock of the entire framework. Every theorem, lemma, and corollary cited in the paper has a direct implementation here. This tier is intentionally isolated from all other tiers: it has no knowledge of training loops, datasets, or configurations. Its only dependencies are PyTorch, NumPy, SciPy, and SymPy.

### Fisher Information Geometry  `src/core/fisher/`

The Fisher information metric G⁽ᵏ⁾ = Jₖᵀ G⁽ᵏ⁻¹⁾ Jₖ quantifies how the network's statistical geometry transforms layer by layer. Six classes cover every aspect of this computation.

`FisherMetric` is the primary class. It implements the layer-wise pushforward via autograd hooks, clips eigenvalues to enforce positive semi-definiteness, and returns the per-layer metric tensor. `FisherEigenvalueAnalyzer` performs spectral decomposition and produces the eigenvalue density used in H1 validation. `FisherConditionTracker` monitors the condition number κ(G⁽ᵏ⁾) across depth as a diagnostic for the critical initialization regime. `FisherEffectiveDimension` computes d_eff = (Tr G)² / Tr(G²) from the eigenvalue spectrum. `FisherMonteCarloEstimator` provides a sampling-based estimator for large networks where exact computation is memory-prohibitive. `FisherAnalyticCalculator` gives closed-form results for specific activation functions and weight distributions, serving as ground truth for unit tests.

### Jacobian Analysis  `src/core/jacobian/`

The cumulative Jacobian J⁽ᵏ⁾ = ∏ᵢ Jᵢ characterizes how perturbations to the input propagate through depth. Five computation strategies are provided because no single approach is optimal for all network sizes.

`AutogradJacobian` uses repeated backward passes  exact but O(N²) in memory for width N. `JVPJacobian` uses forward-mode automatic differentiation via `torch.func.jvp`, which is memory-efficient for computing Jacobian-vector products. `VJPJacobian` uses reverse-mode `vjp` for vector-Jacobian products. `FiniteDifferenceJacobian` provides a central-difference numerical implementation used as an independent verification reference in unit tests. `SymbolicJacobian` uses SymPy for exact symbolic computation of small networks, enabling analytical verification of the autograd implementations.

### Spectral Theory  `src/core/spectral/`

Random matrix theory predicts the eigenvalue distribution of Jacobians at initialization. Three classical distributions are implemented alongside empirical density estimation.

`MarchenkoPasturDistribution` models the bulk eigenvalue density of large rectangular random matrices. It implements the theoretical PDF, CDF, a Wishart matrix sampler for testing, and a Kolmogorov-Smirnov test against empirical spectra. `WignerSemicircleDistribution` covers symmetric (GOE/GUE) ensembles. `TracyWidomDistribution` models the statistics of the largest eigenvalue, providing edge fluctuation analysis. `LevelSpacingDistribution` computes nearest-neighbour spacings for universality class identification. `empirical_spectral_density` uses kernel density estimation to construct a smooth density from a finite set of eigenvalues.

### Correlation Functions  `src/core/correlation/`

`TwoPointCorrelation` implements the mean-field recursion for the correlation coefficient c⁽ᵏ⁾ between two inputs at depth k, propagating q₁₁, q₁₂, q₂₂ through the network. `chi1_gauss_hermite` computes χ₁ = σ_w² · E[φ'(z)²] via Gauss-Hermite quadrature for tanh, ReLU, and GELU activations. `critical_sigma_w2` finds the critical weight variance where χ₁ = 1 by bisection, placing the network at the edge of chaos.

`ExponentialDecayFitter` fits ξ(k) = ξ₀ · exp(−k / k_c) to measured correlation lengths using `scipy.optimize.curve_fit` with bounded parameters. `FisherSpectrumMethod` estimates ξ directly from the Fisher eigenvalue density via ξ = [∫ρ(λ)λ⁻¹dλ]^{−1/2}. `TransferMatrixMethod` uses the ratio of the two largest transfer matrix eigenvalues. `MaximumLikelihoodEstimator` fits in log-space for numerical stability and returns confidence intervals.

### Lyapunov Spectrum  `src/core/lyapunov/`

`StandardQRAlgorithm` implements the Benettin et al. method: forward-pass through the network, then periodic QR re-orthogonalization of the accumulated Jacobian to extract Lyapunov exponents. `AdaptiveQRAlgorithm` monitors the condition number of the accumulated matrix and adjusts the re-orthogonalization interval to maintain numerical stability. `ParallelQRAlgorithm` distributes layers across available devices for very deep networks. `detect_regime` classifies networks as ordered (max exponent < −0.01), critical (|max exponent| < 0.01), or chaotic (max exponent > 0.01). `kaplan_yorke_dimension` computes the attractor dimension from the ordered exponent spectrum.

### Symbolic Proofs  `src/proofs/`

Each file contains a self-contained SymPy derivation. `theorem1_fisher_transform.py` derives G⁽ᵏ⁾ = Jₖᵀ G⁽ᵏ⁻¹⁾ Jₖ symbolically. `theorem2_exponential_decay.py` shows that the mean-field recursion produces exponential decay of c⁽ᵏ⁾ near the critical point. `theorem3_depth_scaling.py` derives L_min ~ k_c · log(ξ_data / ξ_target) from the exponential decay law. `lemma_critical_init.py` verifies the critical initialization conditions σ_w² = 1/N and σ_b² → 0.

### RG Flow Operators  `src/rg_flow/operators/`

`StandardRGOperator` implements h⁽ᵏ⁾ = σ(Wₖ h⁽ᵏ⁻¹⁾ + bₖ) with critical weight initialization σ_w/√N. `ResidualRGOperator` adds a learnable skip connection and projection. `AttentionRGOperator` uses multi-head self-attention as the coarse-graining step, followed by LayerNorm and a feedforward block. `WaveletRGOperator` applies Haar-like decomposition: a low-pass and high-pass filter bank followed by subsampling. `LearnedRGOperator` uses a small hypernetwork (context encoder) to generate scale-dependent scale and shift parameters for adaptive coarse-graining.

---

## Tier 2  Engine Room

**Location:** `src/architectures/`, `src/training/`, `src/scaling/`

The Engine Room transforms the mathematical primitives of Tier 1 into trainable neural networks and provides the infrastructure to train them efficiently at any scale from L=10 to L=1000+.

### RG-Net Architecture  `src/architectures/rg_net/`

Six variants cover the full depth range required by the paper's experiments. `RGNetShallow` (L=10–50) is used for rapid hyperparameter exploration. `RGNetStandard` (L=100) is the primary architecture for main experiments. `RGNetDeep` (L=500) uses gradient checkpointing by default. `RGNetUltraDeep` (L=1000+) requires both gradient checkpointing and mixed precision. `RGNetVariableWidth` allows non-uniform width profiles for information bottleneck studies. `RGNetMultiScale` incorporates explicit multi-scale feature fusion for hierarchical representation experiments.

All variants inherit from a common `RGNetTemplate` base class that handles operator composition, critical initialization, and the optional computation of intermediate Fisher metrics and Jacobians required by H1 and H2 analyses.

### Training Infrastructure  `src/training/`

`Trainer` is the core single-GPU training loop. It integrates `CheckpointManager`, `TelemetryLogger`, `SeedRegistry`, and `DeviceManager` and handles early stopping, gradient clipping, and learning rate scheduling. `DistributedTrainer` extends it for multi-GPU training via `torch.nn.parallel.DistributedDataParallel` with NCCL backend. `MixedPrecisionTrainer` wraps the forward/backward pass in `torch.cuda.amp.autocast` and uses a `GradScaler`. `GradientCheckpointingTrainer` applies `torch.utils.checkpoint.checkpoint` at configurable intervals. `ProgressiveTrainer` implements layer-wise curriculum training, freezing lower layers while the upper layers converge.

The `optimizers/` subdirectory provides: `AdamVariants` (standard Adam, AMSGrad, AdaBound), `SGDMomentum` (Nesterov with cosine annealing), `SecondOrder` (L-BFGS and Fisher natural gradient approximation), and `LayerWiseOptimizers` (exponentially decaying per-layer learning rates).

### Scaling Analysis  `src/scaling/`

`FSSFitter` fits the finite-size scaling ansatz f(L, N) = N^{−β/ν} · g((L − L_c) · N^{1/ν}) to accuracy data across widths. `CriticalExponentExtractor` extracts ν, β, and γ from the best-fit parameters. `DataCollapseVerifier` quantifies collapse quality via a weighted χ² metric and rejects collapses with χ² > 2.0. `BootstrapConfidence` constructs confidence intervals on extracted exponents via stratified bootstrap resampling across widths and seeds.

---

## Tier 3  Audit Bureau

**Location:** `tests/`

The Audit Bureau is deliberately the largest tier by file count, following the principle that test code should exceed production code in volume for research software. Every function, theorem, and architectural claim has a corresponding test.

The eight subdirectories cover distinct validation concerns. `unit/` tests mathematical correctness of isolated classes against known analytical results. `integration/` tests that tiers compose correctly: data flows to model, model produces metrics, checkpoints round-trip exactly. `stability/` tests gradient behaviour  no vanishing, no explosion, consistent norms across layers. `scaling/` tests the quantitative claims of H1, H2, and H3 with explicit pass/fail thresholds (R² ≥ 0.95, AIC preference, Wilcoxon p < 0.01). `ablation/` removes components one at a time to verify each contributes to the claimed performance. `robustness/` applies Gaussian noise, FGSM/PGD adversarial attacks, input corruptions, label noise, and distribution shift. `spectral/` fits RMT distributions to empirical Jacobian spectra and applies KS tests. `validation/` verifies bit-exact determinism across seeds and machines, and runs the full H1/H2/H3 pipeline end-to-end.

The `conftest.py` at the root of `tests/` provides shared fixtures: a minimal fast-track model, deterministic data loaders, temporary checkpoint directories, and pre-configured `SeedRegistry` and `DeviceManager` instances.

---

## Tier 4  Command Center

**Location:** `config/`, `src/datasets/`, `containers/`

The Command Center manages configuration, data, and environment. It has no knowledge of training logic or figure generation; it only serves the other tiers.

### Configuration  `config/`

Configurations are organized by concern rather than by experiment. `config/experiments/` contains the base configurations for each hypothesis validation experiment and is the authoritative source of all hyperparameters reported in the paper. `config/fast_track/` provides override files that reduce all scale parameters to their minimum values  depth from 100 to 10, width from 512 to 64, epochs from 100 to 2, dataset size to 100 samples  enabling three-to-five-minute reviewer verification. `config/architectures/` defines model variants. `config/training/` defines optimizer, scheduler, and mixed-precision configurations. `config/scaling/` defines FSS study parameters. `config/telemetry/` selects logging backends.

Hydra's compose API enables hierarchical overrides at the command line: `python experiments/h1.../run.py +experiment=h1 depth=100,200,500` runs H1 for three depths with all other parameters inherited from the base configuration.

### Datasets  `src/datasets/`

All datasets generate hierarchical correlation structure so that H3 performance differences are meaningful. `HierarchicalDataset` is the abstract base class. `HierarchicalMNIST` and `HierarchicalCIFAR` add multi-scale correlation structure on top of standard benchmarks. `SyntheticHierarchy` generates fully synthetic data with exact, configurable correlation length, enabling controlled H1 and H2 validation. `MedicalHierarchy` uses anatomical hierarchy for real-world validation with a different domain.

The `loaders/` subdirectory provides `DeterministicDataLoader` (fixed seed, reproducible batch order), `DistributedDataLoader` (worker sharding for multi-GPU), `CachedDataLoader` (in-memory pre-loading for small datasets), and `StreamingDataLoader` (out-of-core processing for large datasets).

### Containerization  `containers/`

Containers solve the long-term reproducibility problem. A `requirements.txt` file is insufficient: PyTorch 2.0 may have breaking API changes relative to future versions, and CUDA 11.8 toolchain libraries differ from CUDA 12+. Containers freeze the complete OS, Python version, CUDA version, and all library versions into an immutable image.

`Dockerfile` builds the primary GPU image from `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`. `Dockerfile.cpu` builds a CPU-only image from `python:3.9.18-slim-bullseye` for reviewers without GPU hardware. `Singularity.def` is the HPC container definition, designed for non-root execution in SLURM and PBS environments. `docker-compose.yml` orchestrates four services: GPU training, CPU training, Jupyter Lab on port 8888, and TensorBoard on port 6006.

---

## Tier 5  Publication Machine

**Location:** `experiments/`, `figures/`, `scripts/`

The Publication Machine transforms experimental results into publication outputs. Every figure in the paper is reproducible by a single command. Every script has a corresponding fast-track variant.

### Experimental Pipelines  `experiments/`

Each hypothesis has its own subdirectory. `h1_scale_correspondence/` contains the H1 validation runner, which trains networks at critical initialization, extracts per-layer correlation lengths via all four estimation methods, and fits the exponential decay law. `h2_depth_scaling/` trains networks at varying depths, identifies L_min as the depth at which target accuracy is first achieved, and fits the logarithmic scaling law. `h3_multiscale_generalization/` trains RG-Net and all four baselines on hierarchical datasets and evaluates under correlation structure shift. Each subdirectory also contains an experiment-local figure generation script that delegates to `figures/manuscript/`.

### Figure Generation  `figures/`

`figures/manuscript/` generates Figures 1–5. Every script accepts a `--fast-track` flag that substitutes synthetic data, enabling pipeline verification without running experiments. `figures/extended_data/` generates the six Extended Data figures covering correlation length diagnostics, Jacobian spectral evolution, stability phase diagram, FSS collapse, Lyapunov spectrum, and perturbation growth analysis. `figures/styles/` contains `publication.mplstyle` (300 DPI, Arial 7pt, single 88 mm / double 180 mm column widths), `color_palette.py`, `font_config.py`, and `latex_preamble.tex`. `figures/generate_all.py` is the master orchestrator with `--list`, `--figures`, `--fast-track`, and `--results-root` CLI flags.

### Automation Scripts  `scripts/`

Fourteen scripts cover the complete automation surface. `verify_pipeline.py` is a programmatic smoke test checking seven pipeline properties (imports, device detection, seed registry, forward/backward pass, checkpoint round-trip, spectral computation, correlation length fitting) in under sixty seconds. The `reproduce_fast*.sh` family runs the fast-track pipeline for all three hypotheses. `validate_determinism.sh` runs N independent trials with the same seed and asserts bit-exact equality. `validate_hypotheses.sh` orchestrates all three hypothesis validation scripts and reports pass/fail with p-values. `run_full_validation.sh` runs the complete pytest suite across all eight test categories.

---

## Infrastructure Cross-Layer (ICL)

**Location:** `src/utils/`, `src/telemetry/`, `src/checkpoint/`, `src/orchestration/`, `src/provenance/`

The ICL provides services that every tier consumes. It is the only layer with no tier that depends on it in return  it is purely a provider.

### Determinism  `src/utils/seed_registry.py` and `determinism.py`

`SeedRegistry` is a thread-safe Singleton. On `set_master_seed(seed)`, it calls `torch.manual_seed`, `np.random.seed`, `random.seed`, and `torch.cuda.manual_seed_all` in sequence. `get_worker_seed(worker_id)` returns a deterministic per-worker seed computed as a hash of `master_seed + worker_id`, so DataLoader workers do not share state. `snapshot_state()` and `restore_state()` capture and restore all RNG states for checkpoint resume. `apply_global_determinism()` sets `torch.use_deterministic_algorithms(True)` and the CUBLAS workspace configuration environment variable.

### Device Management  `src/utils/device_manager.py`

`DeviceManager` selects the best available device in CUDA → MPS → CPU priority order. It exposes `to_device(tensor)`, `model_to_device(model)`, and `get_device_info()`. Notably, `model_to_device` wraps the model in `DataParallel` when multiple GPUs are available. No `.cuda()` call appears anywhere else in the codebase.

### Telemetry  `src/telemetry/`

`TelemetryLogger` is a multi-backend logger. On initialization it detects which backends are available (TensorBoard, WandB, MLflow) and falls back to JSONL file logging if none are installed. It exposes `log_scalar`, `log_histogram`, `log_fisher_metric` (stores as HDF5 for large matrices), `log_jacobian_spectrum`, and `log_checkpoint`. Storage backends live in separate files: `hdf5_storage.py`, `parquet_storage.py`, `jsonl_storage.py`.

### Checkpointing  `src/checkpoint/`

`CheckpointManager` orchestrates periodic saves. A checkpoint directory contains `model.pt`, `optimizer.pt`, `rng_state.pkl` (from `SeedRegistry.snapshot_state()`), `metrics.json`, and `config.yaml`. `AsyncCheckpointWriter` performs the actual file I/O on a background thread, reducing training interruption to the time required to copy tensors to CPU. `ModelStateSerializer`, `RNGStateSerializer`, and `MetricStateSerializer` handle their respective state domains independently, allowing partial restoration.

### Orchestration  `src/orchestration/`

`DAGExecutor` represents experiment pipelines as directed acyclic graphs of `PipelineStage` objects and executes them in topological order with dependency resolution. `HydraConfigManager` wraps Hydra's compose API and exposes a simple `load(config_path, overrides)` interface. `pipeline.py` defines the canonical stage graph for the full reproduction pipeline.

### Data Provenance  `src/provenance/`

`DataAuditor.compute_checksum(path)` computes SHA-256 of a file or, recursively, of all files in a directory tree (sorted for determinism). `verify_checksum(path, expected)` raises `DataIntegrityError` on mismatch, halting training before it can proceed with corrupted data. `generate_manifest(dir)` creates a JSON manifest of all files and their checksums. `master_hashes.py` is a version-controlled registry of the official SHA-256 checksums for all datasets used in the paper.

---

## Dependency Structure

Data and control flow through the tiers in a single direction. Tier 1 exposes mathematical primitives. Tier 2 uses Tier 1 primitives to build trainable models. Tier 3 imports from Tiers 1 and 2 to test them. Tier 4 provides configuration and data to Tier 2. Tier 5 imports from all other tiers to produce outputs. The ICL is imported by all tiers but imports nothing from them.

This unidirectional dependency structure means that Tier 1 can be tested and verified independently, that changes to training infrastructure (Tier 2) cannot break mathematical correctness (Tier 1), and that the test suite (Tier 3) has no circular dependencies.
 