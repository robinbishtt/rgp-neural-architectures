# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

*Planned for next release.*

---

## [1.0.0]  2026-02-28

Initial public release.

### Added

**Tier 1  Nervous System (Mathematical Core)**
- `src/core/fisher/`  Fisher information geometry: metric computation, eigenvalue analysis, condition tracking, effective dimension, Monte Carlo estimator, and closed-form analytic calculator.
- `src/core/jacobian/`  Jacobian computation via autograd, JVP (forward-mode), VJP (reverse-mode), finite differences, and SymPy symbolic derivation.
- `src/core/spectral/`  Random matrix theory distributions: Marchenko-Pastur, Wigner semicircle, Tracy-Widom, level spacing, and empirical spectral density via KDE.
- `src/core/correlation/`  Two-point correlation functions, χ₁ via Gauss-Hermite quadrature, critical σ_w² via bisection.
- `src/core/lyapunov/`  Lyapunov spectrum via standard QR, adaptive QR, and parallel QR algorithms. Includes regime detection and Kaplan-Yorke dimension.
- `src/proofs/`  Symbolic SymPy verification of Theorem 1 (Fisher metric transformation), Theorem 2 (exponential correlation decay), Theorem 3 (depth scaling law), and the critical initialization lemma.
- `src/rg_flow/operators/`  Standard, residual, attention-based, wavelet, and learned RG operators.

**Tier 2  Engine Room (Architectures and Training)**
- `src/architectures/rg_net/`  Six RG-Net variants: shallow (L=10–50), standard (L=100), deep (L=500), ultra-deep (L=1000+), variable-width, and multi-scale.
- `src/architectures/baselines/`  ResNet, DenseNet, MLP, and VGG baselines for H3 comparison.
- `src/training/`  Trainer, DistributedTrainer, MixedPrecisionTrainer, GradientCheckpointingTrainer, ProgressiveTrainer.
- `src/training/optimizers/`  Adam variants, SGD with Nesterov momentum, L-BFGS / natural gradient, and layer-wise learning rate optimizers.
- `src/scaling/`  Finite-size scaling fitter, critical exponent extractor, data collapse verifier, and bootstrap confidence intervals.

**Tier 3  Audit Bureau (Tests)**
- `tests/unit/`  8 unit test files covering Fisher symmetry, PSD property, Jacobian chain rule, Jacobian SVD, Lyapunov QR convergence, correlation decay, and MP spectral law.
- `tests/integration/`  6 integration test files covering the full data-to-model pipeline, model-to-metrics pipeline, training convergence, checkpoint round-trips, distributed training, and mixed-precision stability.
- `tests/stability/`  5 stability test files for vanishing gradients, exploding gradients, gradient norm consistency, critical initialization, and FP32/FP64 numerical precision.
- `tests/scaling/`  6 scaling test files for H1/H2/H3 quantitative validation, exponential decay R², logarithmic scaling, and FSS data collapse.
- `tests/ablation/`  6 ablation study files for RG operators, skip connections, scale-aware pooling, critical initialization, multi-scale fusion, and depth sweeps.
- `tests/robustness/`  5 robustness test files for Gaussian noise, FGSM/PGD adversarial attacks, input corruption, label noise, and distribution shift.
- `tests/spectral/`  5 spectral test files for Marchenko-Pastur fitting, Wigner semicircle, level spacing distribution, number variance, and Tracy-Widom edge statistics.
- `tests/validation/`  5 validation test files for seed-level determinism, H1/H2/H3 hypothesis validation, and full pipeline reproducibility.

**Tier 4  Command Center (Configuration and Data)**
- `config/experiments/`  8 YAML experiment configurations for H1, H2, H3, baseline comparison, FSS, ablation, robustness, and extended data reproduction.
- `config/fast_track/`  4 fast-track override configurations reducing all scale parameters for 3–5 minute reviewer verification.
- `config/architectures/`  RG-Net variant configs, baseline comparison configs, and operator ablation sweep configs.
- `config/training/`  Base training, deep network, distributed, fast-track, and optimizer configuration files.
- `config/scaling/`  FSS study parameters and critical exponent sweep configs.
- `config/telemetry/`  Backend selection for TensorBoard, WandB, MLflow, and JSONL.
- `src/datasets/`  HierarchicalDataset, HierarchicalMNIST, HierarchicalCIFAR, SyntheticHierarchy, and MedicalHierarchy.
- `src/datasets/loaders/`  DeterministicDataLoader, DistributedDataLoader, CachedDataLoader, and StreamingDataLoader.
- `containers/`  GPU Dockerfile, CPU Dockerfile, docker-compose.yml, Singularity.def, and singularity_build.sh.

**Tier 5  Publication Machine (Experiments and Figures)**
- `experiments/h1_scale_correspondence/`  H1 validation runner, correlation decay analyzer, statistical tests, and Figure 3 generator.
- `experiments/h2_depth_scaling/`  H2 validation runner, depth scaling analyzer, minimum depth extractor, and Figure 4 generator.
- `experiments/h3_multiscale_generalization/`  H3 validation runner, architecture comparator, OOD evaluator, and Figure 5 / Table 1 generator.
- `figures/manuscript/`  generate_figure1.py through generate_figure5.py with synthetic data fallback for fast-track mode.
- `figures/extended_data/`  run_extended_figure1.py through run_extended_figure6.py covering correlation diagnostics, Jacobian evolution, phase diagram, FSS collapse, Lyapunov diagnostics, and perturbation growth.
- `figures/styles/`  publication.mplstyle (300 DPI, Arial 7pt, single/double-column dimensions), color_palette.py, font_config.py, and latex_preamble.tex.
- `figures/generate_all.py`  Master figure pipeline with `--fast-track`, `--figures`, and `--list` CLI flags.
- `scripts/`  14 automation scripts: verify_pipeline.py/.sh, reproduce_fast.sh, reproduce_fast_h1/h2/h3.sh, validate_determinism.sh, validate_hypotheses.sh, run_full_validation.sh, reproduce_all_figures.sh, reproduce_extended_data.sh, reproduce_tables.sh, download_pretrained_checkpoints.sh, cleanup_artifacts.sh, setup_environment.sh.

**Infrastructure Cross-Layer (ICL)**
- `src/utils/seed_registry.py`  Thread-safe Singleton managing all RNG seeds. Prevents any module from seeding directly.
- `src/utils/device_manager.py`  Hardware-agnostic CUDA → MPS → CPU auto-detection. Eliminates hardcoded `.cuda()` calls.
- `src/utils/determinism.py`  Global determinism configuration applying `torch.use_deterministic_algorithms`.
- `src/utils/determinism_auditor.py`  Audits codebase for rogue `random.seed()` calls.
- `src/utils/error_handler.py`  OOM recovery (batch size halving), NaN recovery (checkpoint rollback, LR reduction), timeout handler for cluster walltime.
- `src/utils/fast_track_validator.py`  Validates fast-track outputs for NaN, shape correctness, and convergence.
- `src/utils/hardware_dispatch.py`  Detects compute capability and selects optimal dtype (FP32/FP16/BF16).
- `src/utils/memory_utils.py`  Memory estimation, guard context manager, gradient checkpointing helper.
- `src/utils/provenance.py`  SHA-256 file and directory checksums, manifest generation and verification.
- `src/telemetry/`  TelemetryLogger, HDF5 storage, Parquet storage, JSONL storage, and Slack/email notifiers.
- `src/checkpoint/`  CheckpointManager, ModelStateSerializer, RNGStateSerializer, MetricStateSerializer, AsyncCheckpointWriter.
- `src/orchestration/`  DAG executor, Hydra configuration manager, and experiment pipeline coordinator.
- `src/provenance/`  DataAuditor class and master_hashes.py registry.

**Root documentation**
- `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `LICENSE`.
- `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`, `docs/REPRODUCIBILITY.md`, `docs/DATASETS.md`, `docs/MODULES.md`.
- `Makefile` with 25+ targets covering setup, fast-track, full reproduction, figure generation, testing, linting, and cleanup.
- `requirements.txt` with all dependencies pinned to exact versions.
- `pyproject.toml` with package metadata, optional dependency groups, and tool configurations.
- `.gitignore` excluding results, checkpoints, logs, data, and all generated artifacts.

---

[Unreleased]: https://github.com/robinbishtt/rgp-neural-architectures/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/robinbishtt/rgp-neural-architectures/releases/tag/v1.0.0
 