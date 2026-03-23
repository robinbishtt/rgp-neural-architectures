# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

*Planned for next release.*

---

## [1.0.1]  2026-03-21

### Fixed (Code-Paper Consistency Audit)

**Critical fixes:**
- `src/core/fisher/fisher_metric.py`: Corrected Fisher metric transformation from pushforward `J G J^T` to **pullback `J^T G J`** (paper Definition 3.1, Theorem 1). The pullback formula is mathematically required for the metric contraction property.
- `src/proofs/theorem1_fisher_transform.py`: Updated numerical verification to test pullback formula.
- `tests/unit/test_fisher_correctness.py`: Updated all Fisher tests to verify pullback; added `test_pullback_contracts_metric()` directly testing Theorem 1.

**Hypothesis implementation fixes:**
- `experiments/h3_multiscale_generalization/run_h3_validation.py`: Added `_welch_ttest()` (Welch's t-test with Welch's degrees-of-freedom correction) as PRIMARY test; added `_cohens_d()` (Cohen's d effect size). Previous primary was Wilcoxon. Paper: p=0.006, Cohen d=1.8.
- `experiments/h3_multiscale_generalization/run_h3_validation.py`: Updated `ACCURACY_PROFILES` to paper Table 1 values (RG-Net, ResNet-50, DenseNet-121, Wavelet-CNN, Tensor-Net).
- `experiments/h2_depth_scaling/run_h2_validation.py`: Fixed `accuracy_threshold` from 0.85 → **0.95** (paper: "minimum depth achieving ≥95% accuracy").
- `experiments/h2_depth_scaling/statistical_analysis.py`: Fixed docstring: α = 0.98 ± **0.06** (was ±0.12).
- `config/experiments/h2_depth_scaling.yaml`: Updated `correlation_lengths` to paper values `[5.0, 15.0, 50.0, 100.0, 200.0]` (Hier-1..5).

**Critical initialization defaults:**
- All sigma_w defaults: 1.0 → **1.4**, sigma_b defaults: 0.05 → **0.3** (tanh critical point).
- Affected files: `src/architectures/rg_net/rg_net.py`, `src/rg_flow/operators/operators.py`, `config/architectures/rg_net.yaml`, `config/experiments/h1_scale_correspondence.yaml`, `config/experiments/h2_depth_scaling.yaml`.

**Architecture fixes:**
- Added `src/architectures/baselines/wavelet_baseline.py` (WaveletCNNBaseline - Haar wavelet decomposition).
- Added `src/architectures/baselines/tensor_net_baseline.py` (TensorNetBaseline - Tensor-Train factorization).
- Updated `config/architectures/baselines.yaml` with paper-matching baseline specifications.
- Updated `config/experiments/h3_multiscale_generalization.yaml` to use wavelet_cnn + tensor_net (replacing mlp + vgg as primary comparison).

**Infrastructure fixes:**
- Added `verify()` wrapper functions to all proof files (`theorem1`, `theorem2`, `theorem3`, `lemma_critical_init`) so `VerificationRunner` imports work correctly.
- Created `src/proofs/proof_utils.py` with `VerificationResult` and `ProofLogger` classes.
- Consolidated `MarchenkoPasturDistribution`: `spectral.py` now re-exports from canonical `marchenko_pastur.py`.
- Fixed `compute_from_model()` approximation note in `fisher_metric.py`.
- Fixed `CITATION.cff`: replaced placeholder URL with actual anonymous repo URL.

**New files added:**
- 18 Jupyter notebooks (`notebooks/`) covering every theorem, hypothesis, and experiment.
- 15 paper figures (`figures/out/fig1_*.png` to `fig15_*.png`).
- 8 ablation study scripts (`ablation/`).
- 3 GitHub Actions CI workflows (`.github/workflows/reproduce-fast.yml` new).
- 5 documentation files (`docs/API.md`, `docs/PAPER_CODE_CORRESPONDENCE.md`, `docs/DATASETS.md`, `docs/INSTALLATION.md`, updated `docs/REPRODUCIBILITY.md`).
- Updated `README.md` with figures, badges, and structure diagram.



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

[Unreleased]: https://anonymous.4open.science/r/rgp-neural-architectures-BB30
[1.0.0]: https://anonymous.4open.science/r/rgp-neural-architectures-BB30
 