# Reproducibility

This document explains every mechanism the codebase uses to guarantee reproducible results. The goal is that any researcher who clones this repository and runs `make reproduce_all` on hardware meeting the minimum requirements should obtain the same numerical results as those reported in the paper, to within floating-point rounding on the same hardware architecture.

---

## The Reproducibility Stack

Reproducibility in this codebase is not a single feature but a stack of interlocking mechanisms. Each layer addresses a different failure mode.

| Layer | Mechanism | Failure Mode Addressed |
|---|---|---|
| Random seeds | `SeedRegistry` Singleton | Non-deterministic training from uncoordinated RNG calls |
| Deterministic algorithms | `torch.use_deterministic_algorithms(True)` | Non-deterministic CUDA operations (atomics, cuDNN) |
| Environment lock | `requirements.txt` exact pins | Library API changes between versions |
| Container | `Dockerfile` / `Singularity.def` | OS-level differences, CUDA toolchain changes |
| Data integrity | SHA-256 checksums via `DataAuditor` | Dataset corruption or generation parameter drift |
| Checkpoint state | `RNGStateSerializer` in every checkpoint | Interrupted runs resuming from different RNG state |

---

## Random Seed Management

Every random number generator in the codebase is seeded through a single point of control: `SeedRegistry` in `src/utils/seed_registry.py`.

`SeedRegistry` is a thread-safe Singleton. The first call to `SeedRegistry.get_instance()` creates the instance; all subsequent calls return the same object. On `set_master_seed(seed)`, it propagates the seed to all four RNG streams simultaneously: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, and `random.seed`. No other module in the codebase calls these functions directly.

DataLoader workers require special handling because each worker runs in a separate process and would otherwise share the same RNG state. `SeedRegistry.get_worker_seed(worker_id)` computes a deterministic per-worker seed as `hash(master_seed || worker_id) mod 2^32`. This value is passed to the `worker_init_fn` parameter of every DataLoader in the codebase.

For checkpoint resume, `SeedRegistry.snapshot_state()` captures all four RNG states as a serialisable dictionary. The `RNGStateSerializer` in `src/checkpoint/` calls this function and includes the result in every checkpoint. On resume, `SeedRegistry.restore_state(state_dict)` restores all four streams to their exact saved state, ensuring that training continues from precisely the same random trajectory it would have followed without interruption.

To verify that seed management is working correctly on your system:

```bash
bash scripts/validate_determinism.sh --seed 42 --n-trials 3
```

This script runs three independent forward passes with the same seed and asserts that the outputs are bit-exact across all three trials.

---

## Deterministic CUDA Operations

GPU operations are non-deterministic by default in PyTorch. Atomic operations in CUDA kernels can accumulate in arbitrary order, and cuDNN selects the fastest available algorithm at runtime, which may differ between runs. Two settings are required to eliminate this non-determinism.

First, `torch.use_deterministic_algorithms(True)` instructs PyTorch to use only operations with deterministic implementations, raising a `RuntimeError` if a non-deterministic operation is called. This is applied globally in `src/utils/determinism.py` via `apply_global_determinism()`, which is called by every training entry point.

Second, the `CUBLAS_WORKSPACE_CONFIG` environment variable must be set to `:4096:8` before any CUDA operations. This allocates a fixed-size workspace buffer for cuBLAS operations, preventing the use of workspace size as a source of non-determinism. In Docker and Singularity containers, this variable is set in the image definition. In bare-metal installations, `setup_environment.sh` adds it to the shell environment, and `apply_global_determinism()` also sets it programmatically via `os.environ`.

Note that deterministic mode has a performance cost, typically fifteen to twenty-five percent slower training due to the avoidance of non-deterministic fast paths. This cost is accepted unconditionally; reproducibility takes priority over speed.

---

## Pinned Dependencies

`requirements.txt` pins every dependency to an exact version using `==`. This prevents automatic upgrades from silently changing numerical behaviour. The versions chosen represent the tested combination used for all experiments in the paper.

Key version constraints and their rationale:

`torch==2.0.1` was the latest stable release at time of publication and is the version used for all reported experiments. `numpy==1.24.3` avoids the NumPy 2.0 API break. `scipy==1.11.1` provides the `curve_fit` version used in `ExponentialDecayFitter`. `sympy==1.12` provides the symbolic computation environment used in `src/proofs/`. `h5py==3.9.0` handles HDF5 storage for large Fisher matrices.

Reviewers who install via `pip install -r requirements.txt` will obtain the exact environment. Reviewers who use the provided Docker or Singularity image will obtain an even more controlled environment, including the OS and CUDA toolchain.

---

## Container-Based Reproducibility

The container approach addresses long-term reproducibility. A `requirements.txt` file guarantees the same Python packages, but it cannot control the operating system, the C standard library, the CUDA driver compatibility layer, or the cuDNN version. In two years, even installing the exact same Python packages on a different Ubuntu version may produce different floating-point results due to differences in the underlying BLAS implementation.

`containers/Dockerfile` builds from `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`. This image pins the Ubuntu version, CUDA version (11.8.0), and cuDNN version (8) alongside the Python packages. `containers/Singularity.def` provides the same guarantee for HPC environments where Docker is not available due to security policies.

Both container definitions install dependencies in a fixed order, avoid any `pip install --upgrade` calls, and include a `%test` / smoke test section that validates the container at build time. The container images, once built, are immutable: rebuilding from the same definition file produces an identical image.

---

## Data Integrity

Hierarchical datasets are generated programmatically with correlation structure that depends on the random seed, NumPy version, and platform. Different reviewers generating data independently on different machines could produce slightly different datasets, leading to irreproducible training results.

The `DataAuditor` class in `src/provenance/data_auditor.py` addresses this by computing SHA-256 checksums of all dataset files before training begins. `DataAuditor.compute_checksum(path)` computes the hash of a single file. For a directory, it hashes the sorted concatenation of all file hashes within it, ensuring sensitivity to both file content and the presence or absence of files. `DataAuditor.verify_checksum(path, expected)` raises `DataIntegrityError` if the computed hash does not match, which halts training before it can proceed with potentially corrupted data.

`src/provenance/master_hashes.py` contains the official SHA-256 checksums for every dataset used in the paper. This file is version-controlled and should not be modified without re-generating and re-verifying the datasets.

The full provenance workflow, called automatically before any training run begins, is as follows. The dataset is generated with the master seed from `SeedRegistry`. Its checksum is computed. The computed checksum is compared against the master hash from `master_hashes.py`. If the checksums match, training proceeds. If they do not match, `DataIntegrityError` is raised with the path and both hash values.

---

## Checkpoint Resume Reproducibility

When a long training run (L=1000, 80 GB VRAM, three days) is interrupted by a cluster timeout or hardware failure, resuming from a checkpoint must produce exactly the same subsequent training trajectory as an uninterrupted run would have.

This requires saving and restoring not only the model weights and optimizer state, but also the complete RNG state. A checkpoint directory produced by `CheckpointManager` contains five files. `model.pt` holds model weights via `state_dict()`. `optimizer.pt` holds optimizer state including moment estimates. `rng_state.pkl` holds all four RNG states captured by `SeedRegistry.snapshot_state()`. `metrics.json` holds training history. `config.yaml` holds the experiment configuration that produced the checkpoint.

On resume, the loader calls `SeedRegistry.restore_state(rng_state)` before any forward pass. The training loop then picks up at the exact batch that would have followed the interrupted one. The `TimeoutHandler` in `src/utils/error_handler.py` monitors remaining walltime and triggers a preemptive checkpoint save when the remaining time falls below a configurable buffer, preventing checkpoint loss at cluster timeouts.

---

## Verification Checklist

The following commands verify each layer of the reproducibility stack independently.

```bash
# Verify pipeline integrity (7 checks, < 60 seconds)
make verify_pipeline

# Verify bit-exact determinism across 3 independent runs
bash scripts/validate_determinism.sh --seed 42 --n-trials 3

# Verify data checksums match master_hashes.py
python3 -c "
from src.provenance.data_auditor import DataAuditor
from src.provenance.master_hashes import MASTER_HASHES
for name, expected_hash in MASTER_HASHES.items():
    DataAuditor.verify_checksum(f'data/{name}', expected_hash)
    print(f'OK: {name}')
"

# Verify hypothesis validation tests pass
bash scripts/validate_hypotheses.sh --fast-track

# Run complete validation suite
make validate
```

---

## Known Limitations

**Cross-architecture reproducibility.** Bit-exact reproducibility is guaranteed across runs on the same hardware architecture (e.g., two different RTX 3090 cards). Results may differ at the last significant figure when comparing between hardware generations (e.g., RTX 3090 vs. A100) due to differences in floating-point unit implementations even with identical software environments. The scientific conclusions of the paper are robust to these differences; the paper reports mean and standard deviation across five seeds.

**MPS (Apple Silicon).** The MPS backend does not support `torch.use_deterministic_algorithms(True)` as of PyTorch 2.0. On Apple Silicon, the codebase automatically disables strict determinism and falls back to best-effort reproducibility. Fast-track verification and all unit tests pass correctly on MPS; bit-exact multi-run reproducibility is not guaranteed on this backend.

**Distributed training.** Multi-GPU training with `DistributedDataParallel` uses gradient synchronization via NCCL all-reduce, which is non-deterministic due to floating-point associativity. Results from distributed training may differ at the last significant figure from single-GPU results. All reported experimental results were obtained with single-GPU training.
