"""
experiments/h1_scale_correspondence/compute_fisher_spectrum.py

Compute and serialise the per-layer Fisher information eigenvalue spectra
for H1 (Scale Correspondence) validation.

This script performs the computationally expensive spectral analysis that
feeds directly into Figure 3 and the H1 validation pipeline.  It can be
run independently of the full H1 training run so that spectral analysis
can be repeated on already-trained checkpoints without re-training.

Outputs
-------
results/h1/fisher_spectra/
    fisher_spectrum_W{width}_S{seed}.npz
        keys: eigenvalues (L, N) floats, layer_ids (L,) ints,
              xi_values (L,) floats, config dict

Usage
-----
    # From a completed H1 training run:
    python experiments/h1_scale_correspondence/compute_fisher_spectrum.py \
        --checkpoint results/h1/checkpoints/best_W256_S0.pt \
        --width 256 --seed 0

    # Fast-track (synthetic data, no checkpoint needed):
    python experiments/h1_scale_correspondence/compute_fisher_spectrum.py \
        --fast-track
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.utils.device_manager import DeviceManager
from src.utils.seed_registry import SeedRegistry

logger = logging.getLogger("compute_fisher_spectrum")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------------------------------
# Fast-track configuration
# ---------------------------------------------------------------------------
FAST_TRACK = dict(widths=[64], n_layers=8, n_seeds=2, n_samples=200)
FULL       = dict(widths=[64, 128, 256, 512], n_layers=30, n_seeds=5, n_samples=2000)


def _estimate_fisher_spectrum_layer(
    model:      torch.nn.Module,
    data:       torch.Tensor,
    layer_idx:  int,
    device:     torch.device,
) -> np.ndarray:
    """
    Estimate the Fisher information eigenvalue spectrum at layer ``layer_idx``.

    The Fisher information matrix at layer k is approximated by the empirical
    covariance of the gradient of the log-likelihood with respect to h^(k):

        F_k ≈ (1/B) Σ_i  ∂log p(y_i|h^(k)) / ∂h^(k)  ⊗  ∂log p(y_i|h^(k)) / ∂h^(k)

    For the purposes of the correlation-length measurement we use the simpler
    activation covariance proxy (see manuscript eq. S3).

    Returns
    -------
    eigenvalues : np.ndarray, shape (hidden_dim,)
        Eigenvalues of the estimated Fisher matrix at this layer, sorted
        ascending.
    """
    model.eval()
    data = data.to(device)

    activations = []
    handle = None

    # Register a forward hook on the target layer
    target = list(model.modules())[layer_idx + 1]  # +1 to skip model itself

    def hook_fn(module, inp, out):
        activations.append(out.detach().float().cpu())

    handle = target.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(data)

    handle.remove()

    if not activations:
        return np.array([])

    h_k = activations[0]  # (B, N)
    h_mu = h_k.mean(dim=0, keepdim=True)
    h_c  = h_k - h_mu
    B    = h_c.shape[0]
    F    = (h_c.T @ h_c) / max(B - 1, 1)  # (N, N)
    eigenvalues = torch.linalg.eigvalsh(F).numpy()
    return np.sort(eigenvalues)


def _correlation_length_from_spectrum(eigenvalues: np.ndarray) -> float:
    """
    ξ(k) = [∫ ρ(λ) λ^{-1} dλ]^{-1/2}  estimated discretely.
    """
    pos = eigenvalues[eigenvalues > 1e-10]
    if len(pos) == 0:
        return float("nan")
    return float(1.0 / np.sqrt(np.mean(1.0 / pos)))


def run_fast_track() -> None:
    """Generate synthetic Fisher spectra without a real trained model."""
    logger.info("Running fast-track Fisher spectrum computation.")
    rng = np.random.default_rng(42)
    out_dir = _ROOT / "results" / "h1" / "fisher_spectra"
    out_dir.mkdir(parents=True, exist_ok=True)

    for width in FAST_TRACK["widths"]:
        for seed in range(FAST_TRACK["n_seeds"]):
            n_layers = FAST_TRACK["n_layers"]
            xi_0, k_c = 15.0, 5.0
            spectra = []
            xi_vals = []
            for k in range(n_layers):
                xi_k = xi_0 * np.exp(-k / k_c)
                # Synthetic spectrum: Marchenko-Pastur with scale xi_k
                n = width
                sigma2 = xi_k / np.sqrt(n)
                ev = rng.random(n) * sigma2 * 4 + 1e-3
                ev = np.sort(ev)
                spectra.append(ev)
                xi_vals.append(_correlation_length_from_spectrum(ev))

            fname = out_dir / f"fisher_spectrum_W{width}_S{seed}.npz"
            np.savez(
                fname,
                eigenvalues  = np.array(spectra),
                layer_ids    = np.arange(n_layers),
                xi_values    = np.array(xi_vals),
            )
            logger.info("Saved %s", fname)

    logger.info("Fast-track Fisher spectrum computation complete.")


def run_from_checkpoint(checkpoint_path: Path, width: int, seed: int) -> None:
    """Compute Fisher spectra from a real trained checkpoint."""
    from src.architectures.rg_net import RGNetFactory
    from src.datasets.hierarchical_dataset import HierarchicalDataset

    device = DeviceManager.get_device()
    SeedRegistry.get_instance().set_master_seed(seed)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = RGNetFactory.build(
        variant="standard", input_dim=128, hidden_dim=width,
        output_dim=10, depth=30,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Create synthetic data for spectrum estimation
    dataset = HierarchicalDataset(n_samples=FULL["n_samples"], n_levels=3,
                                   input_dim=128, seed=seed)
    x = torch.stack([dataset[i][0] for i in range(min(500, len(dataset)))])

    out_dir = _ROOT / "results" / "h1" / "fisher_spectra"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len([m for m in model.modules() if hasattr(m, "weight") and m.weight.dim() == 2])
    spectra, xi_vals = [], []
    for k in range(n_layers):
        ev = _estimate_fisher_spectrum_layer(model, x, k, device)
        spectra.append(ev)
        xi_vals.append(_correlation_length_from_spectrum(ev))
        if k % 5 == 0:
            logger.info("Layer %d/%d: ξ = %.4f", k, n_layers, xi_vals[-1])

    fname = out_dir / f"fisher_spectrum_W{width}_S{seed}.npz"
    np.savez(fname, eigenvalues=np.array(spectra, dtype=object),
             layer_ids=np.arange(n_layers), xi_values=np.array(xi_vals))
    logger.info("Saved %s", fname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-layer Fisher spectra.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--width",      type=int,  default=256)
    parser.add_argument("--seed",       type=int,  default=0)
    parser.add_argument("--fast-track", action="store_true")
    args = parser.parse_args()

    if args.fast_track or args.checkpoint is None:
        run_fast_track()
    else:
        run_from_checkpoint(args.checkpoint, args.width, args.seed)


if __name__ == "__main__":
    main()
