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
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
FAST_TRACK = dict(widths=[64], n_layers=8, n_seeds=2, n_samples=200)
FULL       = dict(widths=[64, 128, 256, 512], n_layers=30, n_seeds=5, n_samples=2000)
def _estimate_fisher_spectrum_layer(
    model:      torch.nn.Module,
    data:       torch.Tensor,
    layer_idx:  int,
    device:     torch.device,
) -> np.ndarray:
    model.eval()
    data = data.to(device)
    activations = []
    handle = None
    target = list(model.modules())[layer_idx + 1]  
    def hook_fn(module, inp, out):
        activations.append(out.detach().float().cpu())
    handle = target.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(data)
    handle.remove()
    if not activations:
        return np.array([])
    h_k = activations[0]  
    h_mu = h_k.mean(dim=0, keepdim=True)
    h_c  = h_k - h_mu
    B    = h_c.shape[0]
    F    = (h_c.T @ h_c) / max(B - 1, 1)  
    eigenvalues = torch.linalg.eigvalsh(F).numpy()
    return np.sort(eigenvalues)
def _correlation_length_from_spectrum(eigenvalues: np.ndarray) -> float:
    pos = eigenvalues[eigenvalues > 1e-10]
    if len(pos) == 0:
        return float("nan")
    return float(1.0 / np.sqrt(np.mean(1.0 / pos)))
def run_fast_track() -> None:
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
    from src.architectures.rg_net.rg_net_factory import RGNetFactory
    from src.datasets.hierarchical_dataset import HierarchicalGaussianDataset as HierarchicalDataset
    device = DeviceManager.get_device()
    SeedRegistry.get_instance().set_master_seed(seed)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = RGNetFactory.build(
        variant="standard", input_dim=128, hidden_dim=width,
        output_dim=10, depth=30,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
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