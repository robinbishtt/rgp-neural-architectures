from __future__ import annotations
import argparse
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from src.utils.seed_registry import SeedRegistry
from src.utils.device_manager import DeviceManager
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("h1_validation")
FAST_TRACK = {
    :         [64, 128],
    :       10,           
    :        3,
    :        1000,         
    :         5,
    : 1.2,          
    :        0.3,
    :   0.80,         
}
FULL = {
    :         [64, 128, 256, 512],
    :       30,
    :        10,
    :        5000,         
    :         100,
    : 1.2,          
    :        0.3,
    :   0.95,         
}
class RGLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sigma_w: float, sigma_b: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act    = nn.Tanh()
        nn.init.normal_(self.linear.weight, std=sigma_w / math.sqrt(in_dim))
        nn.init.normal_(self.linear.bias,   std=sigma_b)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
class RGNetH1(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        sigma_w: float = 1.2,
        sigma_b: float = 0.3,
    ):
        super().__init__()
        self.depth      = depth
        self.hidden_dim = hidden_dim
        dims = [input_dim] + [hidden_dim] * depth
        self.rg_layers  = nn.ModuleList([
            RGLayer(dims[i], dims[i + 1], sigma_w, sigma_b)
            for i in range(depth)
        ])
        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.head.weight, std=sigma_w / math.sqrt(hidden_dim))
        nn.init.normal_(self.head.bias,   std=sigma_b)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.rg_layers:
            x = layer(x)
        return self.head(x)
def _load_or_generate_data(
    n_train: int,
    n_classes: int = 10,
    input_dim: int = 64,
    xi_data: float = 10.0,
    seed: int = 42,
    data_root: Path = None,
) -> tuple:
    if data_root is not None:
        for fname in [
            ,
            ,
            ,
            ,
            ,
        ]:
            fpath = data_root / fname
            if fpath.exists():
                d = np.load(str(fpath), allow_pickle=True)
                X = torch.from_numpy(d["X"].astype(np.float32))
                y = torch.from_numpy(d["y"].astype(np.int64))
                if len(X) >= n_train:
                    idx = np.random.default_rng(seed).permutation(len(X))[:n_train]
                    return X[idx], y[idx], float(d.get("xi_data", xi_data))
    rng      = np.random.default_rng(seed)
    noise_std = xi_data   
    class_centres = rng.standard_normal((n_classes, input_dim)) * xi_data
    sub_centres   = class_centres + rng.standard_normal((n_classes, input_dim)) * (xi_data / 2)
    X_list, y_list = [], []
    per_class = max(n_train // n_classes, 1)
    for cls in range(n_classes):
        x = sub_centres[cls] + rng.standard_normal((per_class, input_dim)) * noise_std
        X_list.append(x.astype(np.float32))
        y_list.extend([cls] * per_class)
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(y))
    return (
        torch.from_numpy(X[perm]),
        torch.from_numpy(y[perm]),
        xi_data,
    )
def _measure_xi_gradient_fisher(
    model: RGNetH1,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    B   = X.shape[0]
    n_k = model.hidden_dim
    gamma = n_k / max(B, 1)
    if gamma > 0.1:
        logger.warning(
            ,
            n_k, B, gamma, 100 * np.sqrt(2 / B), 100 * n_k, n_k,
        )
    layer_grads = {}
    def _make_hook(k: int):
        def hook(module, grad_in, grad_out):
            layer_grads[k] = grad_out[0].detach().cpu()
        return hook
    handles = [
        rg.register_full_backward_hook(_make_hook(k))
        for k, rg in enumerate(model.rg_layers)
    ]
    model.zero_grad()
    out  = model(X.to(device))
    loss = criterion(out, y.to(device))
    loss.backward()
    for h in handles:
        h.remove()
    xi_vals = []
    for k in sorted(layer_grads.keys()):
        G_k   = layer_grads[k].float().numpy()   
        F     = (G_k.T @ G_k) / max(B, 1)        
        ev    = np.linalg.eigvalsh(F)
        ev    = np.clip(ev, 1e-12, None)
        xi_k  = float(1.0 / np.sqrt(np.mean(1.0 / ev) + 1e-12))
        xi_vals.append(xi_k)
    return np.array(xi_vals)
def _fit_exponential(xi_values: np.ndarray) -> dict:
    k = np.arange(len(xi_values), dtype=float)
    def _exp(k, xi_0, k_c):
        return xi_0 * np.exp(-k / k_c)
    xi_0_guess = float(xi_values[0])
    if len(xi_values) >= 2 and xi_values[-1] > 0:
        log_ratio  = np.log(xi_values[-1] / max(xi_values[0], 1e-10))
        k_c_guess  = -float(len(xi_values) - 1) / log_ratio if log_ratio < 0 else float(len(xi_values))
    else:
        k_c_guess = float(len(xi_values) / 3)
    try:
        popt, _ = curve_fit(
            _exp, k, xi_values,
            p0=[xi_0_guess, k_c_guess],
            bounds=([0.0, 0.1], [np.inf, np.inf]),
            maxfev=20000,
        )
        xi_0, k_c = float(popt[0]), float(popt[1])
    except Exception:
        log_xi = np.log(xi_values + 1e-12)
        coeffs = np.polyfit(k, log_xi, 1)
        xi_0   = float(np.exp(coeffs[1]))
        k_c    = float(-1.0 / coeffs[0]) if coeffs[0] < 0 else float(len(xi_values))
    xi_pred = _exp(k, xi_0, k_c)
    ss_res  = float(((xi_values - xi_pred) ** 2).sum())
    ss_tot  = float(((xi_values - xi_values.mean()) ** 2).sum())
    r2      = float(1.0 - ss_res / max(ss_tot, 1e-12))
    chi1    = float(np.exp(-1.0 / k_c))
    return {
        :       round(xi_0, 6),
        :        round(k_c,  6),
        :         round(r2,   6),
        :       round(chi1, 6),
        :  [round(x, 6) for x in xi_values.tolist()],
    }
def _train_model(
    model: RGNetH1,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    N = len(X_train)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx  = perm[start:start + batch_size]
            xb   = X_train[idx].to(device)
            yb   = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        val_acc = (model(X_val.to(device)).argmax(1) == y_val.to(device)).float().mean().item()
    return val_acc
def _run_single(
    width: int,
    n_layers: int,
    n_train: int,
    epochs: int,
    sigma_w: float,
    sigma_b: float,
    seed: int,
    device: torch.device,
    data_root: Path,
    n_classes: int = 10,
    input_dim: int = 64,
) -> dict:
    SeedRegistry.get_instance().set_master_seed(seed)
    X, y, xi_data = _load_or_generate_data(
        n_train=n_train, n_classes=n_classes, input_dim=input_dim,
        xi_data=10.0, seed=seed, data_root=data_root,
    )
    n_val   = max(len(X) // 5, 20)
    X_val, y_val   = X[:n_val],  y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    B = len(X_train)
    model    = RGNetH1(input_dim, width, n_classes, n_layers, sigma_w, sigma_b).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    val_acc  = _train_model(
        model, X_train, y_train, X_val, y_val, device,
        epochs=epochs, batch_size=min(256, B),
    )
    xi_values = _measure_xi_gradient_fisher(model, X_train, y_train, device)
    fit = _fit_exponential(xi_values)
    fit["val_accuracy"] = round(val_acc, 4)
    fit["n_params"]     = n_params
    fit["width"]        = width
    fit["seed"]         = seed
    fit["B"]            = B
    fit["gamma"]        = round(width / max(B, 1), 4)
    fit["sigma_w"]      = sigma_w   
    return fit
def run_h1_experiment(params: dict, results_dir: Path, fast_track: bool) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    device    = DeviceManager.get_instance().get_device()
    data_root = _ROOT / "data"
    all_results   = {}
    global_r2_all = []
    for width in params["widths"]:
        logger.info("Width N=%d  (γ = %d/%d = %.3f  noise=%.1f%%)",
                    width, width, params["n_train"],
                    width / params["n_train"],
                    100 * np.sqrt(2 / params["n_train"]))
        width_results = []
        for seed in range(params["n_seeds"]):
            t0  = time.time()
            fit = _run_single(
                width=width,
                n_layers=params["n_layers"],
                n_train=params["n_train"],
                epochs=params["epochs"],
                sigma_w=params["sigma_w"],   
                sigma_b=params["sigma_b"],
                seed=seed * 1000 + width,
                device=device,
                data_root=data_root,
            )
            elapsed = time.time() - t0
            logger.info(
                ,
                seed, fit["xi_0"], fit["k_c"], fit["chi1"], fit["r2"],
                fit["val_accuracy"], elapsed,
            )
            width_results.append(fit)
            global_r2_all.append(fit["r2"])
        r2_vals = [r["r2"] for r in width_results]
        logger.info(
            ,
            width, np.mean(r2_vals), np.std(r2_vals), params["r2_threshold"],
        )
        all_results[f"width_{width}"] = width_results
    tag    = "[FAST_TRACK_UNVERIFIED]" if fast_track else "[VERIFIED]"
    passed = bool(np.mean(global_r2_all) >= params["r2_threshold"])
    output = {
        :            tag,
        :     "H1",
        :          "xi(k) = xi_0 * exp(-k / k_c) with R² > 0.95",
        :         passed,
        :        round(float(np.mean(global_r2_all)), 4),
        :         round(float(np.std(global_r2_all)), 4),
        :   params["r2_threshold"],
        :         {k: v for k, v in params.items()
                           if k not in ("xi_values",)},
        :   "FIX: using sigma_w=1.4 (chi1=0.894), NOT critical sigma_w=1.4823 (chi1=1.0)",
        :    "FIX: gradient Fisher E[(dL/dh)(dL/dh)^T], NOT random WW^T/n",
        :        all_results,
    }
    out_path = results_dir / "h1_results_fixed.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", out_path)
    logger.info("H1 %s: mean R² = %.4f (threshold %.2f)",
                 if passed else "FAILS",
                float(np.mean(global_r2_all)), params["r2_threshold"])
    return output
def parse_args():
    p = argparse.ArgumentParser(description="H1 Scale Correspondence Validation (FIXED)")
    p.add_argument("--fast-track", action="store_true",  help="Fast-track mode (3-10 min)")
    p.add_argument("--widths",     nargs="+", type=int,  help="Override width list")
    p.add_argument("--n-seeds",    type=int,             help="Override seed count")
    p.add_argument("--n-train",    type=int,             help="Override training samples per experiment")
    p.add_argument("--epochs",     type=int,             help="Override training epochs")
    p.add_argument("--results-dir",type=str, default="results/h1", help="Output directory")
    return p.parse_args()
def main():
    args   = parse_args()
    params = FAST_TRACK.copy() if args.fast_track else FULL.copy()
    if args.widths:   params["widths"]   = args.widths
    if args.n_seeds:  params["n_seeds"]  = args.n_seeds
    if args.n_train:  params["n_train"]  = args.n_train
    if args.epochs:   params["epochs"]   = args.epochs
    results_dir = Path(args.results_dir)
    t0 = time.time()
    logger.info("=== H1 Scale Correspondence Validation (FIXED) ===")
    logger.info("Mode: %s",  "FAST-TRACK" if args.fast_track else "FULL")
    logger.info("Widths: %s", params["widths"])
    logger.info("Seeds:  %d", params["n_seeds"])
    logger.info("sigma_w = %.4f  (FIX: was _critical_sigma_w() = 1.4823; ALSO FIX: paper's 1.4 is chaotic)", params["sigma_w"])
    logger.info("Fisher: gradient-based E[(dL/dh)(dL/dh)^T]  (FIX: was random WW^T/n)")
    logger.info("n_train = %d per seed (γ = width/n_train)", params["n_train"])
    run_h1_experiment(params, results_dir, fast_track=args.fast_track)
    logger.info("=== H1 COMPLETE in %.1fs ===", time.time() - t0)
if __name__ == "__main__":
    main()