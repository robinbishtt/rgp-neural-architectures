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
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from src.utils.seed_registry import SeedRegistry
from src.utils.device_manager import DeviceManager
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_validate_h1")
HIER_DATASETS = {
    : {"xi": 5.0,   "file": "hier1_xi5_n500.npz"},
    : {"xi": 15.0,  "file": "hier2_xi15_n500.npz"},
    : {"xi": 50.0,  "file": "hier3_xi50_n500.npz"},
    : {"xi": 100.0, "file": "hier4_xi100_n500.npz"},
    : {"xi": 200.0, "file": "hier5_xi200_n500.npz"},
}
FAST_PARAMS = {
    :    10,      
    :     1000,    
    :       200,
    :   10,
    :   64,
    :      10,
    :  128,
    :          1e-3,
    : 1.2,     
    :     0.3,
    : 0.70,  
    :        42,
}
FULL_PARAMS = {
    :    30,
    :     5000,
    :       1000,
    :   10,
    :   64,
    :      100,
    :  256,
    :          1e-3,
    : 1.2,
    :     0.3,
    : 0.80,
    :        42,
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
class RGNetDepthSweep(nn.Module):
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
def make_dataset(
    xi_data: float,
    n_samples: int,
    n_classes: int,
    input_dim: int,
    seed: int,
    data_root: Path,
) -> tuple:
    for info in HIER_DATASETS.values():
        if abs(info["xi"] - xi_data) < 1e-3:
            fpath = data_root / info["file"]
            if fpath.exists():
                d = np.load(str(fpath), allow_pickle=True)
                X_raw = d["X"].astype(np.float32)
                y_raw = d["y"].astype(np.int64)
                rng   = np.random.default_rng(seed)
                idx   = rng.permutation(len(X_raw))[:n_samples]
                return torch.from_numpy(X_raw[idx]), torch.from_numpy(y_raw[idx])
    rng          = np.random.default_rng(seed)
    noise_std    = xi_data
    class_centres = rng.standard_normal((n_classes, input_dim)) * xi_data
    sub_centres   = class_centres + rng.standard_normal((n_classes, input_dim)) * (xi_data / 2)
    X_list, y_list = [], []
    per_class = max(n_samples // n_classes, 1)
    for cls in range(n_classes):
        x = sub_centres[cls] + rng.standard_normal((per_class, input_dim)) * noise_std
        X_list.append(x.astype(np.float32))
        y_list.extend([cls] * per_class)
    X    = np.vstack(X_list)
    y    = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(y))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])
def measure_gradient_fisher_xi(
    model: RGNetDepthSweep,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    B         = X.shape[0]
    n_k       = model.hidden_dim
    gamma = n_k / max(B, 1)
    if gamma > 0.1:
        logger.warning("γ = %d/%d = %.3f > 0.1 (noise: %.1f%%)", n_k, B, gamma,
                       100 * np.sqrt(2 / B))
    layer_grads: dict = {}
    def _make_hook(k: int):
        def _hook(module, grad_in, grad_out):
            layer_grads[k] = grad_out[0].detach().cpu()
        return _hook
    handles = [
        rg.register_full_backward_hook(_make_hook(k))
        for k, rg in enumerate(model.rg_layers)
    ]
    model.zero_grad()
    loss = criterion(model(X.to(device)), y.to(device))
    loss.backward()
    for h in handles:
        h.remove()
    xi_vals = []
    for k in sorted(layer_grads.keys()):
        G_k  = layer_grads[k].float().numpy()
        F    = (G_k.T @ G_k) / max(B, 1)
        ev   = np.clip(np.linalg.eigvalsh(F), 1e-12, None)
        xi_k = float(1.0 / np.sqrt(np.mean(1.0 / ev) + 1e-12))
        xi_vals.append(xi_k)
    return np.array(xi_vals)
def train_one(
    model: RGNetDepthSweep,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    params: dict,
) -> tuple:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["epochs"]
    )
    N          = len(X_train)
    batch_size = params["batch_size"]
    val_curve  = []
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb  = X_train[idx].to(device)
            yb  = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_acc = (
                model(X_val.to(device)).argmax(1) == y_val.to(device)
            ).float().mean().item()
        val_curve.append(round(val_acc, 4))
    return val_curve, val_curve[-1]
def find_L_min_from_val_curve(val_curve: list, threshold: float = 0.70) -> float:
    for i, acc in enumerate(val_curve):
        if acc >= threshold:
            return float(i + 1)  
    return float(len(val_curve))  
def run_one_dataset(xi_data: float, params: dict, device: torch.device, data_root: Path) -> dict:
    seed = params["seed"]
    SeedRegistry.get_instance().set_master_seed(seed)
    X, y = make_dataset(
        xi_data=xi_data, n_samples=params["n_train"] + params["n_val"],
        n_classes=params["n_classes"], input_dim=params["input_dim"],
        seed=seed, data_root=data_root,
    )
    X_train, y_train = X[:params["n_train"]],  y[:params["n_train"]]
    X_val,   y_val   = X[params["n_train"]:],  y[params["n_train"]:]
    model = RGNetDepthSweep(
        input_dim=params["input_dim"],
        hidden_dim=64,
        output_dim=params["n_classes"],
        depth=params["n_layers"],
        sigma_w=params["sigma_w"],
        sigma_b=params["sigma_b"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    val_curve, final_val_acc = train_one(
        model, X_train, y_train, X_val, y_val, device, params
    )
    xi_profile = measure_gradient_fisher_xi(model, X_train, y_train, device)
    k_arr = np.arange(len(xi_profile), dtype=float)
    def _exp(k, xi_0, k_c):
        return xi_0 * np.exp(-k / k_c)
    try:
        popt, _ = curve_fit(
            _exp, k_arr, xi_profile,
            p0=[float(xi_profile[0]), float(len(xi_profile) / 3)],
            bounds=([0, 0.1], [np.inf, np.inf]), maxfev=20000,
        )
        xi_0, k_c = float(popt[0]), float(popt[1])
    except Exception:
        coeffs = np.polyfit(k_arr, np.log(xi_profile + 1e-12), 1)
        xi_0   = float(np.exp(coeffs[1]))
        k_c    = float(-1.0 / (coeffs[0] + 1e-12))
    xi_pred = _exp(k_arr, xi_0, k_c)
    ss_res  = float(((xi_profile - xi_pred) ** 2).sum())
    ss_tot  = float(((xi_profile - xi_profile.mean()) ** 2).sum())
    r2      = float(1.0 - ss_res / max(ss_tot, 1e-12))
    chi1    = float(np.exp(-1.0 / k_c))
    L_min = find_L_min_from_val_curve(val_curve, params["acc_threshold"])
    return {
        :        xi_data,
        :           round(xi_0, 4),
        :            round(k_c, 4),
        :           round(chi1, 4),
        :             round(r2, 4),
        :          L_min,
        :  round(final_val_acc, 4),
        :       n_params,
        :    round(64 / params["n_train"], 4),
        :     [round(x, 6) for x in xi_profile.tolist()],
        :      val_curve,
    }
def verify_theorem3(results: list) -> dict:
    xi_arr   = np.array([r["xi_data"] for r in results])
    Lmin_arr = np.array([r["L_min"]   for r in results])
    log_xi   = np.log(xi_arr)
    A = np.vstack([log_xi, np.ones_like(log_xi)]).T
    ab, _, _, _ = np.linalg.lstsq(A, Lmin_arr, rcond=None)
    a, b = float(ab[0]), float(ab[1])
    L_pred = a * log_xi + b
    ss_res = float(((Lmin_arr - L_pred) ** 2).sum())
    ss_tot = float(((Lmin_arr - Lmin_arr.mean()) ** 2).sum())
    r2     = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {
        :     round(a,  4),
        : round(b,  4),
        :          round(r2, 4),
        :      [round(x, 4) for x in log_xi.tolist()],
        :       [round(x, 4) for x in Lmin_arr.tolist()],
        :      [round(x, 4) for x in L_pred.tolist()],
        :      r2 > 0.97,
        :        "Theorem 3: L_min = k_c*log(xi_data/xi_target). R² > 0.997 claimed by paper.",
    }
def main():
    p = argparse.ArgumentParser(description="Train + H1 validate on all hierarchical datasets")
    p.add_argument("--fast",       action="store_true", help="Fast mode (~10 min)")
    p.add_argument("--xi-data",    type=float, nargs="+", help="Select xi_data values (default: all 5)")
    p.add_argument("--n-train",    type=int,  help="Override n_train")
    p.add_argument("--epochs",     type=int,  help="Override epochs")
    p.add_argument("--n-layers",   type=int,  help="Override depth (n_layers)")
    p.add_argument("--results-dir",type=str,  default="results/h1")
    p.add_argument("--seed",       type=int,  default=42)
    args = p.parse_args()
    params = FAST_PARAMS.copy() if args.fast else FULL_PARAMS.copy()
    params["seed"] = args.seed
    if args.n_train:  params["n_train"]  = args.n_train
    if args.epochs:   params["epochs"]   = args.epochs
    if args.n_layers: params["n_layers"] = args.n_layers
    xi_targets = args.xi_data or [5.0, 15.0, 50.0, 100.0, 200.0]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_root   = _ROOT / "data"
    device      = DeviceManager.get_instance().get_device()
    logger.info("=== H1 Train + Validate ===")
    logger.info("Mode: %s", "FAST" if args.fast else "FULL")
    logger.info("Datasets: xi in %s", xi_targets)
    logger.info("sigma_w = %.4f → chi1 ≈ 0.894 → k_c ≈ 8.9", params["sigma_w"])
    logger.info("Fisher: gradient-based F^(k) = E[(dL/dh^k)(dL/dh^k)^T]")
    logger.info("n_train = %d, gamma = %.4f, noise = %.1f%%",
                params["n_train"], 64 / params["n_train"],
                100 * np.sqrt(2 / params["n_train"]))
    all_results = []
    for xi in xi_targets:
        logger.info("─── xi_data = %.1f ───", xi)
        t0 = time.time()
        res = run_one_dataset(xi, params, device, data_root)
        logger.info(
            ,
            xi, res["r2"], res["xi_0"], res["k_c"], res["chi1"],
            res["final_val_acc"], time.time() - t0,
        )
        all_results.append(res)
    thm3 = verify_theorem3(all_results)
    logger.info("")
    logger.info("Theorem 3 (L_min ~ log(xi_data)):")
    logger.info("  L_min = %.3f * log(xi) + %.3f", thm3["slope_a"], thm3["intercept_b"])
    logger.info("  R² = %.4f  (paper claim: 0.997)  PASS=%s", thm3["r2"], thm3["passes"])
    mean_r2 = np.mean([r["r2"] for r in all_results])
    logger.info("")
    logger.info("H1 mean R² across datasets: %.4f (threshold 0.95)", mean_r2)
    combined = {
        :           params,
        :     "FIX: 1.4 (chi1=0.894), NOT 1.4823 (chi1=1.0)",
        :    "gradient E[(dL/dh)(dL/dh)^T]",
        :       round(float(mean_r2), 4),
        :        bool(mean_r2 >= 0.95),
        :         thm3,
        :      all_results,
    }
    out_path = results_dir / "combined_h1_results.json"
    out_path.write_text(json.dumps(combined, indent=2))
    logger.info("Saved: %s", out_path)
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'xi_data':>8}  {'R²(xi decay)':>14}  {'k_c':>8}  {'chi1':>8}  {'val_acc':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['xi_data']:>8.1f}  {r['r2']:>14.4f}  "
              f"{r['k_c']:>8.3f}  {r['chi1']:>8.4f}  {r['final_val_acc']:>8.4f}")
    print("-" * 60)
    print(f"{'MEAN':>8}  {mean_r2:>14.4f}")
    print()
    print(f"Theorem 3 R² = {thm3['r2']:.4f}  (paper: 0.997)")
if __name__ == "__main__":
    main()