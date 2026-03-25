from __future__ import annotations
import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.seed_registry import SeedRegistry
from src.utils.device_manager import DeviceManager
from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter
warnings.filterwarnings("ignore", category=UserWarning)
class RGLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sigma_w: float = 1.2, sigma_b: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act    = nn.Tanh()
        nn.init.normal_(self.linear.weight, std=sigma_w / math.sqrt(in_dim))
        nn.init.normal_(self.linear.bias,   std=sigma_b)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
class SmallRGNet(nn.Module):
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
        self.sigma_w    = sigma_w
        self.sigma_b    = sigma_b
        self.depth      = depth
        self.hidden_dim = hidden_dim
        layers = []
        dims   = [input_dim] + [hidden_dim] * depth
        for i in range(depth):
            layers.append(RGLayer(dims[i], dims[i + 1], sigma_w, sigma_b))
        self.rg_layers = nn.ModuleList(layers)
        self.head       = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.head.weight, std=sigma_w / math.sqrt(hidden_dim))
        nn.init.normal_(self.head.bias,   std=sigma_b)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.rg_layers:
            x = layer(x)
        return self.head(x)
def make_hard_hierarchical_data(
    n: int,
    d: int,
    n_classes: int,
    xi: float,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    noise_std = xi                          
    class_centres = rng.standard_normal((n_classes, d)) * xi
    sub_centres   = class_centres + rng.standard_normal((n_classes, d)) * (xi / 2)
    X_list, y_list = [], []
    per_class = n // n_classes
    for cls in range(n_classes):
        x = sub_centres[cls] + rng.standard_normal((per_class, d)) * noise_std
        X_list.append(x.astype(np.float32))
        y_list.extend([cls] * per_class)
    X    = np.vstack(X_list)
    y    = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(y))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])
def measure_gradient_fisher_xi(
    model: SmallRGNet,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
) -> tuple:
    model.eval()
    B = X.shape[0]
    n = model.hidden_dim
    gamma = n / B
    if gamma > 0.1:
        print(f"  WARNING: γ = n/B = {n}/{B} = {gamma:.3f} > 0.1  "
              f"(need B ≥ 100n = {100*n} for 1% accuracy). "
              f"Current noise per eigenvalue: {100*np.sqrt(2/B):.1f}%")
    layer_grads: dict = {}  
    hooks = []
    for k, rg_layer in enumerate(model.rg_layers):
        def _hook_fn(module, grad_in, grad_out, k=k):
            layer_grads[k] = grad_out[0].detach().cpu()
        hooks.append(rg_layer.register_full_backward_hook(_hook_fn))
    x_dev = X.to(device)
    y_dev = y.to(device)
    model.zero_grad()
    out  = model(x_dev)
    loss = criterion(out, y_dev)
    loss.backward()
    for h in hooks:
        h.remove()
    xi_vals  = []
    eta_vals = []
    for k in sorted(layer_grads.keys()):
        G_k  = layer_grads[k].numpy()  
        n_k  = G_k.shape[1]
        F    = (G_k.T @ G_k) / max(B, 1)   
        ev   = np.linalg.eigvalsh(F)
        ev   = np.clip(ev, 1e-12, None)
        xi_k  = float(1.0 / np.sqrt(np.mean(1.0 / ev) + 1e-12))
        eta_k = float(ev.max())
        xi_vals.append(xi_k)
        eta_vals.append(eta_k)
    return np.array(xi_vals), np.array(eta_vals)
def train_model(
    model: SmallRGNet,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
) -> dict:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    train_accs, val_accs, train_losses = [], [], []
    N = len(X_train)
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm    = torch.randperm(N)
        ep_loss = 0.0
        correct = 0
        total   = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb  = X_train[idx].to(device)
            yb  = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_acc    = (val_logits.argmax(1) == y_val.to(device)).float().mean().item()
        train_acc = correct / max(total, 1)
        train_accs.append(round(train_acc, 4))
        val_accs.append(round(val_acc, 4))
        train_losses.append(round(ep_loss / max(total, 1), 6))
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}: loss={train_losses[-1]:.4f}  "
                  f"train={train_acc:.4f}  val={val_acc:.4f}")
    return {"train_accs": train_accs, "val_accs": val_accs, "train_losses": train_losses}
def run_proof_of_life(
    depth: int   = 20,       
    width: int   = 32,
    input_dim: int = 64,
    n_classes: int = 10,     
    xi_data: float = 10.0,
    n_train: int = 5000,     
    n_epochs: int = 50,      
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    SeedRegistry.get_instance().set_master_seed(seed)
    device = DeviceManager.get_instance().get_device()
    criterion = nn.CrossEntropyLoss()
    print(f"  Device:    {device}")
    print(f"  Model:     RGNet depth={depth}, width={width}")
    print(f"  Data:      xi_data={xi_data}, N_train={n_train}, d={input_dim}, K={n_classes}")
    print(f"  Init:      sigma_w=1.2, sigma_b=0.3  →  chi1≈0.894, k_c≈8.9")
    print(f"  Batch:     B={n_train} for Fisher (γ={width/n_train:.4f}, noise={100*np.sqrt(2/n_train):.1f}%)")
    print()
    model   = SmallRGNet(input_dim, width, n_classes, depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    X_train, y_train = make_hard_hierarchical_data(n_train, input_dim, n_classes, xi_data, seed)
    X_val,   y_val   = make_hard_hierarchical_data(n_train // 5, input_dim, n_classes, xi_data, seed + 1)
    print("  Measuring gradient Fisher xi profile at INITIALIZATION...")
    xi_init, eta_init = measure_gradient_fisher_xi(model, X_train, y_train, device, criterion)
    print(f"  xi_init (first 5 layers): {xi_init[:5].tolist()}")
    print()
    print("  Training...")
    train_result = train_model(
        model, X_train, y_train, X_val, y_val, device,
        n_epochs=n_epochs, lr=lr, batch_size=batch_size,
    )
    print()
    print("  Measuring gradient Fisher xi profile AFTER TRAINING...")
    xi_after, eta_after = measure_gradient_fisher_xi(model, X_train, y_train, device, criterion)
    print(f"  xi_after (first 5): {xi_after[:5].round(4).tolist()}")
    if np.any(np.isnan(xi_after)):
        print("  WARNING: NaN in xi_after — eigenvalue collapse. Increase n_train.")
    k_arr  = np.arange(len(xi_after), dtype=float)
    fitter = ExponentialDecayFitter(p0_xi0=float(xi_after[0]), p0_kc=float(depth / 2))
    try:
        fit = fitter.fit(k_arr, xi_after)
        xi_fit = {"xi_0": fit.xi_0, "k_c": fit.k_c, "r2": fit.r2, "chi1": fit.chi1}
    except Exception as e:
        xi_fit = {"xi_0": float(xi_after[0]), "k_c": float(depth), "r2": 0.0, "chi1": 0.0, "error": str(e)}
    chi1_theory = np.exp(-1.0 / 7.18)   
    eta_theory  = eta_after[0] * (chi1_theory ** k_arr[:len(eta_after)])
    ss_res  = np.sum((eta_after - eta_theory) ** 2)
    ss_tot  = np.sum((eta_after - eta_after.mean()) ** 2)
    thm1_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    if len(xi_after) >= 2 and np.all(xi_after > 1e-10):
        slope_real   = float(np.polyfit(k_arr[:len(xi_after)], np.log(xi_after + 1e-10), 1)[0])
        slope_theory = float(-1.0 / 8.925)
        trend_ratio  = slope_real / (slope_theory + 1e-12)
        trend_ok     = bool(0.5 <= abs(trend_ratio) <= 2.0)
    else:
        slope_real, slope_theory, trend_ratio, trend_ok = 0, 0, 0, False
    print()
    print("  ── Results ─────────────────────────────────────────────")
    print(f"  Final val accuracy:        {train_result['val_accs'][-1]:.4f}")
    print(f"  Final train accuracy:      {train_result['train_accs'][-1]:.4f}")
    print(f"  Xi decay R² (post-train):  {xi_fit['r2']:.4f}  (threshold 0.95 for depth={depth})")
    print(f"  Fitted k_c:                {xi_fit['k_c']:.3f}  (theory: 8.925)")
    print(f"  Fitted chi1:               {xi_fit['chi1']:.4f}  (theory: 0.894)")
    print(f"  Theorem 1 R² (eta contraction): {thm1_r2:.4f}")
    print(f"  Trend ratio (slope_real/slope_theory): {trend_ratio:.3f}  {'PASS' if trend_ok else 'FAIL'}")
    print()
    if xi_fit["r2"] < 0.90:
        print("  DIAGNOSTICS (R² < 0.90):")
        gamma = width / n_train
        print(f"    γ = n/B = {width}/{n_train} = {gamma:.4f}  (need < 0.01 for 1% noise)")
        print(f"    depth={depth}  (need ≥ 20 for R² > 0.95)")
        print(f"    If gamma OK but R² still low → model may not have converged")
        print(f"    Try: --epochs 100 --n-train {10*width}")
    return {
        : {
            : depth, "width": width, "input_dim": input_dim,
            : n_classes, "xi_data": xi_data,
            : n_train, "n_epochs": n_epochs, "seed": seed,
            : 1.2, "sigma_b": 0.3,
            : round(width / n_train, 4),
            : [
                ,
                ,
                ,
                ,
                ,
            ],
        },
        : {
            : train_result["train_losses"],
            :   train_result["train_accs"],
            :     train_result["val_accs"],
            : train_result["val_accs"][-1],
            : train_result["train_accs"][-1],
            : n_params,
        },
        : {
            :  [round(x, 6) for x in xi_init.tolist()],
            : [round(x, 6) for x in xi_after.tolist()],
            : [round(x, 6) for x in eta_init.tolist()],
            :[round(x, 6) for x in eta_after.tolist()],
            :   "gradient_fisher_E[(dL/dh)(dL/dh)^T]",
            : round(width / n_train, 4),
        },
        : xi_fit,
        : {
            : round(thm1_r2, 4),
            :     round(chi1_theory, 4),
        },
        : {
            :   round(slope_real, 4),
            : round(slope_theory, 4),
            :        round(trend_ratio, 3),
            :       trend_ok,
        },
    }
def main():
    p = argparse.ArgumentParser(description="Proof-of-life: fixed training verification")
    p.add_argument("--depth",    type=int,   default=20,    help="Network depth (need ≥20)")
    p.add_argument("--width",    type=int,   default=32,    help="Hidden width")
    p.add_argument("--xi-data",  type=float, default=10.0,  help="Data correlation length")
    p.add_argument("--n-train",  type=int,   default=5000,  help="Training samples (need ≥100*width)")
    p.add_argument("--epochs",   type=int,   default=50,    help="Training epochs")
    p.add_argument("--n-classes",type=int,   default=10,    help="Number of classes")
    p.add_argument("--fast",     action="store_true",       help="Fast mode (depth=5, n=500)")
    p.add_argument("--output",   type=str,   default="results/proof_of_life/")
    args = p.parse_args()
    if args.fast:
        args.depth    = 5
        args.n_train  = 500
        args.epochs   = 10
        args.n_classes = 4
        print("FAST MODE: depth=5, n=500, epochs=10 (for smoke-test only)")
        print("WARNING: gamma =", args.width, "/", args.n_train, "=",
              round(args.width / args.n_train, 3), "— high noise, R² may be low")
        print()
    t0 = time.time()
    print("=== Proof-of-Life Training (FIXED) ===")
    print()
    results = run_proof_of_life(
        depth=args.depth,
        width=args.width,
        xi_data=args.xi_data,
        n_train=args.n_train,
        n_epochs=args.epochs,
        n_classes=args.n_classes,
    )
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pol_results_fixed.json"
    out_path.write_text(json.dumps(results, indent=2))
    elapsed = time.time() - t0
    print(f"=== Complete in {elapsed:.1f}s ===")
    print(f"Results: {out_path}")
    print()
    print("Summary:")
    print(f"  Val accuracy:  {results['training']['final_val_acc']:.4f}")
    print(f"  Train accuracy:{results['training']['final_train_acc']:.4f}")
    print(f"  Xi decay R²:   {results['exponential_fit']['r2']:.4f}")
    print(f"  Theorem 1 R²:  {results['theorem1_verification']['contraction_r2']:.4f}")
    print(f"  Trend OK:      {results['trend_consistency']['passes']}")
if __name__ == "__main__":
    main()