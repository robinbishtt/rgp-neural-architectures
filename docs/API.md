# API Reference

## Core Mathematics (`src/core/`)

### Fisher Information

```python
from src.core.fisher.fisher_metric import FisherMetric

fm = FisherMetric(clip_eigenvalues=True, min_eigenvalue=1e-10)

# PULLBACK (correct): g^(k) = J^T g^(k-1) J
G_k = fm.pullback(G_prev, J_k)

# Compute metrics for all layers of a model
metrics = fm.compute_from_model(model, x, layer_indices=[0,5,10,20])
```

### Correlation Length

```python
from src.core.correlation.two_point import chi1_gauss_hermite, critical_sigma_w2

# chi1 = sigma_w^2 * E[phi'(z)^2]
chi1 = chi1_gauss_hermite(sigma_w2=1.4**2, nonlinearity="tanh")

# Find critical sigma_w via bisection
sigma_w_star_sq = critical_sigma_w2("tanh")  # returns sigma_w^2

# xi_depth = -1/log(chi1)
xi_depth = -1.0 / np.log(chi1)
```

### Exponential Decay Fitter

```python
from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter

fitter = ExponentialDecayFitter(p0_xi0=10.0, p0_kc=20.0)
result = fitter.fit(layers, xi_values)
# result.xi_0, result.k_c, result.r2, result.chi1
```

## Architectures (`src/architectures/`)

### RG-Net

```python
from src.architectures.rg_net.rg_net import RGNetStandard, build_rg_net

model = RGNetStandard(input_dim=784, hidden_dim=512, output_dim=10, depth=100)
model = build_rg_net("standard", input_dim=784, hidden_dim=512, output_dim=10, depth=100)
```

### Baselines

```python
from src.architectures.baselines.resnet_baseline import ResNetBaseline
from src.architectures.baselines.wavelet_baseline import WaveletCNNBaseline
from src.architectures.baselines.tensor_net_baseline import TensorNetBaseline

resnet  = ResNetBaseline(in_features=784, n_classes=10, depth=50, width=512)
wavelet = WaveletCNNBaseline(input_dim=784, hidden_dim=256, output_dim=10, n_scales=4)
tensor  = TensorNetBaseline(input_dim=784, hidden_dim=512, output_dim=10, depth=4, bond_dim=32)
```

## Training (`src/training/`)

```python
from src.training.trainer import Trainer, TrainingConfig

cfg = TrainingConfig(n_epochs=100, lr=1e-3, batch_size=256, seed=42)
trainer = Trainer(model, cfg, device=device, checkpoint_dir="checkpoints/")
result = trainer.train(train_loader, val_loader)
# result.best_val_acc, result.train_losses, result.elapsed_s
```

## Utilities

```python
from src.utils.seed_registry import SeedRegistry
from src.utils.device_manager import DeviceManager

# Global determinism
SeedRegistry.get_instance().set_master_seed(42)

# Hardware-agnostic device detection
device = DeviceManager.get_instance().get_device()  # CUDA > MPS > CPU
```
