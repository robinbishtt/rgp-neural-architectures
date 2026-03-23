"""
data/generation_script.py

Generate all hierarchical datasets used in the paper.

Hier-1 through Hier-5 datasets with correlation lengths xi in {5, 15, 50, 100, 200}.

Usage:
    # Full datasets (paper scale: N=50000 per split)
    python data/generation_script.py

    # Fast-track (N=1000 per split, for reviewers)
    python data/generation_script.py --fast-track

    # Specific dataset
    python data/generation_script.py --hier 3

Outputs:
    data/full/hier1_xi5_train.npz    (N=50000, d=784, 10 classes)
    data/full/hier1_xi5_val.npz      (N=10000)
    data/full/hier1_xi5_ood.npz      (N=5000, shifted correlation structure)
    ...
    data/full/hier5_xi200_train.npz
    data/full/hier5_xi200_val.npz
    data/full/hier5_xi200_ood.npz

    data/fast/  (same structure, N=1000/200/100)

Checksums:
    SHA-256 checksums written to data/checksums.json.
    Verified at training time by src/provenance/data_auditor.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np


# ── Dataset specifications ────────────────────────────────────────────
HIER_SPECS = {
    1: {"xi": 5.0,   "n_scales": 2, "name": "Hier-1"},
    2: {"xi": 15.0,  "n_scales": 3, "name": "Hier-2"},
    3: {"xi": 50.0,  "n_scales": 4, "name": "Hier-3"},
    4: {"xi": 100.0, "n_scales": 5, "name": "Hier-4"},
    5: {"xi": 200.0, "n_scales": 6, "name": "Hier-5"},
}

FULL_SIZES  = {"train": 50000, "val": 10000, "ood": 5000}
FAST_SIZES  = {"train": 1000,  "val": 200,   "ood": 100}

INPUT_DIM  = 784
N_CLASSES  = 10
MASTER_SEED = 42


# ── Core generation function ──────────────────────────────────────────

def generate_hierarchical(
    n_samples: int,
    xi:        float,
    n_scales:  int,
    n_classes: int  = N_CLASSES,
    input_dim: int  = INPUT_DIM,
    ood_shift: float = 0.0,
    seed:      int  = MASTER_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one hierarchical dataset split.

    Parameters
    ----------
    n_samples : number of samples
    xi        : target correlation length
    n_scales  : number of hierarchical scale levels
    n_classes : number of output classes
    input_dim : feature dimension
    ood_shift : OOD covariance rotation magnitude (0 = in-distribution)
    seed      : random seed

    Returns
    -------
    X : float32 (n_samples, input_dim)
    y : int64   (n_samples,)
    """
    rng = np.random.default_rng(seed)

    # Hierarchical covariance structure:
    # Level 0 (coarsest): cluster centres at scale xi
    # Level 1: sub-clusters at scale xi/2
    # ...
    # Level n_scales-1 (finest): individual samples at scale xi/2^(n_scales-1)

    class_centres = rng.standard_normal((n_classes, input_dim)) * xi

    # Build scale-level sub-structures
    current_centres = class_centres.copy()
    all_sub_centres = [current_centres]
    for s in range(1, n_scales):
        scale = xi / (2.0 ** s)
        sub   = current_centres[:, np.newaxis, :] +                 rng.standard_normal((n_classes, 2, input_dim)) * scale
        # Reshape: (n_classes * 2, input_dim)
        current_centres = sub.reshape(-1, input_dim)
        all_sub_centres.append(current_centres)

    # Apply OOD covariance rotation
    if ood_shift > 0.0:
        Q, _ = np.linalg.qr(rng.standard_normal((input_dim, input_dim)))
        rotation = (1.0 - ood_shift) * np.eye(input_dim) + ood_shift * Q
    else:
        rotation = np.eye(input_dim)

    # Sample from finest level
    finest_centres = all_sub_centres[-1]   # (n_classes * 2^(n_scales-1), input_dim)
    n_finest_per_class = finest_centres.shape[0] // n_classes
    finest_noise_std = xi / (2.0 ** n_scales)

    X_list, y_list = [], []
    per_class = n_samples // n_classes

    for cls in range(n_classes):
        idx_start = cls * n_finest_per_class
        idx_end   = idx_start + n_finest_per_class
        centres   = finest_centres[idx_start:idx_end]

        # Distribute samples across sub-centres within class
        for i, centre in enumerate(centres):
            n_here = per_class // n_finest_per_class + (1 if i < per_class % n_finest_per_class else 0)
            noise  = rng.standard_normal((n_here, input_dim)) * finest_noise_std
            x_here = (centre + noise) @ rotation.T
            X_list.append(x_here.astype(np.float32))
            y_list.extend([cls] * n_here)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main generation pipeline ──────────────────────────────────────────

def generate_all(fast_track: bool = False, hier_subset: list = None) -> None:
    sizes    = FAST_SIZES if fast_track else FULL_SIZES
    tag      = "fast" if fast_track else "full"
    out_root = Path(__file__).parent / tag
    out_root.mkdir(parents=True, exist_ok=True)

    checksums = {}
    specs = {k: v for k, v in HIER_SPECS.items()
             if hier_subset is None or k in hier_subset}

    t_total = time.time()
    for k, spec in specs.items():
        xi      = spec["xi"]
        n_scales = spec["n_scales"]
        name    = spec["name"]
        print(f"  Generating {name} (xi={xi}, {n_scales} scales)...")

        for split, n in sizes.items():
            seed_split = MASTER_SEED + k * 100 + {"train": 0, "val": 1, "ood": 2}[split]
            ood_shift  = 0.5 if split == "ood" else 0.0

            t0 = time.time()
            X, y = generate_hierarchical(
                n_samples=n, xi=xi, n_scales=n_scales,
                ood_shift=ood_shift, seed=seed_split,
            )
            fname = out_root / f"hier{k}_xi{int(xi)}_{split}.npz"
            np.savez_compressed(str(fname), X=X, y=y,
                                xi_data=xi, n_classes=N_CLASSES,
                                split=split, seed=seed_split)

            checksum       = sha256_file(fname)
            checksums[str(fname.name)] = {"sha256": checksum, "n_samples": n,
                                           "xi": xi, "split": split}
            elapsed = time.time() - t0
            print(f"    [{split:5s}] N={n:6d}  d={X.shape[1]}  "
                  f"-> {fname.name}  ({elapsed:.1f}s)")

    # Write checksums
    chk_path = Path(__file__).parent / f"checksums_{tag}.json"
    with open(chk_path, "w") as f:
        json.dump(checksums, f, indent=2)

    total_time = time.time() - t_total
    print(f"\n  Generated {len(specs) * len(sizes)} files in {total_time:.1f}s")
    print(f"  Checksums: {chk_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate hierarchical datasets for RGP experiments."
    )
    p.add_argument("--fast-track", action="store_true",
                   help=f"Small datasets (train={FAST_SIZES['train']}) for fast-track.")
    p.add_argument("--hier", type=int, nargs="+", choices=[1, 2, 3, 4, 5],
                   help="Generate specific Hier-k datasets only (default: all 5).")
    args = p.parse_args()

    print("=== Dataset Generation ===")
    print(f"  Scale: {'fast-track' if args.fast_track else 'full (paper)'}")
    print(f"  Sizes: {FAST_SIZES if args.fast_track else FULL_SIZES}")
    print()
    generate_all(fast_track=args.fast_track, hier_subset=args.hier)
    print("=== Done ===")
