from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)
_CLASS_HIERARCHY: Dict[int, Tuple[int, int]] = {
    **{i: (i // 5, i // 20) for i in range(100)}  
}
class ImageNetHierarchy(Dataset):
    N_CLASSES          = 100
    N_SUPERCLASSES     = 20
    N_MID_CLASSES      = 50
    IMAGE_SIZE         = 224
    MEAN               = (0.485, 0.456, 0.406)
    STD                = (0.229, 0.224, 0.225)
    def __init__(
        self,
        root:                  Optional[Path] = None,
        split:                 str = "train",
        transform:             Optional[Callable] = None,
        correlation_level:     int = 3,
        ood_shift_magnitude:   float = 0.0,
        use_synthetic:         bool = True,
        n_samples_synthetic:   int  = 10_000,
        seed:                  int  = 42,
    ) -> None:
        self.root                = Path(root) if root else None
        self.split               = split
        self.correlation_level   = correlation_level
        self.ood_shift_magnitude = ood_shift_magnitude
        self.use_synthetic       = use_synthetic or (root is None)
        self.seed                = seed
        if self.use_synthetic:
            self._samples, self._labels = self._generate_synthetic(
                n_samples   = n_samples_synthetic,
                input_dim   = 128,  
                seed        = seed,
            )
        else:
            self._samples, self._labels = self._load_imagenet(split)
        if ood_shift_magnitude > 0:
            self._apply_ood_shift(ood_shift_magnitude, seed=seed + 1000)
        self.transform = transform or self._default_transform()
        self._effective_labels = self._remap_to_level(
            self._labels, correlation_level
        )
        logger.info(
            ,
            split, len(self._samples), correlation_level, self.use_synthetic,
        )
    def __len__(self) -> int:
        return len(self._samples)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self._samples[idx]
        y = int(self._effective_labels[idx])
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if self.transform is not None and x.dim() == 3:  
            x = self.transform(x)
        return x, y
    def get_hierarchy(self) -> Dict[int, Tuple[int, int]]:
        return dict(_CLASS_HIERARCHY)
    def get_correlation_length_gt(self) -> List[float]:
        return [16.0, 4.0, 1.0]  
    def _generate_synthetic(
        self,
        n_samples: int,
        input_dim: int,
        seed:      int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_per_class = n_samples // self.N_CLASSES
        sigma_super = 3.0
        sigma_mid   = 1.5
        sigma_fine  = 0.8
        sigma_noise = 0.3
        super_centroids = rng.normal(0, sigma_super, (self.N_SUPERCLASSES, input_dim))
        mid_centroids   = rng.normal(0, sigma_mid,   (self.N_MID_CLASSES,  input_dim))
        fine_centroids  = rng.normal(0, sigma_fine,  (self.N_CLASSES,      input_dim))
        samples = []
        labels  = []
        for cls in range(self.N_CLASSES):
            mid_cls, super_cls = _CLASS_HIERARCHY[cls]
            mid_cls   = mid_cls   % self.N_MID_CLASSES
            super_cls = super_cls % self.N_SUPERCLASSES
            mu = (super_centroids[super_cls] + mid_centroids[mid_cls] +
                  fine_centroids[cls])
            x  = rng.normal(mu, sigma_noise, (n_per_class, input_dim))
            samples.append(x)
            labels.extend([cls] * n_per_class)
        return np.concatenate(samples, axis=0).astype(np.float32), np.array(labels)
    def _load_imagenet(
        self,
        split: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from torchvision.datasets import ImageFolder
            split_dir = self.root / ("train" if split == "train" else "val")
            if not split_dir.exists():
                raise FileNotFoundError(str(split_dir))
            dataset = ImageFolder(root=str(split_dir))
            return dataset.samples, np.array([s[1] for s in dataset.samples])
        except (ImportError, FileNotFoundError) as exc:
            logger.warning(
                , exc
            )
            return self._generate_synthetic(
                n_samples=5000, input_dim=128, seed=self.seed,
            )
    def _remap_to_level(
        self,
        labels: np.ndarray,
        level:  int,
    ) -> np.ndarray:
        if level == 3 or level not in {1, 2}:
            return labels
        idx = 1 if level == 2 else 0  
        return np.array([_CLASS_HIERARCHY.get(int(l), (0, 0))[idx] for l in labels])
    def _apply_ood_shift(self, magnitude: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n   = len(self._labels)
        n_shift = int(magnitude * n)
        idx = rng.choice(n, size=n_shift, replace=False)
        self._labels[idx] = rng.integers(0, self.N_CLASSES, size=n_shift)
    @staticmethod
    def _default_transform() -> Optional[Callable]:
        try:
            import torchvision.transforms as T
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=ImageNetHierarchy.MEAN, std=ImageNetHierarchy.STD),
            ])
        except ImportError:
            return None