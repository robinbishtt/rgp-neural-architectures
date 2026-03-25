from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader, IterableDataset
class StreamingHDF5Dataset(IterableDataset):
    def __init__(self, hdf5_path: str, chunk_size: int = 1024) -> None:
        super().__init__()
        self.path       = Path(hdf5_path)
        self.chunk_size = chunk_size
    def __iter__(self):
        import h5py
        with h5py.File(self.path, "r") as f:
            data   = f["data"]
            labels = f["labels"]
            n      = len(data)
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                x = torch.tensor(data[start:end], dtype=torch.float32)
                y = torch.tensor(labels[start:end], dtype=torch.long)
                for i in range(len(x)):
                    yield x[i], y[i]
class StreamingDataLoader(DataLoader):
    def __init__(self, hdf5_path: str, batch_size: int = 256,
                 num_workers: int = 4, **kwargs) -> None:
        dataset = StreamingHDF5Dataset(hdf5_path)
        super().__init__(dataset, batch_size=batch_size,
                         num_workers=num_workers, **kwargs)