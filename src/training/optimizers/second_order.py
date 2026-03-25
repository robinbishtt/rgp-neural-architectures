from __future__ import annotations
import torch
def build_lbfgs(params, lr: float = 1.0, max_iter: int = 20,
                history_size: int = 100):
    return torch.optim.LBFGS(params, lr=lr, max_iter=max_iter,
                              history_size=history_size, line_search_fn="strong_wolfe")