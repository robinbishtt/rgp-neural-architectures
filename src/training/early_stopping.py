from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import torch
logger = logging.getLogger(__name__)
@dataclass
class EarlyStoppingState:
    best_value:   float
    counter:      int
    best_epoch:   int
    stopped:      bool
class EarlyStopping:
    def __init__(
        self,
        patience:              int   = 20,
        min_delta:             float = 1e-4,
        mode:                  str   = "maximize",
        restore_best_weights:  bool  = True,
        verbose:               bool  = True,
    ) -> None:
        if mode not in {"minimize", "maximize"}:
            raise ValueError(f"mode must be 'minimize' or 'maximize', got '{mode}'")
        self.patience             = patience
        self.min_delta            = min_delta
        self.mode                 = mode
        self.restore_best_weights = restore_best_weights
        self.verbose              = verbose
        self._counter:     int   = 0
        self._best_epoch:  int   = 0
        self._best_value:  float = float("inf") if mode == "minimize" else float("-inf")
        self._best_state:  Optional[dict] = None
        self._stopped:     bool  = False
    def step(
        self,
        metric: float,
        epoch:  int = 0,
        model:  Optional[torch.nn.Module] = None,
    ) -> bool:
        improved = self._is_improved(metric)
        if improved:
            if self.verbose:
                logger.info(
                    ,
                    epoch, self._best_value, metric,
                )
            self._best_value = metric
            self._best_epoch = epoch
            self._counter    = 0
            if model is not None and self.restore_best_weights:
                import copy
                self._best_state = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1
            if self.verbose:
                logger.info(
                    ,
                    self._counter, self.patience,
                )
            if self._counter >= self.patience:
                self._stopped = True
                logger.info(
                    ,
                    epoch, self._best_epoch, self._best_value,
                )
        return self._stopped
    def restore(self, model: torch.nn.Module) -> torch.nn.Module:
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            logger.info(
                ,
                self._best_epoch,
            )
        return model
    @property
    def stopped(self) -> bool:
        return self._stopped
    @property
    def best_epoch(self) -> int:
        return self._best_epoch
    @property
    def best_value(self) -> float:
        return self._best_value
    def state_dict(self) -> EarlyStoppingState:
        return EarlyStoppingState(
            best_value  = self._best_value,
            counter     = self._counter,
            best_epoch  = self._best_epoch,
            stopped     = self._stopped,
        )
    def load_state_dict(self, state: EarlyStoppingState) -> None:
        self._best_value = state.best_value
        self._counter    = state.counter
        self._best_epoch = state.best_epoch
        self._stopped    = state.stopped
    def _is_improved(self, metric: float) -> bool:
        if self.mode == "minimize":
            return metric < (self._best_value - self.min_delta)
        else:
            return metric > (self._best_value + self.min_delta)