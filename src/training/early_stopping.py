"""
src/training/early_stopping.py

EarlyStopping: patience-based early stopping for RGP training runs.

Motivation
----------
Ultra-deep networks (L=500-1000) can train for 48-72 hours on A100 GPUs.
Early stopping prevents wasted compute when the network has converged or
diverged.  For the scaling law experiments (H2), we require the *minimum*
depth to reach a target accuracy — early stopping must therefore track
*accuracy*, not loss, as the convergence criterion.

Two modes are supported:
    ``minimize``  — stops when metric stops decreasing (e.g., validation loss)
    ``maximize``  — stops when metric stops increasing (e.g., validation accuracy)

Integration with H2 validation:
    The H2 experiment records the epoch at which early stopping fires as
    the proxy for "training time to convergence", distinct from L_min which
    measures the *depth* at which a fixed-epoch-budget reaches target accuracy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingState:
    """Snapshot of early stopping state suitable for checkpoint serialisation."""
    best_value:   float
    counter:      int
    best_epoch:   int
    stopped:      bool


class EarlyStopping:
    """
    Early stopping with configurable patience, delta, and mode.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training stops.
        For H2 experiments use patience=20 (full) or patience=5 (fast-track).
    min_delta : float
        Minimum absolute change in monitored metric to qualify as improvement.
        Prevents stopping on noise.  Default 1e-4.
    mode : str
        ``"minimize"`` (stop when val_loss plateaus) or
        ``"maximize"`` (stop when val_acc plateaus).
    restore_best_weights : bool
        If True, roll back model weights to the best observed checkpoint
        when stopping is triggered.
    verbose : bool
        Log an INFO message on each patience update.

    Example
    -------
    ::
        es = EarlyStopping(patience=20, mode="maximize")
        for epoch in range(max_epochs):
            val_acc = evaluate(model, val_loader)
            if es.step(val_acc, model=model):
                break
        model = es.restore(model)  # load best weights
    """

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

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(
        self,
        metric: float,
        epoch:  int = 0,
        model:  Optional[torch.nn.Module] = None,
    ) -> bool:
        """
        Update internal state with new metric value.

        Parameters
        ----------
        metric : float
            Current epoch metric (val_loss or val_acc).
        epoch : int
            Current epoch index (for logging).
        model : nn.Module, optional
            If provided and restore_best_weights is True, best weights
            are saved whenever the metric improves.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        improved = self._is_improved(metric)

        if improved:
            if self.verbose:
                logger.info(
                    "EarlyStopping: improvement at epoch %d "
                    "(%.6f → %.6f). Resetting counter.",
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
                    "EarlyStopping: no improvement (counter %d/%d).",
                    self._counter, self.patience,
                )
            if self._counter >= self.patience:
                self._stopped = True
                logger.info(
                    "EarlyStopping: triggered at epoch %d (best epoch %d, value %.6f).",
                    epoch, self._best_epoch, self._best_value,
                )

        return self._stopped

    def restore(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Load best saved weights into model.

        No-op if restore_best_weights=False or model was never passed
        to step().
        """
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            logger.info(
                "EarlyStopping: restored best weights from epoch %d.",
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

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> EarlyStoppingState:
        """Return serialisable state for checkpoint inclusion."""
        return EarlyStoppingState(
            best_value  = self._best_value,
            counter     = self._counter,
            best_epoch  = self._best_epoch,
            stopped     = self._stopped,
        )

    def load_state_dict(self, state: EarlyStoppingState) -> None:
        """Restore from checkpoint."""
        self._best_value = state.best_value
        self._counter    = state.counter
        self._best_epoch = state.best_epoch
        self._stopped    = state.stopped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_improved(self, metric: float) -> bool:
        if self.mode == "minimize":
            return metric < (self._best_value - self.min_delta)
        else:
            return metric > (self._best_value + self.min_delta)
