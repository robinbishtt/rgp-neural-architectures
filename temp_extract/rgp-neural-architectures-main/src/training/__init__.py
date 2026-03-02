"""
src/training — Training loop, schedulers, optimizers, and curriculum utilities.

Primary entry points:
    Trainer              — standard single-GPU training loop
    CurriculumTrainer    — progressive curriculum with difficulty scheduling
    TrainingMonitor      — real-time anomaly detection (NaN, explosion, plateau)
    build_scheduler      — factory for all LR scheduler types
    training_utils       — shared utility functions (accuracy, gradient_norm, etc.)

Optimizers (all in src/training/optimizers/):
    LinearWarmupScheduler      — linear warmup + cosine/linear/constant decay
    CosineAnnealingWithRestarts — SGDR warm restarts (Loshchilov & Hutter 2017)
    DiagonalNaturalGradient    — diagonal Fisher preconditioned gradient
    FisherOptimizer            — K-FAC block-diagonal natural gradient
    LearningRateFinder         — Smith (2017) LR range test
"""
from src.training.trainer               import Trainer, TrainingConfig, TrainingResult
from src.training.training_monitor      import TrainingMonitor, TrainingEvent
from src.training.learning_rate_scheduler import build_scheduler
from src.training.curriculum_trainer    import CurriculumTrainer
from src.training.training_utils        import (
    compute_accuracy,
    count_parameters,
    gradient_norm,
    clip_gradients,
    freeze_layers,
    cosine_similarity_layers,
)
from src.training.optimizers.warmup_scheduler  import LinearWarmupScheduler
from src.training.optimizers.cosine_annealing  import CosineAnnealingWithRestarts
from src.training.optimizers.natural_gradient  import DiagonalNaturalGradient
from src.training.optimizers.fisher_optimizer  import FisherOptimizer
from src.training.optimizers.learning_rate_finder import LearningRateFinder

__all__ = [
    # Core trainers
    "Trainer", "TrainingConfig", "TrainingResult",
    "CurriculumTrainer",
    # Monitoring
    "TrainingMonitor", "TrainingEvent",
    # LR scheduling
    "build_scheduler",
    "LinearWarmupScheduler",
    "CosineAnnealingWithRestarts",
    # Optimizers
    "DiagonalNaturalGradient",
    "FisherOptimizer",
    "LearningRateFinder",
    # Utilities
    "compute_accuracy",
    "count_parameters",
    "gradient_norm",
    "clip_gradients",
    "freeze_layers",
    "cosine_similarity_layers",
]
