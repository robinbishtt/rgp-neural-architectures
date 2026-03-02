"""
src/architectures/baselines — Baseline architectures for H3 comparative evaluation.

All seven baseline architectures are exported here. Tests and experiments
import from this package rather than individual files.
"""
from src.architectures.baselines.resnet_baseline       import ResNetBaseline
from src.architectures.baselines.densenet_baseline     import DenseNetBaseline
from src.architectures.baselines.mlp_baseline          import MLPBaseline
from src.architectures.baselines.vgg_baseline          import VGGBaseline
from src.architectures.baselines.transformer_baseline  import TransformerBaseline
from src.architectures.baselines.attention_baseline    import AttentionBaseline
from src.architectures.baselines.inception_baseline    import InceptionBaseline

__all__ = [
    "ResNetBaseline",
    "DenseNetBaseline",
    "MLPBaseline",
    "VGGBaseline",
    "TransformerBaseline",
    "AttentionBaseline",
    "InceptionBaseline",
]
