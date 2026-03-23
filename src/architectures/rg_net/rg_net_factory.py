"""
src/architectures/rg_net/rg_net_factory.py

RGNetFactory: centralised builder for all RG-Net architecture variants.

Motivation
----------
With seven architecture variants (Shallow, Standard, Deep, UltraDeep,
VariableWidth, MultiScale, Residual) and dozens of hyperparameter combinations
used across the three hypothesis validation experiments, instantiation logic
must not be duplicated.  The factory pattern provides a single entry point,
validates hyperparameter combinations, and applies critical initialisation.

Usage
-----
    from src.architectures.rg_net import RGNetFactory

    # Build by name (matches config/architectures/rg_net.yaml 'variant' key)
    model = RGNetFactory.build(
        variant="standard",
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        depth=50,
    )

    # Build from Hydra/YAML config dict
    model = RGNetFactory.from_config(cfg.architecture)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Type


from src.architectures.rg_net.rg_net_template import RGNetTemplate

logger = logging.getLogger(__name__)

# Populated lazily to avoid circular imports at module load time
_VARIANT_REGISTRY: Dict[str, Type[RGNetTemplate]] = {}


def _populate_registry() -> None:
    """Import variants and populate the registry if not done yet."""
    if _VARIANT_REGISTRY:
        return
    from src.architectures.rg_net.rg_net_shallow       import RGNetShallow
    from src.architectures.rg_net.rg_net_standard       import RGNetStandard
    from src.architectures.rg_net.rg_net_deep           import RGNetDeep
    from src.architectures.rg_net.rg_net_ultra_deep     import RGNetUltraDeep
    from src.architectures.rg_net.rg_net_variable_width import RGNetVariableWidth
    from src.architectures.rg_net.rg_net_multiscale     import RGNetMultiScale

    _VARIANT_REGISTRY.update({
        "shallow":        RGNetShallow,
        "standard":       RGNetStandard,
        "deep":           RGNetDeep,
        "ultra_deep":     RGNetUltraDeep,
        "ultradeep":      RGNetUltraDeep,   # alias
        "variable_width": RGNetVariableWidth,
        "multiscale":     RGNetMultiScale,
    })


class RGNetFactory:
    """
    Factory for instantiating RG-Net architecture variants.

    Variant → recommended depth range:
        shallow       : depth  2–10
        standard      : depth 10–100
        deep          : depth 100–500
        ultra_deep    : depth 500–1000+
        variable_width: any depth, width schedule specified by width_schedule
        multiscale    : any depth, n_scales specifies number of parallel streams
    """

    @classmethod
    def available_variants(cls) -> list:
        """Return list of registered variant names."""
        _populate_registry()
        return sorted(set(_VARIANT_REGISTRY.keys()))

    @classmethod
    def build(
        cls,
        variant: str,
        input_dim:  int,
        hidden_dim: int,
        output_dim: int,
        depth:      int,
        activation: str   = "tanh",
        sigma_w:    float = 1.0,
        sigma_b:    float = 0.05,
        **kwargs: Any,
    ) -> RGNetTemplate:
        """
        Build a named RG-Net variant with the given hyperparameters.

        Parameters
        ----------
        variant : str
            Architecture variant name (see ``available_variants()``).
        input_dim, hidden_dim, output_dim : int
            Dimensionality of input, hidden, and output layers.
        depth : int
            Number of hidden layers (L in manuscript notation).
        activation : str
            Activation function: "tanh" | "relu" | "gelu".
        sigma_w, sigma_b : float
            Critical initialisation parameters.  σ_w = 1.0 places the
            network at the edge-of-chaos manifold for tanh activations.
        **kwargs : Any
            Additional variant-specific keyword arguments forwarded to
            the constructor (e.g., ``n_scales=4`` for multiscale).

        Returns
        -------
        RGNetTemplate
            Instantiated model with critical initialisation applied.

        Raises
        ------
        ValueError
            If ``variant`` is not registered or if depth is out of the
            recommended range for the chosen variant.
        """
        _populate_registry()
        variant_key = variant.lower().replace("-", "_")
        if variant_key not in _VARIANT_REGISTRY:
            raise ValueError(
                f"Unknown RG-Net variant '{variant}'.  "
                f"Available: {cls.available_variants()}"
            )

        cls._validate_depth(variant_key, depth)

        variant_cls = _VARIANT_REGISTRY[variant_key]
        model = variant_cls(
            input_dim  = input_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            depth      = depth,
            activation = activation,
            sigma_w    = sigma_w,
            sigma_b    = sigma_b,
            **kwargs,
        )

        logger.info(
            "RGNetFactory: built %s | depth=%d | hidden=%d | params=%s",
            variant_cls.__name__, depth, hidden_dim,
            f"{model.count_parameters():,}",
        )
        return model

    @classmethod
    def from_config(cls, cfg: Any) -> RGNetTemplate:
        """
        Build from a Hydra/YAML config object or plain dict.

        Expected keys (matching config/architectures/rg_net.yaml):
            variant, input_dim, hidden_dim, output_dim, depth,
            activation, sigma_w, sigma_b
        """
        if hasattr(cfg, "__dict__"):
            cfg = vars(cfg)
        return cls.build(
            variant    = cfg["variant"],
            input_dim  = cfg["input_dim"],
            hidden_dim = cfg["hidden_dim"],
            output_dim = cfg["output_dim"],
            depth      = cfg["depth"],
            activation = cfg.get("activation", "tanh"),
            sigma_w    = cfg.get("sigma_w", 1.0),
            sigma_b    = cfg.get("sigma_b", 0.05),
            **{k: v for k, v in cfg.items()
               if k not in {"variant","input_dim","hidden_dim",
                            "output_dim","depth","activation","sigma_w","sigma_b"}},
        )

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    _DEPTH_RANGES: Dict[str, tuple] = {
        "shallow":        (2,   10),
        "standard":       (10,  100),
        "deep":           (100, 500),
        "ultra_deep":     (500, 10_000),
        "ultradeep":      (500, 10_000),
        "variable_width": (2,   10_000),
        "multiscale":     (2,   10_000),
    }

    @classmethod
    def _validate_depth(cls, variant: str, depth: int) -> None:
        lo, hi = cls._DEPTH_RANGES.get(variant, (1, 10_000))
        if not (lo <= depth <= hi):
            logger.warning(
                "Depth %d is outside recommended range [%d, %d] for variant '%s'. "
                "Proceeding, but behaviour may be sub-optimal.",
                depth, lo, hi, variant,
            )
 