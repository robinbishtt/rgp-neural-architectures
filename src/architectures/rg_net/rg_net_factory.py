from __future__ import annotations
import logging
from typing import Any, Dict, Type
from src.architectures.rg_net.rg_net_template import RGNetTemplate
logger = logging.getLogger(__name__)
_VARIANT_REGISTRY: Dict[str, Type[RGNetTemplate]] = {}
def _populate_registry() -> None:
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
        "ultra-deep":     RGNetUltraDeep,
        "variable_width": RGNetVariableWidth,
        "multiscale":     RGNetMultiScale,
    })
class RGNetFactory:
    @classmethod
    def available_variants(cls) -> list:
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
        _populate_registry()
        variant_key = variant.lower().replace("-", "_")
        if variant_key not in _VARIANT_REGISTRY:
            raise ValueError(
                f"Unknown RG-Net variant '{variant}'.  "
                f"Available: {cls.available_variants()}"
            )
        cls._validate_depth(variant_key, depth)
        variant_cls = _VARIANT_REGISTRY[variant_key]
        if variant_key == "variable_width":
            model = variant_cls(
                in_features=input_dim,
                width_schedule=kwargs.pop("width_schedule", [hidden_dim] * depth),
                n_classes=output_dim,
                activation=activation,
                **kwargs,
            )
        elif variant_key == "multiscale":
            model = variant_cls(
                in_features=input_dim,
                n_classes=output_dim,
                depth=depth,
                width=hidden_dim,
                **kwargs,
            )
        else:
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
        params = model.count_parameters() if hasattr(model, "count_parameters") else sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            "Built %s | depth=%d | width=%d | params=%s",
            variant_cls.__name__, depth, hidden_dim,
            f"{params:,}",
        )
        return model
    @classmethod
    def from_config(cls, cfg: Any) -> RGNetTemplate:
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
    _DEPTH_RANGES: Dict[str, tuple] = {
        "shallow":        (10,  50),
        "standard":       (10,  100),
        "deep":           (100, 500),
        "ultra_deep":     (500, 10_000),
        "ultra-deep":     (500, 10_000),
        "variable_width": (2,   10_000),
        "multiscale":     (2,   10_000),
    }
    @classmethod
    def _validate_depth(cls, variant: str, depth: int) -> None:
        lo, hi = cls._DEPTH_RANGES.get(variant, (1, 10_000))
        if not (lo <= depth <= hi):
            logger.warning(
                "Depth %d out of recommended range [%d, %d] for variant '%s'",
                depth, lo, hi, variant,
            )
