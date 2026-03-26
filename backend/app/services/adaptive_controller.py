# backend/app/services/adaptive_controller.py
"""
Adaptive controller for the cyclic DD+NST pipeline.

AdaptiveController holds mutable state across cycles and adjusts
DeepDream / NST parameters based on image metrics computed by image_analyzer.
"""

import copy
import math


# ─────────────────────────────────────────────────────────────────────────────
# Default adaptive configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ADAPTIVE_CONFIG = {
    # dream_weight decays each cycle; DD lr and iteration count are scaled by it
    "dream_weight_init": 1.0,
    "dream_weight_decay": 0.85,

    # If high_freq_ratio exceeds this, DD lr is reduced and NST tv_weight boosted
    "hf_noise_threshold": 0.35,
    # Fractional reduction applied to DD lr when threshold is exceeded
    "hf_lr_reduction": 0.30,
    # Fractional boost applied to NST tv_weight when threshold is exceeded
    "hf_tv_boost": 0.50,

    # If edge_density exceeds this, DD pyramid_size is reduced by 1
    "edge_density_threshold": 0.18,

    # Hard bounds for DD lr
    "lr_min": 0.02,
    "lr_max": 0.15,

    # Hard bounds for NST tv_weight
    "tv_weight_min": 1e-4,
    "tv_weight_max": 5e-3,
}


# ── Tiny helper to avoid importing numpy just for a clamp ────────────────────

class _Clamp:
    @staticmethod
    def clip_scalar(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveController:
    """
    Maintains per-cycle state and returns adjusted configs for each stage.

    Parameters
    ----------
    base_dd_conf  : dict — baseline DeepDream config (as built by the orchestrator)
    base_nst_conf : dict — baseline NST config (as built by the orchestrator)
    config        : dict — adaptive hyper-parameters; any key from
                    DEFAULT_ADAPTIVE_CONFIG can be overridden here
    """

    def __init__(self, base_dd_conf: dict, base_nst_conf: dict, config: dict | None = None):
        self.base_dd_conf = base_dd_conf
        self.base_nst_conf = base_nst_conf

        # Merge caller overrides on top of defaults
        self.cfg = {**DEFAULT_ADAPTIVE_CONFIG, **(config or {})}

        # Running dream_weight — starts at init value
        self.dream_weight: float = self.cfg["dream_weight_init"]

        # Experiment log: each entry is one (cycle, stage) record
        self.log: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def adjust_dd(self, metrics: dict, cycle: int) -> dict:
        """
        Return an adjusted DeepDream config for the given cycle.

        Steps:
        1. Decay dream_weight
        2. Scale lr and num_gradient_ascent_iterations by dream_weight
        3. If high_freq_ratio is too high → cut lr further
        4. If edge_density is too high    → reduce pyramid_size
        5. Clamp lr to [lr_min, lr_max]
        """
        # Step 1: decay
        self.dream_weight = (
            self.cfg["dream_weight_init"]
            * (self.cfg["dream_weight_decay"] ** (cycle - 1))
        )

        dd = copy.deepcopy(self.base_dd_conf)
        adjustments = {"dream_weight": round(self.dream_weight, 4)}

        # Step 2: scale by dream_weight
        base_lr = self.base_dd_conf.get("lr", 0.09)
        lr = base_lr * self.dream_weight

        base_iters = self.base_dd_conf.get("num_gradient_ascent_iterations", 10)
        iters = max(1, round(base_iters * self.dream_weight))

        # Step 3: noise penalty
        hf = metrics.get("high_freq_ratio", 0.0)
        if hf > self.cfg["hf_noise_threshold"]:
            reduction = self.cfg["hf_lr_reduction"]
            lr *= (1.0 - reduction)
            adjustments["hf_lr_reduction_applied"] = True
            adjustments["high_freq_ratio"] = round(hf, 4)

        # Step 4: edge density penalty
        ed = metrics.get("edge_density", 0.0)
        pyramid_size = dd.get("pyramid_size", 4)
        if ed > self.cfg["edge_density_threshold"] and pyramid_size > 1:
            pyramid_size = max(1, pyramid_size - 1)
            adjustments["pyramid_size_reduced_to"] = pyramid_size
            adjustments["edge_density"] = round(ed, 4)

        # Step 5: clamp
        lr = float(_Clamp.clip_scalar(lr, self.cfg["lr_min"], self.cfg["lr_max"]))

        dd["lr"] = round(lr, 4)
        dd["num_gradient_ascent_iterations"] = iters
        dd["pyramid_size"] = pyramid_size
        adjustments["lr"] = dd["lr"]
        adjustments["num_gradient_ascent_iterations"] = iters

        self.log.append({
            "cycle": cycle,
            "stage": "deepdream",
            "metrics_before": metrics,
            "adjustments": adjustments,
            "params_used": {k: dd[k] for k in ("lr", "num_gradient_ascent_iterations", "pyramid_size")},
        })

        return dd

    def adjust_nst(self, metrics: dict, cycle: int) -> dict:
        """
        Return an adjusted NST config for the given cycle.

        If high_freq_ratio is high → boost tv_weight to smooth the result.
        """
        nst = copy.deepcopy(self.base_nst_conf)
        adjustments: dict = {}

        base_tv = self.base_nst_conf.get("tv_weight", 5e-4)
        tv_weight = base_tv

        hf = metrics.get("high_freq_ratio", 0.0)
        if hf > self.cfg["hf_noise_threshold"]:
            tv_weight = base_tv * (1.0 + self.cfg["hf_tv_boost"])
            adjustments["tv_weight_boosted"] = True
            adjustments["high_freq_ratio"] = round(hf, 4)

        # Clamp
        tv_weight = max(self.cfg["tv_weight_min"], min(self.cfg["tv_weight_max"], tv_weight))
        nst["tv_weight"] = tv_weight
        adjustments["tv_weight"] = tv_weight

        self.log.append({
            "cycle": cycle,
            "stage": "nst",
            "metrics_before": metrics,
            "adjustments": adjustments,
            "params_used": {
                "num_steps": nst.get("num_steps"),
                "style_weight": nst.get("style_weight"),
                "tv_weight": tv_weight,
            },
        })

        return nst
