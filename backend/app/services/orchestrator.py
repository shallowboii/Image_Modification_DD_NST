# backend/app/services/orchestrator.py
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .deepdream_service import run_deepdream_once
from .nst_service import run_nst
from .image_analyzer import analyze_image
from .adaptive_controller import AdaptiveController
from utils.utilities import utilities
from utils.utilities.constants import PRESET_FILE

def load_presets():
    print("Loading presets from:", PRESET_FILE)
    if os.path.exists(PRESET_FILE):
        with open(PRESET_FILE) as f:
            return json.load(f)
    return {}


def _upsample_to_original(current_np, orig_w, orig_h, process_w):
    """
    If the original image was larger than the processing width, upsample
    the pipeline result back to original dimensions using Lanczos resampling.
    This keeps processing fast (at ~1080px) while delivering a full-size output.
    """
    out_h, out_w = current_np.shape[:2]
    if orig_w <= process_w:
        # Original was not larger than what we processed at — nothing to upsample.
        return current_np
    pil_img = Image.fromarray((np.clip(current_np, 0, 1) * 255).astype(np.uint8))
    pil_img = pil_img.resize((orig_w, orig_h), Image.LANCZOS)
    print(f"[Resolution] Upsampled {out_w}x{out_h} → {orig_w}x{orig_h} (Lanczos)")
    return np.array(pil_img).astype(np.float32) / 255.0

def run_cyclic_pipeline(input_path, parameters, dump_dir):
    """
    input_path: path to input image
    parameters: dict of params (preset name and/or individual overrides)
    dump_dir: where to save final result
    returns final_output_path
    """
    print("Starting cyclic pipeline with input:", input_path, "preset:", parameters.get("preset", ""), "overrides:", parameters, "dump_dir:", dump_dir)
    presets = load_presets()
    preset = presets.get(parameters.get("preset", ""), {})
    dd_preset = preset.get("dd", {})
    nst_preset = preset.get("nst", {})
    print("Using preset:", parameters.get("preset", ""))
    process_w = parameters.get("img_width") or 1080
    with Image.open(input_path) as _im:
        orig_w, orig_h = _im.size  # (width, height)
    print(f"[Resolution] Original: {orig_w}x{orig_h}  Processing at width: {process_w}")

    img = utilities.load_image(input_path, target_shape=process_w)
    current = img

    # BUG FIX: parameters.get("cycles") is only set when the user sends individual params.
    # When a preset is chosen, cycles lives inside dd_preset — check both.
    num_cycles = parameters.get("cycles") or dd_preset.get("cycles", 3)

    for c in range(num_cycles):
        # prepare dd config
        dd_conf = {
            "model_name": parameters.get("model_name") or "RESNET50",
            "pretrained_weights": parameters.get("pretrained_weights") or "MYIMAGENET",
            "img_width": parameters.get("img_width") or 1080,
            "pyramid_size": parameters.get("pyramid_size") or dd_preset.get("pyramid_size"),
            "pyramid_ratio": parameters.get("pyramid_ratio") or dd_preset.get("pyramid_ratio"),
            "num_gradient_ascent_iterations": parameters.get("num_gradient_ascent_iterations") or dd_preset.get("num_gradient_ascent_iterations"),
            "lr": parameters.get("lr") or dd_preset.get("lr"),
            "dream_blend": parameters.get("dream_blend") or dd_preset.get("dream_blend", 1.0),
            "layers_to_use": [parameters.get("layer")] if parameters.get("layer") else dd_preset.get("layer"),
            "use_noise": parameters.get("use_noise") or False,
            "dump_dir": dump_dir,
            "input_name": os.path.basename(input_path)
        }
        print(f"Cycle {c+1}/{num_cycles} [DeepDream] "
              f"lr={dd_conf['lr']}  iters={dd_conf['num_gradient_ascent_iterations']}"
              f"  pyramid={dd_conf['pyramid_size']}  blend={dd_conf['dream_blend']}")
        current = run_deepdream_once(current, dd_conf)

        # run NST
        nst_conf = {
            "style_path": parameters.get("style") or nst_preset.get("style_path"),
            "num_steps": parameters.get("num_iterations_nst") or nst_preset.get("num_steps", 200),
            "style_weight": parameters.get("style_weight") or nst_preset.get("style_weight", 1e6),
        }
        print(f"Cycle {c+1}/{num_cycles} [NST] "
              f"steps={nst_conf['num_steps']}  style_w={nst_conf['style_weight']:.0e}")
        current = run_nst(current, nst_conf["style_path"], num_steps=nst_conf["num_steps"],
                          style_weight=nst_conf["style_weight"])

    current = _upsample_to_original(current, orig_w, orig_h, process_w)

    # save final
    out_path = utilities.save_and_maybe_display_image(
        {"dump_dir": dump_dir, "should_display": False, "input_name": os.path.basename(input_path)},
        current
    )
    print("Final output saved to:", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _build_base_configs(input_path, parameters, dump_dir):
    """Extract the same base DD/NST configs used by the fixed pipeline."""
    presets = load_presets()
    preset = presets.get(parameters.get("preset", ""), {})
    dd_preset = preset.get("dd", {})
    nst_preset = preset.get("nst", {})

    dd_conf = {
        "model_name": parameters.get("model_name") or "RESNET50",
        "pretrained_weights": parameters.get("pretrained_weights") or "MYIMAGENET",
        "img_width": parameters.get("img_width") or 1080,
        "pyramid_size": parameters.get("pyramid_size") or dd_preset.get("pyramid_size", 4),
        "pyramid_ratio": parameters.get("pyramid_ratio") or dd_preset.get("pyramid_ratio", 1.8),
        "num_gradient_ascent_iterations": parameters.get("num_gradient_ascent_iterations") or dd_preset.get("num_gradient_ascent_iterations", 10),
        "lr": parameters.get("lr") or dd_preset.get("lr", 0.09),
        "dream_blend": parameters.get("dream_blend") or dd_preset.get("dream_blend", 1.0),
        "layers_to_use": [parameters.get("layer")] if parameters.get("layer") else dd_preset.get("layer"),
        "use_noise": parameters.get("use_noise") or False,
        "dump_dir": dump_dir,
        "input_name": os.path.basename(input_path),
    }

    nst_conf = {
        "style_path": parameters.get("style") or nst_preset.get("style_path"),
        "num_steps": parameters.get("num_iterations_nst") or nst_preset.get("num_steps", 200),
        "style_weight": parameters.get("style_weight") or nst_preset.get("style_weight", 1e6),
        "tv_weight": parameters.get("tv_weight") or nst_preset.get("tv_weight", 5e-4),
    }

    return dd_conf, nst_conf


def _save_intermediate(img_np, path):
    """Save a HxWx3 float32 [0,1] numpy array as a JPEG."""
    arr = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=92)


def _plot_metrics(experiment_log, save_path):
    """Save a multi-panel plot of key metrics and parameters over cycles."""
    dd_entries = [e for e in experiment_log if e["stage"] == "deepdream"]
    nst_entries = [e for e in experiment_log if e["stage"] == "nst"]

    cycles = [e["cycle"] for e in dd_entries]
    hf_ratios_dd = [e["metrics_before"].get("high_freq_ratio", 0) for e in dd_entries]
    edge_density_dd = [e["metrics_before"].get("edge_density", 0) for e in dd_entries]
    dream_weights = [e["adjustments"].get("dream_weight", 1.0) for e in dd_entries]
    tv_weights = [e["params_used"].get("tv_weight", 5e-4) for e in nst_entries]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Adaptive Pipeline — Metrics & Parameter Evolution", fontsize=13)

    axes[0, 0].plot(cycles, hf_ratios_dd, marker="o", color="coral")
    axes[0, 0].axhline(0.35, linestyle="--", color="gray", linewidth=0.8, label="threshold")
    axes[0, 0].set_title("High-Freq Ratio (before DD)")
    axes[0, 0].set_xlabel("Cycle")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(cycles, edge_density_dd, marker="o", color="steelblue")
    axes[0, 1].axhline(0.18, linestyle="--", color="gray", linewidth=0.8, label="threshold")
    axes[0, 1].set_title("Edge Density (before DD)")
    axes[0, 1].set_xlabel("Cycle")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(cycles, dream_weights, marker="s", color="seagreen")
    axes[1, 0].set_title("Dream Weight (DD intensity)")
    axes[1, 0].set_xlabel("Cycle")
    axes[1, 0].grid(alpha=0.3)

    nst_cycles = [e["cycle"] for e in nst_entries]
    axes[1, 1].plot(nst_cycles, tv_weights, marker="s", color="mediumpurple")
    axes[1, 1].set_title("NST TV Weight (smoothness)")
    axes[1, 1].set_xlabel("Cycle")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def run_adaptive_cyclic_pipeline(
    input_path: str,
    parameters: dict,
    dump_dir: str,
    adaptive_config: dict | None = None,
    save_intermediates: bool = True,
    plot_metrics: bool = False,
) -> str:
    """
    Adaptive DD → NST cyclic pipeline.

    After every stage the intermediate image is analyzed and the next stage's
    parameters are adjusted by AdaptiveController. The fixed pipeline
    (run_cyclic_pipeline) is left completely unchanged.

    Parameters
    ----------
    input_path       : absolute path to the input content image
    parameters       : same parameter dict used by the fixed pipeline
    dump_dir         : directory for the final result and experiment artifacts
    adaptive_config  : optional dict to override AdaptiveController defaults
    save_intermediates: if True, save one JPEG per (cycle, stage) pair
    plot_metrics     : if True, save a metrics/parameter evolution plot

    Returns
    -------
    Absolute path to the final output image.
    """
    os.makedirs(dump_dir, exist_ok=True)

    # Load presets now to resolve num_cycles (same fix as in run_cyclic_pipeline)
    presets = load_presets()
    preset = presets.get(parameters.get("preset", ""), {})
    num_cycles = parameters.get("cycles") or preset.get("dd", {}).get("cycles", 3)

    print(f"\n[Adaptive] Starting adaptive pipeline  cycles={num_cycles}  input={input_path}")

    base_dd_conf, base_nst_conf = _build_base_configs(input_path, parameters, dump_dir)
    controller = AdaptiveController(base_dd_conf, base_nst_conf, config=adaptive_config)

    process_w = parameters.get("img_width") or 1080
    with Image.open(input_path) as _im:
        orig_w, orig_h = _im.size
    print(f"[Resolution] Original: {orig_w}x{orig_h}  Processing at width: {process_w}")

    img = utilities.load_image(input_path, target_shape=process_w)
    current = img

    for c in range(1, num_cycles + 1):
        print(f"\n[Adaptive] ── Cycle {c}/{num_cycles} ──────────────────────────────")

        # ── DeepDream stage ───────────────────────────────────────────────
        metrics_pre_dd = analyze_image(current)
        print(f"  [DD pre-metrics]  hf={metrics_pre_dd['high_freq_ratio']:.3f}"
              f"  edge={metrics_pre_dd['edge_density']:.3f}"
              f"  sat={metrics_pre_dd['mean_saturation']:.3f}"
              f"  contrast={metrics_pre_dd['contrast']:.3f}")

        dd_conf = controller.adjust_dd(metrics_pre_dd, cycle=c)
        print(f"  [DD params]  lr={dd_conf['lr']}  iters={dd_conf['num_gradient_ascent_iterations']}"
              f"  pyramid={dd_conf['pyramid_size']}  dream_weight={controller.dream_weight:.3f}")

        current = run_deepdream_once(current, dd_conf)

        if save_intermediates:
            path = os.path.join(dump_dir, f"cycle_{c:02d}_deepdream.jpg")
            _save_intermediate(current, path)
            print(f"  ↳ Intermediate saved: {path}")

        # ── NST stage ─────────────────────────────────────────────────────
        metrics_pre_nst = analyze_image(current)
        print(f"  [NST pre-metrics] hf={metrics_pre_nst['high_freq_ratio']:.3f}"
              f"  edge={metrics_pre_nst['edge_density']:.3f}")

        nst_conf = controller.adjust_nst(metrics_pre_nst, cycle=c)
        print(f"  [NST params]  steps={nst_conf['num_steps']}"
              f"  style_w={nst_conf['style_weight']:.0e}"
              f"  tv_w={nst_conf['tv_weight']:.2e}")

        current = run_nst(
            current,
            nst_conf["style_path"],
            num_steps=nst_conf["num_steps"],
            style_weight=nst_conf["style_weight"],
        )

        if save_intermediates:
            path = os.path.join(dump_dir, f"cycle_{c:02d}_nst.jpg")
            _save_intermediate(current, path)
            print(f"  ↳ Intermediate saved: {path}")

    current = _upsample_to_original(current, orig_w, orig_h, process_w)

    # ── Save final result ─────────────────────────────────────────────────
    out_path = utilities.save_and_maybe_display_image(
        {"dump_dir": dump_dir, "should_display": False, "input_name": os.path.basename(input_path)},
        current,
    )
    print(f"\n[Adaptive] Final output saved: {out_path}")

    # ── Save experiment log ───────────────────────────────────────────────
    log_path = os.path.join(dump_dir, "experiment_log.json")
    with open(log_path, "w") as f:
        json.dump(controller.log, f, indent=2, default=str)
    print(f"[Adaptive] Experiment log saved: {log_path}")

    # ── Optional metrics plot ─────────────────────────────────────────────
    if plot_metrics and controller.log:
        plot_path = os.path.join(dump_dir, "metrics_plot.png")
        _plot_metrics(controller.log, plot_path)
        print(f"[Adaptive] Metrics plot saved: {plot_path}")

    return out_path
