"""
analysis/psychedelic_feature_analysis.py
=========================================
Quantitative analysis comparing original and processed images across
psychedelic-relevant visual metrics.

Scientific basis:
  - Spatial frequency slope (α): Billock & Tsou (2012), Field (1987)
  - Fractal dimension (D):        Klüver form constants, box-counting method
  - Saturation enhancement:       5D-ASC questionnaire subscale (Dittrich 1998)
  - Autocorrelation periodicity:  Ermentrout–Cowan (1979) lattice/grating constants

Usage
-----
Single pair:
    python analysis/psychedelic_feature_analysis.py \\
        --input path/to/original.jpg \\
        --output path/to/processed.jpg \\
        --label "LSD preset"

Batch (multiple presets):
    python analysis/psychedelic_feature_analysis.py \\
        --batch analysis/batch_config.json \\
        --out-dir analysis/results/

Batch config format (batch_config.json):
    [
        {"input": "images/photo.jpg", "output": "results/1/abc_photo.jpg", "label": "LSD"},
        {"input": "images/photo.jpg", "output": "results/2/abc_photo.jpg", "label": "GENTLE"},
        ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # headless — safe on server, change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

_MAX_ANALYSIS_DIM = 1024  # cap resolution for metric computation — values are statistically stable above this

def load_image(path: str) -> np.ndarray:
    """Load image as float32 HxWx3 array in [0, 1], downsampled to _MAX_ANALYSIS_DIM on the long edge."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_ANALYSIS_DIM:
        scale = _MAX_ANALYSIS_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: Spatial Frequency Slope (α)
# ─────────────────────────────────────────────────────────────────────────────

def spatial_frequency_slope(img: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the slope α of the radially averaged power spectrum in log-log space.

    Natural images: α ≈ 2.0  (1/f² spectrum)
    After Deep Dream: α decreases → flatter spectrum → more high-frequency detail
    After drug hallucinations: α reported 1.3–1.7 (more HF energy, complex patterns)

    Returns
    -------
    alpha       : slope of log10(power) vs log10(frequency) line fit
    freqs       : 1D radial frequency bins used for the fit
    power       : corresponding mean power values
    """
    # Convert to luminance
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    h, w = lum.shape
    fft = np.fft.fft2(lum)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Build radial frequency bins
    cy, cx = h // 2, w // 2
    ys, xs = np.mgrid[:h, :w]
    radius = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_bins = np.zeros(max_r)
    counts = np.zeros(max_r)
    for r in range(max_r):
        mask = radius == r
        radial_bins[r] = power[mask].mean()
        counts[r] = mask.sum()

    # Use frequencies from 1 to max_r (skip DC component at 0)
    freqs = np.arange(1, max_r)
    pwr = radial_bins[1:max_r]

    # Fit log-log line (exclude near-DC and very high freq noise)
    fit_start = max(1, max_r // 20)
    fit_end = max_r - max_r // 10
    log_f = np.log10(freqs[fit_start:fit_end])
    log_p = np.log10(pwr[fit_start:fit_end] + 1e-10)

    alpha, _ = np.polyfit(log_f, log_p, 1)

    return float(alpha), freqs, pwr


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: Fractal Dimension (Box-counting)
# ─────────────────────────────────────────────────────────────────────────────

def fractal_dimension(img: np.ndarray, threshold: float = 0.1) -> float:
    """
    Estimate fractal dimension of the image edge structure using box-counting.

    Steps:
      1. Compute gradient magnitude (edge map)
      2. Binarize at `threshold`
      3. Count boxes of decreasing size that contain at least one edge pixel
      4. Fit slope in log(count) vs log(1/box_size) space

    Natural scenes:       D ≈ 1.2 – 1.4
    Klüver form constants: D ≈ 1.5 – 1.8  (more complex, self-similar structure)
    """
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # Gradient magnitude
    gx = np.diff(lum, axis=1, prepend=lum[:, :1])
    gy = np.diff(lum, axis=0, prepend=lum[:1, :])
    edge_map = (np.sqrt(gx ** 2 + gy ** 2) > threshold).astype(np.uint8)

    # Box sizes as powers of 2 up to min dimension
    h, w = edge_map.shape
    max_box = min(h, w) // 4
    box_sizes = [2 ** k for k in range(1, int(np.log2(max_box)) + 1)]

    counts = []
    for bs in box_sizes:
        # Count non-empty boxes
        n_y = h // bs
        n_x = w // bs
        trimmed = edge_map[:n_y * bs, :n_x * bs]
        blocks = trimmed.reshape(n_y, bs, n_x, bs)
        non_empty = (blocks.sum(axis=(1, 3)) > 0).sum()
        counts.append(non_empty)

    counts = np.array(counts, dtype=float)
    sizes = np.array(box_sizes, dtype=float)

    # D = slope of log(count) vs log(1/size)
    valid = counts > 0
    if valid.sum() < 2:
        return float("nan")

    log_inv_s = np.log(1.0 / sizes[valid])
    log_c = np.log(counts[valid])
    D, _ = np.polyfit(log_inv_s, log_c, 1)
    return float(D)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3: Saturation (already in image_analyzer — reimplemented standalone)
# ─────────────────────────────────────────────────────────────────────────────

def mean_saturation(img: np.ndarray) -> float:
    """Mean HSV saturation of the image."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-6, delta / (cmax + 1e-6), 0.0)
    return float(sat.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4: Autocorrelation Periodicity
# ─────────────────────────────────────────────────────────────────────────────

def autocorrelation_peak_ratio(img: np.ndarray, center_exclusion: int = 5) -> float:
    """
    Measure the strength of periodic structure via 2D normalized autocorrelation.

    Lattice / grating hallucinations (Klüver form constants 1 and 3) produce
    strong off-center peaks in the autocorrelation map.

    Returns the ratio: max_off_center_peak / mean_background
    Higher value → stronger periodic / repeating pattern.

    center_exclusion : radius (pixels) around DC peak to ignore
    """
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # Normalized autocorrelation via FFT (Wiener–Khinchin theorem)
    f = np.fft.fft2(lum - lum.mean())
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)

    # Normalize to [0, 1]
    ac_min, ac_max = ac.min(), ac.max()
    if ac_max - ac_min < 1e-10:
        return 1.0
    ac_norm = (ac - ac_min) / (ac_max - ac_min)

    # Mask the central DC peak
    h, w = ac_norm.shape
    cy, cx = h // 2, w // 2
    ys, xs = np.mgrid[:h, :w]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    ac_masked = ac_norm.copy()
    ac_masked[dist < center_exclusion] = 0.0

    peak = ac_masked.max()
    # Mean of non-masked, non-peak region
    background_mask = (dist >= center_exclusion) & (ac_masked < peak * 0.9)
    bg_mean = ac_masked[background_mask].mean() if background_mask.any() else 1e-6

    return float(peak / (bg_mean + 1e-6))


# ─────────────────────────────────────────────────────────────────────────────
# Combined analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(img: np.ndarray) -> dict:
    alpha, freqs, pwr = spatial_frequency_slope(img)
    return {
        "spatial_freq_slope_alpha": round(alpha, 4),
        "fractal_dimension":        round(fractal_dimension(img), 4),
        "mean_saturation":          round(mean_saturation(img), 4),
        "autocorr_peak_ratio":      round(autocorrelation_peak_ratio(img), 4),
        # Store for plotting
        "_freqs": freqs,
        "_pwr":   pwr,
    }


def compare(input_path: str, output_path: str) -> dict:
    """Return before/after metrics and deltas for one image pair."""
    img_in  = load_image(input_path)
    img_out = load_image(output_path)

    m_in  = compute_all_metrics(img_in)
    m_out = compute_all_metrics(img_out)

    scalar_keys = [k for k in m_in if not k.startswith("_")]
    delta = {k: round(m_out[k] - m_in[k], 4) for k in scalar_keys}

    return {"before": m_in, "after": m_out, "delta": delta,
            "input_path": input_path, "output_path": output_path}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "spatial_freq_slope_alpha": "Spatial freq. slope (alpha)",
    "fractal_dimension":        "Fractal dimension D",
    "mean_saturation":          "Mean saturation",
    "autocorr_peak_ratio":      "Autocorr. peak ratio",
}

# Direction expected for psychedelic-like outputs
EXPECTED_DIRECTION = {
    "spatial_freq_slope_alpha": "down (less steep = more HF energy)",
    "fractal_dimension":        "up (more complex patterns)",
    "mean_saturation":          "up (more vivid colors)",
    "autocorr_peak_ratio":      "up (more periodic structure)",
}


def print_report(result: dict, label: str = ""):
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*60}")
    print(f"  Analysis{tag}")
    print(f"  Input : {result['input_path']}")
    print(f"  Output: {result['output_path']}")
    print(f"{'='*60}")
    print(f"  {'Metric':<32} {'Before':>8}  {'After':>8}  {'Delta':>8}  Expected")
    print(f"  {'-'*70}")
    for k, label_text in METRIC_LABELS.items():
        b = result["before"][k]
        a = result["after"][k]
        d = result["delta"][k]
        exp = EXPECTED_DIRECTION[k]
        print(f"  {label_text:<32} {b:>8.4f}  {a:>8.4f}  {d:>+8.4f}  {exp}")
    print()


def save_report_json(results: list[dict], out_path: str):
    # Remove non-serialisable numpy arrays before dumping
    clean = []
    for r in results:
        rc = {}
        for section in ("before", "after", "delta"):
            rc[section] = {k: v for k, v in r[section].items() if not k.startswith("_")}
        rc["input_path"]  = r["input_path"]
        rc["output_path"] = r["output_path"]
        if "label" in r:
            rc["label"] = r["label"]
        clean.append(rc)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
    print(f"[ok] JSON report saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_single(result: dict, label: str = "", out_path: str | None = None):
    """
    4-panel figure for one before/after pair:
      1. Input image
      2. Output image
      3. Power spectrum comparison (log-log)
      4. Metric bar chart (delta)
    """
    img_in  = load_image(result["input_path"])
    img_out = load_image(result["output_path"])

    fig = plt.figure(figsize=(14, 8), facecolor="#1a1a1a")
    fig.suptitle(f"Psychedelic Feature Analysis — {label}", color="white", fontsize=14)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_in   = fig.add_subplot(gs[0, 0])
    ax_out  = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[0, 2])
    ax_bar  = fig.add_subplot(gs[1, :])

    for ax in (ax_in, ax_out, ax_spec, ax_bar):
        ax.set_facecolor("#2a2a2a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    # Images
    ax_in.imshow(img_in)
    ax_in.set_title("Input", color="white", fontsize=10)
    ax_in.axis("off")

    ax_out.imshow(img_out)
    ax_out.set_title("Output", color="white", fontsize=10)
    ax_out.axis("off")

    # Power spectrum
    freqs_in,  pwr_in  = result["before"]["_freqs"], result["before"]["_pwr"]
    freqs_out, pwr_out = result["after"]["_freqs"],  result["after"]["_pwr"]
    ax_spec.loglog(freqs_in,  pwr_in,  color="#aaaaff", linewidth=1.5, label="Input",  alpha=0.8)
    ax_spec.loglog(freqs_out, pwr_out, color="#ff8c8c", linewidth=1.5, label="Output", alpha=0.8)
    ax_spec.set_title("Power Spectrum (log-log)", color="white", fontsize=10)
    ax_spec.set_xlabel("Spatial frequency", color="#aaa", fontsize=8)
    ax_spec.set_ylabel("Power",             color="#aaa", fontsize=8)
    ax_spec.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8)

    # Annotate slope α
    ax_spec.text(0.05, 0.12,
        f"alpha before: {result['before']['spatial_freq_slope_alpha']:.2f}\n"
        f"alpha after:  {result['after']['spatial_freq_slope_alpha']:.2f}",
        transform=ax_spec.transAxes, color="white", fontsize=8,
        verticalalignment="bottom",
        bbox=dict(facecolor="#111", alpha=0.6, edgecolor="none"))

    # Delta bar chart
    metrics = list(METRIC_LABELS.keys())
    labels  = [METRIC_LABELS[m] for m in metrics]
    deltas  = [result["delta"][m] for m in metrics]
    colors  = ["#7c6aff" if d >= 0 else "#ff6a6a" for d in deltas]
    bars = ax_bar.barh(labels, deltas, color=colors, height=0.5)
    ax_bar.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_bar.set_title("Metric Deltas (output - input)", color="white", fontsize=10)
    ax_bar.set_xlabel("Delta value", color="#aaa", fontsize=8)
    for bar, val in zip(bars, deltas):
        ax_bar.text(
            val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center",
            ha="left" if val >= 0 else "right",
            color="white", fontsize=8,
        )
    ax_bar.yaxis.label.set_color("white")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[ok] Plot saved -> {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_batch_comparison(results: list[dict], out_path: str | None = None):
    """
    Multi-preset comparison bar chart.
    Each group shows all 4 metric deltas side-by-side per preset.
    """
    if not results:
        return

    metric_keys = list(METRIC_LABELS.keys())
    n_metrics   = len(metric_keys)
    n_presets   = len(results)

    x       = np.arange(n_metrics)
    width   = 0.8 / n_presets
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, n_presets))

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#1a1a1a")
    ax.set_facecolor("#2a2a2a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    for i, (res, color) in enumerate(zip(results, palette)):
        deltas = [res["delta"][k] for k in metric_keys]
        offset = (i - n_presets / 2 + 0.5) * width
        bars = ax.bar(x + offset, deltas, width * 0.9, label=res.get("label", f"Preset {i+1}"), color=color, alpha=0.85)

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in metric_keys], color="white", fontsize=9)
    ax.set_ylabel("Delta (output - input)", color="#aaa")
    ax.set_title("Psychedelic Feature Deltas by Preset", color="white", fontsize=13)
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[ok] Batch comparison plot saved -> {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative psychedelic feature analysis for Deep Dream + NST outputs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  type=str, help="Path to original image")
    group.add_argument("--batch",  type=str, help="Path to batch config JSON")

    parser.add_argument("--output",  type=str, help="Path to processed image (single mode)")
    parser.add_argument("--label",   type=str, default="", help="Preset/run label")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for plots/reports")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        # ── Single mode ──────────────────────────────────────────────────────
        if not args.output:
            print("Error: --output is required in single mode.", file=sys.stderr)
            sys.exit(1)

        print(f"Analysing: {args.input}  ->  {args.output}")
        result = compare(args.input, args.output)
        result["label"] = args.label
        print_report(result, label=args.label)

        if not args.no_plot:
            plot_single(result, label=args.label,
                        out_path=str(out_dir / f"analysis_{args.label or 'result'}.png"))

        save_report_json([result], str(out_dir / f"report_{args.label or 'result'}.json"))

    else:
        # ── Batch mode ───────────────────────────────────────────────────────
        with open(args.batch, encoding="utf-8") as fh:
            entries = json.load(fh)

        all_results = []
        for entry in entries:
            label = entry.get("label", "")
            print(f"\nProcessing: {label}")
            result = compare(entry["input"], entry["output"])
            result["label"] = label
            print_report(result, label=label)
            all_results.append(result)

        if not args.no_plot:
            # Individual plots
            for res in all_results:
                lbl = res.get("label", "unknown").replace(" ", "_")
                plot_single(res, label=res.get("label", ""),
                            out_path=str(out_dir / f"analysis_{lbl}.png"))
            # Combined comparison
            plot_batch_comparison(all_results,
                                  out_path=str(out_dir / "preset_comparison.png"))

        save_report_json(all_results, str(out_dir / "batch_report.json"))


if __name__ == "__main__":
    main()
