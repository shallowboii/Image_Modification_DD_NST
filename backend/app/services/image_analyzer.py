# backend/app/services/image_analyzer.py
"""
Image analysis utilities for the adaptive pipeline.

All functions operate on HxWx3 float32 numpy arrays with values in [0, 1].
No additional dependencies beyond numpy are required.
"""

import numpy as np


def _rgb_to_luminance(img: np.ndarray) -> np.ndarray:
    """Convert HxWx3 [0,1] image to HxW luminance using BT.601 coefficients."""
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def _rgb_to_hsv_saturation(img: np.ndarray) -> np.ndarray:
    """Return HxW saturation channel of the HSV representation."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    # Saturation is 0 where cmax == 0 (black pixels)
    saturation = np.where(cmax > 1e-6, delta / (cmax + 1e-6), 0.0)
    return saturation.astype(np.float32)


def _high_freq_ratio(luminance: np.ndarray, cutoff_fraction: float = 0.2) -> float:
    """
    Fraction of FFT energy that lives above `cutoff_fraction` of the max frequency.

    A value close to 0 means the image is mostly smooth / low-frequency.
    A value close to 1 means the image is dominated by high-frequency noise.
    """
    fft = np.fft.fft2(luminance)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) ** 2

    h, w = luminance.shape
    cy, cx = h // 2, w // 2
    # Radius beyond which content counts as "high frequency"
    cutoff_radius = cutoff_fraction * min(cy, cx)

    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    total_energy = magnitude.sum() + 1e-10
    high_freq_energy = magnitude[dist > cutoff_radius].sum()
    return float(high_freq_energy / total_energy)


def _edge_density(luminance: np.ndarray) -> float:
    """
    Mean gradient magnitude using a simple finite-difference Sobel approximation.

    Range: [0, ~1]. Higher values = busier / sharper image.
    """
    # Horizontal and vertical gradients via numpy diff (cheap, no scipy needed)
    gx = np.diff(luminance, axis=1, prepend=luminance[:, :1])
    gy = np.diff(luminance, axis=0, prepend=luminance[:1, :])
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad_mag.mean())


def analyze_image(img: np.ndarray) -> dict:
    """
    Compute image quality metrics from a HxWx3 float32 [0,1] numpy array.

    Returns
    -------
    dict with keys:
        high_freq_ratio  : float in [0, 1]  — noise / high-freq artifact level
        edge_density     : float in [0, ~1] — sharpness / visual busyness
        mean_saturation  : float in [0, 1]  — colour richness
        contrast         : float in [0, 0.5]— tonal range (std of luminance)
    """
    img = img.astype(np.float32)
    luminance = _rgb_to_luminance(img)

    return {
        "high_freq_ratio": _high_freq_ratio(luminance),
        "edge_density": _edge_density(luminance),
        "mean_saturation": float(_rgb_to_hsv_saturation(img).mean()),
        "contrast": float(luminance.std()),
    }
