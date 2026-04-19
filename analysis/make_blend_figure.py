"""Generate deepdream_blend.png — shows the effect of dream_blend.

Runs DeepDream once on church.jpg at dream_blend=1.0, then linearly
blends with the original to produce the 0.5 panel. No re-running needed.

Panels: Eredeti  |  dream_blend = 0.5  |  dream_blend = 1.0
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.deepdream_service import run_deepdream_once
from utils.utilities import utilities

INPUT       = ROOT / "utils" / "data" / "in-images" / "chruch.jpg"
OUT_PATH    = ROOT / "Thesis" / "DeepDream_NST_szakdolgozat" / "Pictures" / "deepdream_blend.png"
PROCESS_W   = 900
PANEL_GAP   = 12
LABEL_H     = 48
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def _font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _np_to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def main():
    print(f"[blend] Loading {INPUT}")
    img = utilities.load_image(str(INPUT), target_shape=PROCESS_W)
    original = img.copy()

    dd_config = {
        "model_name":                     "RESNET50",
        "pretrained_weights":             "MYIMAGENET",
        "img_width":                      PROCESS_W,
        "pyramid_size":                   4,
        "pyramid_ratio":                  1.8,
        "num_gradient_ascent_iterations": 15,
        "lr":                             0.09,
        "dream_blend":                    1.0,
        "layers_to_use":                  ["layer3"],
        "use_noise":                      False,
        "dump_dir":                       str(ROOT / "analysis" / "results"),
        "input_name":                     "chruch_blend_full.jpg",
    }

    print("[blend] Running DeepDream (dream_blend=1.0)...")
    full = run_deepdream_once(img, dd_config)

    half = 0.5 * full + 0.5 * original
    half = np.clip(half, 0, 1).astype(np.float32)

    panels = [
        ("Eredeti",              original),
        ("dream_blend = 0{,}5",  half),
        ("dream_blend = 1{,}0",  full),
    ]

    pil_imgs = [_np_to_pil(a) for _, a in panels]
    w, h = pil_imgs[0].size

    total_w = 3 * w + 2 * PANEL_GAP
    total_h = h + LABEL_H
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(22)

    x = 0
    for (label, _), pil in zip(panels, pil_imgs):
        canvas.paste(pil, (x, LABEL_H))
        clean = label.replace("{,}", ",")
        bbox = draw.textbbox((0, 0), clean, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, 10), clean, fill="black", font=font)
        x += w + PANEL_GAP

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH, "PNG", quality=95)
    print(f"[blend] Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
