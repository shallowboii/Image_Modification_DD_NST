"""Generate nst_triplet.png — tartalom / stílus / NST kimenet hármas.

Runs a single pure-NST pass on chruch.jpg with kaleidoscope.jpg as style,
then lays the three images out side-by-side with labels.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.utilities import utilities
from utils.neural_style_transfer import run_style_transfer_from_numpy

CONTENT     = ROOT / "utils" / "data" / "in-images" / "chruch.jpg"
STYLE       = "kaleidoscope.jpg"
STYLE_FULL  = ROOT / "utils" / "data" / "in-images" / "styles" / STYLE
OUT_PATH    = ROOT / "Thesis" / "DeepDream_NST_szakdolgozat" / "Pictures" / "nst_triplet.png"
PROCESS_W   = 900
NST_STEPS   = 200
STYLE_W     = 1e6
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


def _pil_resize_to_height(pil, target_h):
    w, h = pil.size
    new_w = int(w * target_h / h)
    return pil.resize((new_w, target_h), Image.LANCZOS)


def main():
    print(f"[nst] Loading content: {CONTENT}")
    content_np = utilities.load_image(str(CONTENT), target_shape=PROCESS_W)

    print(f"[nst] Running NST ({NST_STEPS} steps, style_weight={STYLE_W:.0e})...")
    nst_out = run_style_transfer_from_numpy(
        content_np,
        style_path=STYLE,
        num_steps=NST_STEPS,
        style_weight=STYLE_W,
        content_weight=5.0,
        tv_weight=5e-4,
        imsize=PROCESS_W,
    )

    content_pil = _np_to_pil(content_np)
    out_pil     = _np_to_pil(nst_out)
    style_pil   = Image.open(STYLE_FULL).convert("RGB")

    target_h = content_pil.size[1]
    content_pil = _pil_resize_to_height(content_pil, target_h)
    out_pil     = _pil_resize_to_height(out_pil,     target_h)
    style_pil   = _pil_resize_to_height(style_pil,   target_h)

    panels = [
        ("Tartalomkép",                         content_pil),
        (f"Stíluskép ({STYLE})",                style_pil),
        (f"NST kimenet (style\\_weight = 1e6)", out_pil),
    ]

    total_w = sum(p.size[0] for _, p in panels) + PANEL_GAP * (len(panels) - 1)
    total_h = target_h + LABEL_H
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(22)

    x = 0
    for label, pil in panels:
        canvas.paste(pil, (x, LABEL_H))
        clean = label.replace("\\_", "_")
        bbox = draw.textbbox((0, 0), clean, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (pil.size[0] - tw) // 2, 10), clean, fill="black", font=font)
        x += pil.size[0] + PANEL_GAP

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH, "PNG", quality=95)
    print(f"[nst] Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
