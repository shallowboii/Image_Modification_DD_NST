"""
analysis/make_collage.py
========================
Creates a labeled grid collage from a list of images.

Usage:
    python analysis/make_collage.py \
        --config analysis/collage_church.json \
        --out Pictures/preset_comparison_church.png \
        --cols 4 \
        --title "Preset összehasonlítás – Katedrális"
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── Layout constants ──────────────────────────────────────────────────────────
THUMB_W      = 600          # width of each thumbnail (px)
LABEL_H      = 48           # height of the label bar below each image
PADDING      = 12           # gap between cells
TITLE_H      = 60           # height of the top title bar (0 = no title)
BG_COLOR     = (20, 20, 20) # dark background
LABEL_BG     = (40, 40, 40)
LABEL_COLOR  = (240, 240, 240)
TITLE_COLOR  = (255, 255, 255)
FONT_SIZE    = 22
TITLE_FONT_SIZE = 28


def _load_font(size: int):
    """Try to load a reasonable font; fall back to default."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_collage(entries: list[dict], out_path: str, cols: int = 4, title: str = ""):
    """
    entries : list of {"path": "...", "label": "..."}
    out_path: destination file (PNG/JPG)
    cols    : number of columns in the grid
    title   : optional title text at the top
    """
    rows = (len(entries) + cols - 1) // cols

    # ── Square thumbnails (center-cropped) regardless of source aspect ────────
    thumb_h = THUMB_W
    cell_w = THUMB_W
    cell_h = thumb_h + LABEL_H

    canvas_w = cols * cell_w + (cols + 1) * PADDING
    canvas_h = rows * cell_h + (rows + 1) * PADDING + (TITLE_H if title else 0)

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)
    font   = _load_font(FONT_SIZE)
    title_font = _load_font(TITLE_FONT_SIZE)

    # ── Title bar ─────────────────────────────────────────────────────────────
    y_offset = 0
    if title:
        draw.rectangle([(0, 0), (canvas_w, TITLE_H)], fill=(30, 30, 30))
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(((canvas_w - tw) // 2, (TITLE_H - (bbox[3] - bbox[1])) // 2),
                  title, fill=TITLE_COLOR, font=title_font)
        y_offset = TITLE_H

    # ── Cells ─────────────────────────────────────────────────────────────────
    for idx, entry in enumerate(entries):
        col = idx % cols
        row = idx // cols

        x = PADDING + col * (cell_w + PADDING)
        y = y_offset + PADDING + row * (cell_h + PADDING)

        # Thumbnail — center-crop to square, then resize
        img = Image.open(entry["path"]).convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        img  = img.crop((left, top, left + side, top + side))
        img  = img.resize((cell_w, thumb_h), Image.LANCZOS)
        canvas.paste(img, (x, y))

        # Label bar
        label_rect = [(x, y + thumb_h), (x + cell_w, y + thumb_h + LABEL_H)]
        draw.rectangle(label_rect, fill=LABEL_BG)

        label = entry.get("label", "")
        bbox  = draw.textbbox((0, 0), label, font=font)
        lw    = bbox[2] - bbox[0]
        lh    = bbox[3] - bbox[1]
        draw.text(
            (x + (cell_w - lw) // 2, y + thumb_h + (LABEL_H - lh) // 2),
            label, fill=LABEL_COLOR, font=font,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"Saved collage -> {out_path}  ({canvas_w}x{canvas_h})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="JSON file with list of {path, label} entries")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        entries = json.load(f)

    make_collage(entries, args.out, cols=args.cols, title=args.title)


if __name__ == "__main__":
    main()
