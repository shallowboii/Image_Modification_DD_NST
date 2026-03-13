import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, PROJECT_ROOT)

from utils import main_deepdream as md

def run_deepdream_once(np_img, config):
    """
    np_img: HxWx3 float32 in [0,1]
    config: dict using the same keys as main_deepdream expects (pyramid_size, pyramid_ratio, ...)
            Optional key: dream_blend (float 0-1, default 1.0)
              1.0 = full dream output, 0.5 = equal mix with original, 0.0 = original unchanged.
    Returns: numpy image HxWx3 float32 in [0,1]
    """
    original = np_img.copy()
    out_img = md.deep_dream_static_image(config, img=np_img)

    dream_blend = float(config.get("dream_blend", 1.0))
    if dream_blend < 1.0:
        out_img = dream_blend * out_img + (1.0 - dream_blend) * original
        out_img = np.clip(out_img, 0.0, 1.0).astype(np.float32)

    return out_img
