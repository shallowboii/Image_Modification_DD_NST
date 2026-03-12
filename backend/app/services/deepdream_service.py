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
    Returns: numpy image HxWx3 float32 in [0,1]
    """
    out_img = md.deep_dream_static_image(config, img=np_img)
    return out_img
