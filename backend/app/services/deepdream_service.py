import numpy as np
import sys
PROJECT_ROOT = r"C:\Users\marci\Desktop\Image_Modification_DD_NST"

# Put it at the front of sys.path so it has highest priority
sys.path.insert(0, PROJECT_ROOT)
from utils import main_deepdream as md
from utils.utilities.constants import DEVICE

def run_deepdream_once(np_img, config):
    """
    np_img: HxWx3 float32 in [0,1]
    config: dict using the same keys as main_deepdream expects (pyramid_size, pyramid_ratio, ...)
    Returns: numpy image HxWx3 float32 in [0,1]
    """
    # Reuse deep_dream_static_image by passing img directly
    # Note: deep_dream_static_image expects 'config' to contain required keys (model_name, pretrained_weights, img_width, ...)
    out_img = md.deep_dream_static_image(config, img=np_img)
    return out_img