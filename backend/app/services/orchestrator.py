# backend/app/services/orchestrator.py
import json
from .deepdream_service import run_deepdream_once
from .nst_service import run_nst
from utils.utilities import utilities
import os
import numpy as np

PRESETS = {
    "LSD": {
        "dd": {"pyramid_size":5, "pyramid_ratio":1.8, "num_gradient_ascent_iterations":12, "lr":0.12, "layers_to_use":["layer3"]},
        "nst": {"style_path":"kaleidoscope.jpg", "num_steps":200, "style_weight":1e6}
    },
    "DMT": {
        "dd": {"pyramid_size":6, "pyramid_ratio":1.6, "num_gradient_ascent_iterations":8, "lr":0.2, "layers_to_use":["layer4","layer3"]},
        "nst": {"style_path":"mandala.jpg", "num_steps":300, "style_weight":2e6}
    }
}

def run_cyclic_pipeline(input_path, preset_name, overrides, dump_dir):
    """
    input_path: path to input image
    preset_name: LSD/DMT/...
    overrides: dict of params (edges: dd.num_gradient_ascent_iterations, lr etc)
    dump_dir: where to save final result
    returns final_output_path
    """
    print("Starting cyclic pipeline with input:", input_path, "preset:", preset_name, "overrides:", overrides, "dump_dir:", dump_dir)
    preset = PRESETS.get(preset_name)
    print("Using preset:", preset_name, preset)
    if preset is None:
        raise ValueError("Unknown preset")

    # load input image as numpy
    print("Loading input image:", input_path)
    img = utilities.load_image(input_path, target_shape=overrides.get("img_width", 600))
    # possibly start with noise
    print("Use noise is: "+str(overrides.get("use_noise", False)))
    if overrides.get("use_noise"):
        img = np.random.uniform(low=0.0, high=1.0, size=img.shape).astype(np.float32)
    print("Input image shape:", img.shape)
    num_cycles = overrides.get("cycles", 3)
    current = img
    for c in range(num_cycles):
        # prepare dd config
        dd_conf = {
            "model_name": overrides.get("model_name", "RESNET50"),
            "pretrained_weights": overrides.get("pretrained_weights", "MYIMAGENET"),
            "img_width": overrides.get("img_width", 600),
            "pyramid_size": overrides.get("pyramid_size", preset["dd"]["pyramid_size"]),
            "pyramid_ratio": overrides.get("pyramid_ratio", preset["dd"]["pyramid_ratio"]),
            "num_gradient_ascent_iterations": overrides.get("num_gradient_ascent_iterations", preset["dd"]["num_gradient_ascent_iterations"]),
            "lr": overrides.get("lr", preset["dd"]["lr"]),
            "layers_to_use": overrides.get("layers_to_use", preset["dd"].get("layers_to_use", ["layer3"])),
            "use_noise": overrides.get("use_noise", False),
            "dump_dir": dump_dir,
            "input_name": os.path.basename(input_path)
        }
        print(f"Cycle {c+1}/{num_cycles}: Running DeepDream with config:", dd_conf)
        # run dd
        current = run_deepdream_once(current, dd_conf)

        # run NST
        nst_conf = preset["nst"]
        print(f"Cycle {c+1}/{num_cycles}: Running NST with config:", nst_conf)
        current = run_nst(current, nst_conf["style_path"], num_steps=nst_conf.get("num_steps",200),
                          style_weight=nst_conf.get("style_weight",1e6))
    # save final
    out_name = os.path.join(dump_dir, "final_result.jpg")
    utilities.save_and_maybe_display_image({"dump_dir":dump_dir, "should_display":False, "input_name":os.path.basename(input_path)}, current)
    print("Final output saved to:", out_name)
    return out_name