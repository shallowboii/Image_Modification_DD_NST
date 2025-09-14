# backend/app/services/orchestrator.py
import json
from .deepdream_service import run_deepdream_once
from .nst_service import run_nst
from utils.utilities import utilities
from utils.utilities.constants import PRESET_FILE
import os
import numpy as np

def load_presets():
    print("Loading presets from:", PRESET_FILE)
    if os.path.exists(PRESET_FILE):
        with open(PRESET_FILE) as f:
            return json.load(f)
    return {}

def run_cyclic_pipeline(input_path, parameters, dump_dir):
    """
    input_path: path to input image
    preset_name: LSD/DMT/...
    overrides: dict of params (edges: dd.num_gradient_ascent_iterations, lr etc)
    dump_dir: where to save final result
    returns final_output_path
    """
    print("Starting cyclic pipeline with input:", input_path, "preset:", parameters.get("preset", ""), "overrides:", parameters, "dump_dir:", dump_dir)
    presets = load_presets()
    preset = presets.get(parameters.get("preset", ""), {})
    print("Using preset:", parameters.get("preset", ""))
    img = utilities.load_image(input_path, target_shape=parameters.get("img_width", 512))
    current = img
    for c in range(parameters.get("cycles", 3)):
            # prepare dd config
            dd_conf = {
                "model_name": parameters.get("model_name") or "RESNET50",
                "pretrained_weights": parameters.get("pretrained_weights") or "MYIMAGENET",
                "img_width": parameters.get("img_width") or 512,
                "pyramid_size": parameters.get("pyramid_size") or preset["dd"]["pyramid_size"],
                "pyramid_ratio": parameters.get("pyramid_ratio") or preset["dd"]["pyramid_ratio"],
                "num_gradient_ascent_iterations": parameters.get("num_gradient_ascent_iterations") or preset["dd"]["num_gradient_ascent_iterations"],
                "lr": parameters.get("lr") or preset["dd"]["lr"],
                "layers_to_use": [parameters.get("layer")] if parameters.get("layer") else preset["dd"]["layer"],
                "use_noise": parameters.get("use_noise") or False,
                "dump_dir": dump_dir,
                "input_name": os.path.basename(input_path)
            }
            print(f"Cycle {c+1}/{parameters.get("cycles",3)}: Running DeepDream with config:", dd_conf)
            current = run_deepdream_once(current, dd_conf)

            # run NST
            nst_conf = {
                "style_path": parameters.get("style") or preset["nst"]["style_path"],
                "num_steps": parameters.get("num_iterations_nst", preset["nst"]["num_steps"]),
                "style_weight": parameters.get("style_weight", preset["nst"]["style_weight"]),
            }
            print(f"Cycle {c+1}/{parameters.get("cycles",3)}: Running NST with config:", nst_conf)
            current = run_nst(current, nst_conf["style_path"], num_steps=nst_conf.get("num_steps",200),
                            style_weight=nst_conf.get("style_weight",1e4))
    # load input image as numpy
    print("Loading input image:", input_path)
    
    # save final
    out_name = os.path.join(dump_dir, "final_result.jpg")
    utilities.save_and_maybe_display_image({"dump_dir":dump_dir, "should_display":False, "input_name":os.path.basename(input_path)}, current)
    print("Final output saved to:", out_name)
    return out_name