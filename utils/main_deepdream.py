"""
    This file contains the implementation of the DeepDream algorithm.

    If you have problems understanding any parts of the code,
    go ahead and experiment with functions in the playground.py file.
"""

import os
import argparse
import shutil
import time


import numpy as np
import torch
import torchvision
import cv2 as cv


from .utilities import utilities
from .utilities.constants import *

def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    # Step 0: Feed forward pass
    out = model(input_tensor)

    # Step 1: Grab activations/feature maps of interest
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]

    # Step 2: Calculate loss over activations
    losses = []
    for layer_activation in activations:
        # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
        # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
        # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
        # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
        loss_component = torch.nn.MSELoss(reduction='mean')
        output = loss_component(layer_activation, torch.zeros_like(layer_activation))
        losses.append(output)

    loss = torch.mean(torch.stack(losses))
    loss.backward()

    # Step 3: Process image gradients (smoothing + normalization)
    grad = input_tensor.grad.data

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    #sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    #smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

    # Normalize the gradients (make them have mean = 0 and std = 1)
    g_std = torch.std(grad)
    g_mean = torch.mean(grad)
    grad = grad - g_mean
    grad = grad / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += config['lr'] * grad

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def deep_dream_static_image(config, img):
    #model loading
    model = utilities.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
    try:
        layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']] # amelyik layernek az aktivacioit akarjuk felerősíteni
    except Exception as e:  # making sure you set the correct layer name for this specific model
        raise ValueError(
            f'Invalid layer names {config["layers_to_use"]}. '
            f'Available layers for model {config["model_name"]}: {model.layer_names}'
        ) from e

    if img is None:  # betölteni a képet
        img_path = utilities.parse_input_file(config['input'])
        
        img = utilities.load_image(img_path, target_shape=config['img_width']) #beolvas méretre vágja utana mivel rgb értékek vannak ezert 0-255ig vannak az egyes pixel értékek
        # elosztja 255 hogy 0-1 közötti értékekkel dolgozzunk
        print(f"Use noise is: {config['use_noise']}")
        if config['use_noise']:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = utilities.pre_process_numpy_img(img) #normalize
    base_shape = img.shape[:-1]  # save initial height and width
    original_img = img.copy()  # full-resolution reference for octave detail-injection

    # Classic DeepDream octave scheme: preserve high-frequency detail across scales.
    # At each octave we run gradient ascent on the downscaled base, then extract the
    # "detail" (dreamed - downscaled_base) and add it back to the next octave's
    # downscaled base. This keeps the original full-res sharpness intact instead of
    # throwing it away on the first downscale.
    detail = np.zeros_like(cv.resize(
        original_img,
        (utilities.get_new_shape(config, base_shape, 0)[1],
         utilities.get_new_shape(config, base_shape, 0)[0])
    ))

    for pyramid_level in range(config['pyramid_size']):
        new_shape = utilities.get_new_shape(config, base_shape, pyramid_level)
        octave_base = cv.resize(original_img, (new_shape[1], new_shape[0]))

        if detail.shape[:2] != octave_base.shape[:2]:
            detail = cv.resize(detail, (new_shape[1], new_shape[0]))

        img = octave_base + detail
        input_tensor = utilities.pytorch_input_adapter(img, DEVICE)

        for iteration in range(config['num_gradient_ascent_iterations']):
            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)

        img = utilities.pytorch_output_adapter(input_tensor)
        detail = img - octave_base

    return utilities.post_process_numpy_img(img)



if __name__ == "__main__":

    # Only a small subset is exposed by design to avoid cluttering
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument("--input", type=str, help="Input IMAGE or VIDEO name that will be used for dreaming", default='golden_gate.jpg')
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=600)
    parser.add_argument("--layers_to_use", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=['layer3'])
    parser.add_argument("--model_name", choices=[m.name for m in SupportedModels],
                        help="Neural network (model) to use for dreaming", default=SupportedModels.RESNET50.name)
    parser.add_argument("--pretrained_weights", choices=[pw.name for pw in SupportedPretrainedWeights],
                        help="Pretrained weights to use for the above model", default=SupportedPretrainedWeights.MYIMAGENET.name)

    # Main params for experimentation (especially pyramid_size and pyramid_ratio)
    parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=5)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.8)
    parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.09)


    # You usually won't need to change these as often
    parser.add_argument("--should_display", action='store_true', help="Display intermediate dreaming results (default False)")
    parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)
    parser.add_argument("--smoothing_coefficient", type=float, help='Directly controls standard deviation for gradient smoothing', default=0.5)
    parser.add_argument("--use_noise", action='store_true', help="Use noise as a starting point instead of input image (default False)")
    args = parser.parse_args()

    # Wrapping configuration into a dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['dump_dir'] = OUT_IMAGES_PATH
    config['dump_dir'] = os.path.join(config['dump_dir'], f'{config["model_name"]}_{config["pretrained_weights"]}')
    config['input_name'] = os.path.basename(config['input'])

    # Create Ouroboros video (feeding neural network's output to it's input)
    print('Dreaming started!')
        
    img = deep_dream_static_image(config,img=None)  # img=None -> will be loaded inside of deep_dream_static_image
    dump_path = utilities.save_and_maybe_display_image(config, img)
    print(f'Saved DeepDream static image to: {os.path.relpath(dump_path)}\n')

