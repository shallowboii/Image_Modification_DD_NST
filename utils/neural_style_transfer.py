# utils/neural_style_transfer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2

from utils.utilities import utilities
from utils.utilities.constants import DEVICE, SupportedModels, SupportedPretrainedWeights
from torchvision.models.resnet import Bottleneck


# ---- Normalization (for input images) ----
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)


# ---- Gram matrix ----
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


# ---- Total Variation Loss (smoothness) ----
def tv_loss(img):
    return (
        torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
        torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    )


# ---- Feature extraction using your ResNet50 ----
def get_features(image, model):
    x = image
    features = {}

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    i = 0

    layer1 = model.layerDict["layer1"]
    layer2 = model.layerDict["layer2"]
    layer3 = model.layerDict["layer3"]
    layer4 = model.layerDict["layer4"]

    for key, value in layer1.items():
        for name, layer in value._modules.items():
            if not isinstance(layer, nn.Sequential):
                x = layer(x)
                i += 1
                features[f"bneck_{name}_{i}"] = x

    for key, value in layer2.items():
        for name, layer in value._modules.items():
            if not isinstance(layer, nn.Sequential):
                x = layer(x)
                i += 1
                features[f"bneck_{name}_{i}"] = x

    for key, value in layer3.items():
        for name, layer in value._modules.items():
            if not isinstance(layer, nn.Sequential):
                x = layer(x)
                i += 1
                features[f"bneck_{name}_{i}"] = x

    for key, value in layer4.items():
        if isinstance(value, Bottleneck):
            for name, layer in value._modules.items():
                if not isinstance(layer, nn.Sequential):
                    x = layer(x)
                    i += 1
                    features[f"bneck_{name}_{i}"] = x
        else:
            if not isinstance(layer, nn.Sequential):
                x = layer(x)
                i += 1
                features[f"bneck_{name}_{i}"] = x

    return features


# ---- Style transfer run ----
def run_style_transfer_from_numpy(
    content_np,
    style_path,
    num_steps=300,
    style_weight=1e6,
    content_weight=5,
    imsize=1080,
):
    """
    Faster + better-looking Neural Style Transfer using custom ResNet50.
    """
    # loader
    h, w, _ = content_np.shape

    # Decide on max side
    if max(h, w) > imsize:
        scale = imsize / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    loader = transforms.Compose([
        transforms.Resize((new_h, new_w)),  # force both height and width
        transforms.ToTensor()
    ])

    # prepare images
    content_pil = Image.fromarray((content_np * 255).astype(np.uint8))
    content_img = loader(content_pil).unsqueeze(0).to(DEVICE)

    style_path = os.path.join(utilities.STYLE_DIR, style_path)
    style_pil = Image.open(style_path).convert("RGB")
    style_img = loader(style_pil).unsqueeze(0).to(DEVICE)

    # get model
    cnn = utilities.fetch_and_prepare_model(
        SupportedModels.RESNET50.name,
        SupportedPretrainedWeights.MYIMAGENET.name,
        DEVICE,
    )
    for m in cnn.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    for p in cnn.parameters():
        p.requires_grad = False

    # extract features
    content_features = get_features(content_img, cnn)
    style_features = get_features(style_img, cnn)
    style_grams = {k: gram_matrix(v) for k, v in style_features.items()}

    # choose layers
    content_layer = "bneck_conv2_10"  # better for structure
    style_layers = ["bneck_conv2_3", "bneck_conv3_26", "bneck_conv1_71"]
    style_weights = {
        "bneck_conv2_3": 5.0,
        "bneck_conv3_26": 3.0,
        "bneck_conv1_71": 1.0,
    }

    # optimize target
    target = content_img.clone().requires_grad_(True).to(DEVICE)
    optimizer = optim.LBFGS([target])

    run = [0]
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()

            target_features = get_features(target, cnn)

            # content loss
            content_loss = torch.mean(
                (target_features[content_layer] - content_features[content_layer]) ** 2
            )

            # style loss
            style_loss = 0
            for layer in style_layers:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                style_loss += style_weights[layer] * torch.mean(
                    (target_gram - style_gram) ** 2
                ) / (d * h * w)

            # tv loss
            tv = tv_loss(target)

            total_loss = content_weight * content_loss + style_weight * style_loss + 1e-5 * tv
            total_loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]} / {num_steps} | Content: {content_loss.item():.4f}, Style: {style_loss.item():.4f}, TV: {tv.item():.4f}")

            return total_loss.detach()

        optimizer.step(closure)

    with torch.no_grad():
        target.clamp_(0, 1)

    # convert back to numpy (denormalize to [0,1])
    
    out_np = target.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    out_np = np.clip(out_np, 0, 1)
    return out_np
