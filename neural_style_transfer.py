# utils/neural_style_transfer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os

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


# ---- Feature extraction using your ResNet50 ----
def get_features(image, model):
    x = image
    features = {}
    # -_-
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    i = 0
    layer1=model.layerDict["layer1"]
    layer2=model.layerDict["layer2"]
    layer3=model.layerDict["layer3"]
    layer4=model.layerDict["layer4"]
    #layer1, layer2, layer3, layer4 : 
    for key,value in layer1.items():
    #    print(key)
    #    print(value)
    #    print("-----------------------------------------")
       for name, layer in value._modules.items():
            # print(name)
            # print(layer)
            # print("-----------------------------------------")
            if not isinstance(layer,nn.Sequential):
                x = layer(x)
                i+=1
                features[f"bneck_{name}_{i}"] = x
    for key,value in layer2.items():
    #    print(key)
    #    print(value)
    #    print("-----------------------------------------")
       for name, layer in value._modules.items():
            # print(name)
            # print(layer)
            # print("-----------------------------------------")
            if not isinstance(layer,nn.Sequential):
                x = layer(x)
                i+=1
                features[f"bneck_{name}_{i}"] = x

    for key,value in layer3.items():
    #    print(key)
    #    print(value)
    #    print("-----------------------------------------")
       for name, layer in value._modules.items():
        #   print(name)
        #   print(layer)
        #   print("-----------------------------------------")
          if not isinstance(layer,nn.Sequential):
                x = layer(x)
                i+=1
                features[f"bneck_{name}_{i}"] = x

    for key,value in layer4.items():
        # print(key)
        # print(value)
        # print("-----------------------------------------")
        if isinstance(value,Bottleneck):
            for name, layer in value._modules.items():
                if not isinstance(layer,nn.Sequential):
                    x = layer(x)
                    i+=1
                    features[f"bneck_{name}_{i}"] = x
        else:
           if not isinstance(layer,nn.Sequential):
                    x = layer(x)
                    i+=1
                    features[f"bneck_{name}_{i}"] = x
          
       
    

    return features 

# ---- Style transfer run ----
def run_style_transfer_from_numpy(
    content_np,
    style_path,
    num_steps=500,
    style_weight=1e6,
    content_weight=1,
    imsize=512,
):
    """
    Neural style transfer using your custom ResNet50.
    """
    # loader
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

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

    # disable grads on model
    for p in cnn.parameters():
        p.requires_grad = False

    # extract features
    content_features = get_features(content_img, cnn)
    style_features = get_features(style_img, cnn)
    style_grams = {k: gram_matrix(v) for k, v in style_features.items()}

    # choose which layers to use
    content_layer = "bneck_conv1_36"  # example: pick one deep content layer
    style_layers = [
        "bneck_conv2_3",
        "bneck_conv1_8",
        "bneck_conv2_10",
        "bneck_conv3_19",
        "bneck_conv1_22",
        "bneck_conv3_26",
        "bneck_conv3_33",
        "bneck_conv1_43",
    ]
    style_weights = {layer: 1.0 for layer in style_layers}  # uniform weights for now

    # optimize target image
    target = content_img.clone().requires_grad_(True).to(DEVICE)
    optimizer = optim.Adam([target], lr=0.003)

    for step in range(1, num_steps + 1):
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

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # convert back to numpy
    out_np = target.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    out_np = np.clip(out_np, 0, 1)
    return out_np
