from collections import namedtuple
import torch
import os
import sys
from torchvision import models
from torch.hub import download_url_to_file
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ..utilities.constants import *

# Get the directory of the current file
BASE_DIR = os.path.dirname(__file__)




class ResNet50(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        cudaAvailable = torch.cuda.is_available()
        print(f"CUDA available: {cudaAvailable}")
        if pretrained_weights == SupportedPretrainedWeights.MYIMAGENET.name:
            resnet50 = models.resnet50(pretrained=False,progress=show_progress)
            # Build the path to the weights file
            modelpath = os.path.join(BASE_DIR, "weights", "CE_imagenet1k_resnet50_lr_0.1_acc48_bs128.pth")
            checkpoint = torch.load(modelpath,weights_only=False)
            # Access the state dictionaries
            model_state_dict = checkpoint['model']
            # Now, you can use the state_dicts to load into your model and optimizer
            resnet50.load_state_dict(model_state_dict)
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        # 3
        self.layer10 = resnet50.layer1[0]
        self.layer11 = resnet50.layer1[1]
        self.layer12 = resnet50.layer1[2]

        # 4
        self.layer20 = resnet50.layer2[0]
        self.layer21 = resnet50.layer2[1]
        self.layer22 = resnet50.layer2[2]
        self.layer23 = resnet50.layer2[3]

        # 6
        self.layer30 = resnet50.layer3[0]
        self.layer31 = resnet50.layer3[1]
        self.layer32 = resnet50.layer3[2]
        self.layer33 = resnet50.layer3[3]
        self.layer34 = resnet50.layer3[4]
        self.layer35 = resnet50.layer3[5]

        # 3
        self.layer40 = resnet50.layer4[0]
        self.layer41 = resnet50.layer4[1]
        # self.layer42 = resnet50.layer4[2]

        # Go even deeper into ResNet's BottleNeck module for layer 42
        self.layer42_conv1 = resnet50.layer4[2].conv1
        self.layer42_bn1 = resnet50.layer4[2].bn1
        self.layer42_conv2 = resnet50.layer4[2].conv2
        self.layer42_bn2 = resnet50.layer4[2].bn2
        self.layer42_conv3 = resnet50.layer4[2].conv3
        self.layer42_bn3 = resnet50.layer4[2].bn3
        self.layer42_relu = resnet50.layer4[2].relu

        self.layerDict =  {"layer1": {
                "layer10": self.layer10,
                "layer11": self.layer11,
                "layer12": self.layer12,
            },
            "layer2": {
                "layer20": self.layer20,
                "layer21": self.layer21,
                "layer22": self.layer22,
                "layer23": self.layer23,
            },
            "layer3": {
                "layer30": self.layer30,
                "layer31": self.layer31,
                "layer32": self.layer32,
                "layer33": self.layer33,
                "layer34": self.layer34,
                "layer35": self.layer35,
            },
            "layer4": {
                "layer40": self.layer40,
                "layer41": self.layer41,
                "layer42": {
                    "conv1": self.layer42_conv1,
                    "bn1": self.layer42_bn1,
                    "conv2": self.layer42_conv2,
                    "bn2": self.layer42_bn2,
                    "conv3": self.layer42_conv3,
                    "bn3": self.layer42_bn3,
                    "relu": self.layer42_relu,
                },
            },
        }

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        layer10 = x
        x = self.layer11(x)
        layer11 = x
        x = self.layer12(x)
        layer12 = x
        x = self.layer20(x)
        layer20 = x
        x = self.layer21(x)
        layer21 = x
        x = self.layer22(x)
        layer22 = x
        x = self.layer23(x)
        layer23 = x
        x = self.layer30(x)
        layer30 = x
        x = self.layer31(x)
        layer31 = x
        x = self.layer32(x)
        layer32 = x
        x = self.layer33(x)
        layer33 = x
        x = self.layer34(x)
        layer34 = x
        x = self.layer35(x)
        layer35 = x
        x = self.layer40(x)
        layer40 = x
        x = self.layer41(x)
        layer41 = x

        layer42_identity = layer41
        x = self.layer42_conv1(x)
        layer420 = x
        x = self.layer42_bn1(x)
        layer421 = x
        x = self.layer42_relu(x)
        layer422 = x
        x = self.layer42_conv2(x)
        layer423 = x
        x = self.layer42_bn2(x)
        layer424 = x
        x = self.layer42_relu(x)
        layer425 = x
        x = self.layer42_conv3(x)
        layer426 = x
        x = self.layer42_bn3(x)
        layer427 = x
        x += layer42_identity
        layer428 = x
        x = self.relu(x)
        layer429 = x

        # Feel free to experiment with different layers.
        net_outputs = namedtuple("ResNet50Outputs", self.layer_names)
        out = net_outputs(layer10, layer23, layer35, layer40)
        # layer35 is my favourite
        return out