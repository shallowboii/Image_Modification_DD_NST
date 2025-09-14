import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils import neural_style_transfer as nst_module
from utils import utilities
from utils.utilities.constants import DEVICE

def pil_from_numpy(np_img):
    arr = (np_img * 255).astype('uint8')
    return Image.fromarray(arr)

def image_tensor_from_numpy(np_img, imsize=None):
    # convert numpy [0,1] HxWx3 -> torch tensor shape (1,3,H,W)
    pil = pil_from_numpy(np_img)
    loader = transforms.Compose([transforms.Resize(imsize) if imsize else transforms.Lambda(lambda x:x), transforms.ToTensor()])
    tensor = loader(pil).unsqueeze(0).to(DEVICE)
    return tensor

def run_nst(content_np, style_path, num_steps=200, style_weight=1e6, content_weight=1):
    return nst_module.run_style_transfer_from_numpy(
        content_np,
        style_path,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight
    )


def run_nst_old(content_np, style_path, num_steps=200, style_weight=1e6, content_weight=1):
    """
    content_np: numpy (H,W,3) in [0,1]
    style_path: filesystem path to style image
    returns: numpy HxWx3 [0,1] (post-processed)
    """
    # Convert to tensors
    # choose imsize according to device
    imsize = 512 if torch.cuda.is_available() else 128
    content = image_tensor_from_numpy(content_np, imsize=imsize)
    style = nst_module.image_loader(style_path)  # reuse loader from your file (it expects a path)
    # Ensure sizes match
    if style.size() != content.size():
        # resize style to content size:
        style = transforms.Resize([content.shape[2], content.shape[3]])(style.squeeze(0)).unsqueeze(0)

    input_img = content.clone()
    output = nst_module.run_style_transfer(nst_module.cnn, nst_module.cnn_normalization_mean,
                                           nst_module.cnn_normalization_std,
                                           content, style, input_img,
                                           num_steps=num_steps, style_weight=style_weight,
                                           content_weight=content_weight)
    # output is a torch tensor in [0,1], convert to numpy
    out_np = output.cpu().detach().squeeze(0).permute(1,2,0).numpy()
    return out_np