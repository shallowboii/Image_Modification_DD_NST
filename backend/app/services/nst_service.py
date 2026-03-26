from PIL import Image
import torchvision.transforms as transforms
from utils import neural_style_transfer as nst_module
from utils.utilities.constants import DEVICE

def pil_from_numpy(np_img):
    arr = (np_img * 255).astype('uint8')
    return Image.fromarray(arr)

def image_tensor_from_numpy(np_img, imsize=None):
    # convert numpy [0,1] HxWx3 -> torch tensor shape (1,3,H,W)
    pil = pil_from_numpy(np_img)
    loader = transforms.Compose([transforms.Resize(imsize) if imsize else transforms.Lambda(lambda x: x), transforms.ToTensor()])
    tensor = loader(pil).unsqueeze(0).to(DEVICE)
    return tensor

def run_nst(content_np, style_path, num_steps=200, style_weight=1e6, content_weight=1, tv_weight=5e-4):
    return nst_module.run_style_transfer_from_numpy(
        content_np,
        style_path,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight,
        tv_weight=tv_weight,
    )
