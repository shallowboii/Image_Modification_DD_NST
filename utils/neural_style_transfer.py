# utils/neural_style_transfer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server use
import matplotlib.pyplot as plt

from utils.utilities import utilities
from utils.utilities.constants import DEVICE, STYLE_DIR, SupportedModels, SupportedPretrainedWeights


# ────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet(img_tensor):
    """Normalize a [0, 1] BCHW tensor to ImageNet stats in-place safe way."""
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std  = IMAGENET_STD.to(img_tensor.device)
    return (img_tensor - mean) / std


# ────────────────────────────────────────────────────────────────────────────
# Core loss components
# ────────────────────────────────────────────────────────────────────────────

def gram_matrix(tensor):
    """
    Compute the Gram matrix (style descriptor) for a feature map.

    Normalized by (C * H * W) so that style weights stay meaningful
    regardless of layer spatial size or number of channels.
    """
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)


def tv_loss(img):
    """Total variation loss — penalizes pixel-level noise for a smoother result."""
    diff_h = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    diff_v = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return diff_h.mean() + diff_v.mean()


# ────────────────────────────────────────────────────────────────────────────
# Feature extraction — uses ResNet50's own forward() pass
# ────────────────────────────────────────────────────────────────────────────

def get_features(image, model):
    """
    Extract multi-scale feature maps from the trained ResNet50.

    The model's forward() already returns a namedtuple with four outputs,
    one per ResNet block group:
        layer1 — 256 channels  (early: edges, colours, fine textures)
        layer2 — 512 channels  (mid:   patterns, small structures)
        layer3 — 1024 channels (deep:  semantic content, large structures)
        layer4 — 2048 channels (deepest: high-level semantics)

    Images are normalized with ImageNet statistics before being passed
    through the network, which is required for any pretrained ResNet.
    """
    image_norm = normalize_imagenet(image)
    outputs = model(image_norm)  # namedtuple: .layer1 .layer2 .layer3 .layer4
    return {
        "layer1": outputs.layer1,
        "layer2": outputs.layer2,
        "layer3": outputs.layer3,
        "layer4": outputs.layer4,
    }


# ────────────────────────────────────────────────────────────────────────────
# Progress helpers
# ────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor):
    """Convert a [0, 1] BCHW tensor to a PIL image. NaN-safe."""
    arr = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _save_progress(tensor, step, progress_dir):
    path = os.path.join(progress_dir, f"step_{step:04d}.jpg")
    _tensor_to_pil(tensor).save(path)
    return path


def _plot_loss_curves(history, save_dir):
    """Save a 3-panel loss curve plot to save_dir/loss_curves.png."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels  = ["content", "style", "tv"]
    colours = ["steelblue", "coral", "seagreen"]
    titles  = ["Content Loss", "Style Loss", "TV Loss"]
    for ax, key, col, title in zip(axes, labels, colours, titles):
        ax.plot(history[key], color=col, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=0.3)
    fig.suptitle("NST Loss Curves", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _progress_bar(step, total, width=30):
    filled = int(width * step / total)
    return "█" * filled + "░" * (width - filled)


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def run_style_transfer_from_numpy(
    content_np,
    style_path,
    num_steps=300,
    style_weight=1e6,
    content_weight=5.0,
    tv_weight=5e-4,
    imsize=1080,
    save_every=50,
    progress_dir=None,
    plot_losses=False,
):
    """
    Neural Style Transfer using the custom trained ResNet50 as feature extractor.

    Args:
        content_np:     numpy HxWx3 float32 in [0, 1]  — the content image
        style_path:     path to style image (absolute, or relative to STYLE_DIR)
        num_steps:      number of Adam optimizer steps
        style_weight:   weight for the style loss term
        content_weight: weight for the content loss term
        tv_weight:      weight for the total variation smoothness term
        imsize:         resize longest side to this before processing
        save_every:     save an intermediate image every N steps (0 = disabled)
        progress_dir:   folder for intermediate images (default: results/nst_progress)
        plot_losses:    if True, save a loss-curve plot at the end

    Returns:
        numpy HxWx3 float32 in [0, 1]
    """

    # ── Resize to imsize on the longest side ──────────────────────────────
    h, w, _ = content_np.shape
    if max(h, w) > imsize:
        scale = imsize / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    loader = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
    ])

    # ── Load images ───────────────────────────────────────────────────────
    content_pil = Image.fromarray((content_np * 255).astype(np.uint8))
    content_img = loader(content_pil).unsqueeze(0).to(DEVICE)

    if not os.path.isabs(style_path):
        style_path = os.path.join(STYLE_DIR, style_path)
    style_pil = Image.open(style_path).convert("RGB")
    style_img = loader(style_pil).unsqueeze(0).to(DEVICE)

    # ── Load ResNet50 ─────────────────────────────────────────────────────
    cnn = utilities.fetch_and_prepare_model(
        SupportedModels.RESNET50.name,
        SupportedPretrainedWeights.MYIMAGENET.name,
        DEVICE,
    )
    cnn.eval()
    # Disable in-place ReLU so autograd works on intermediate tensors
    for m in cnn.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    for p in cnn.parameters():
        p.requires_grad = False

    # ── Extract reference features ────────────────────────────────────────
    with torch.no_grad():
        content_features = get_features(content_img, cnn)
        style_features   = get_features(style_img, cnn)

    # Content: layer3 — preserves scene structure without being too pixel-exact
    CONTENT_LAYER = "layer3"

    # Style: layers 1–3 capture textures at multiple scales.
    # layer4 is too semantically abstract to be useful for texture transfer.
    STYLE_LAYERS = {
        "layer1": 1.0,   # fine textures, colour patches, brushstrokes
        "layer2": 0.8,   # mid-scale patterns
        "layer3": 0.4,   # large-scale stylistic structures
    }

    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in STYLE_LAYERS
    }

    # ── Progress setup ────────────────────────────────────────────────────
    if save_every > 0:
        progress_dir = progress_dir or "results/nst_progress"
        os.makedirs(progress_dir, exist_ok=True)

    loss_history = {"content": [], "style": [], "tv": []}

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Adam is used instead of LBFGS: it is stable on non-convex landscapes,
    # does not use a closure/line-search (which caused NaN with LBFGS), and
    # clamps cleanly after each step.
    target = content_img.clone().requires_grad_(True).to(DEVICE)
    optimizer = optim.Adam([target], lr=0.02)

    print(f"\n[NST] Starting style transfer")
    print(f"      Size    : {new_w}x{new_h}px  |  Device: {DEVICE}")
    print(f"      Steps   : {num_steps}  |  content={content_weight}  style={style_weight:.0e}  tv={tv_weight:.0e}")
    print(f"      Layers  : content={CONTENT_LAYER}  style={list(STYLE_LAYERS.keys())}")
    print("─" * 68)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        feats = get_features(target, cnn)

        # Content loss — MSE on deep feature maps
        c_loss = torch.mean(
            (feats[CONTENT_LAYER] - content_features[CONTENT_LAYER]) ** 2
        )

        # Style loss — gram matrix distance, summed across layers
        s_loss = torch.tensor(0.0, device=DEVICE)
        for layer, weight in STYLE_LAYERS.items():
            tgt_gram = gram_matrix(feats[layer])
            s_loss = s_loss + weight * torch.mean(
                (tgt_gram - style_grams[layer]) ** 2
            )

        # Total variation loss — spatial smoothness
        tv = tv_loss(target)

        total = content_weight * c_loss + style_weight * s_loss + tv_weight * tv
        total.backward()

        # Clip gradients to prevent any remaining instability
        torch.nn.utils.clip_grad_norm_([target], max_norm=1.0)

        optimizer.step()

        # Clamp after the update — safe here because we are outside autograd
        with torch.no_grad():
            target.clamp_(0, 1)

        c_l  = c_loss.item()
        s_l  = s_loss.item()
        tv_l = tv.item()

        loss_history["content"].append(c_l)
        loss_history["style"].append(s_l)
        loss_history["tv"].append(tv_l)

        # Console progress
        bar = _progress_bar(step, num_steps)
        print(f"[{bar}] {step:3d}/{num_steps} "
              f"| C: {c_l:.4f}  S: {s_l:.4f}  TV: {tv_l:.5f}")

        # Intermediate image save
        if save_every > 0 and step % save_every == 0:
            saved = _save_progress(target, step, progress_dir)
            print(f"  ↳ Progress image saved → {saved}")

    # ── Finalize ──────────────────────────────────────────────────────────
    with torch.no_grad():
        target.clamp_(0, 1)

    print("─" * 68)
    print(f"[NST] Done.  Final content loss: {loss_history['content'][-1]:.4f}"
          f"  style: {loss_history['style'][-1]:.4f}\n")

    if plot_losses and progress_dir:
        curve_path = _plot_loss_curves(loss_history, progress_dir)
        print(f"  ↳ Loss curves saved → {curve_path}")

    out_np = target.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(out_np, 0, 1)
