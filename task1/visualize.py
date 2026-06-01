"""Task-1 insight visualizations.

Given the trained "best" model (reports/models/best_resnet18.pth) this script
produces the figures required by assignment item 5 (filters / loss landscape /
network interpretation):

  1. first-layer convolution filters
  2. feature maps for a sample image
  3. training & validation curves (from the saved JSON histories)
  4. confusion matrix on the test set
  5. Grad-CAM saliency overlays for a few test images
  6. ablation bar charts (filters / loss / activation) and optimizer curves

Run:
    python task1/visualize.py
"""
import os
import sys
import glob
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import get_cifar_loader, CIFAR_MEAN, CIFAR_STD, CLASSES
from common.utils import load_json
from task1.models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(ROOT, "reports", "figures")
LOG_DIR = os.path.join(ROOT, "reports", "logs")
MODEL_DIR = os.path.join(ROOT, "reports", "models")
os.makedirs(FIG_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
_STD = torch.tensor(CIFAR_STD).view(3, 1, 1)


def denorm(x):
    """Undo CIFAR normalization for display."""
    return (x.cpu() * _STD + _MEAN).clamp(0, 1)


def load_best_model(tag="best_resnet18"):
    path = os.path.join(MODEL_DIR, f"{tag}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing weights {path}; run train.py --exp best first")
    rec = load_json(os.path.join(LOG_DIR, f"{tag}.json"))
    model = build_model(rec["model"], **rec["model_kwargs"]).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model, rec


# ---------------------------------------------------------------------------
# 1. first-layer conv filters
# ---------------------------------------------------------------------------
def plot_first_layer_filters(model, save="task1_filters.png"):
    # first Conv2d in the model
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.detach().cpu()           # [out, in, kh, kw]
    w = w[:64]                                # first 64 filters
    n = w.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            f = w[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f.permute(1, 2, 0).numpy())
    fig.suptitle("First convolutional layer filters")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, save)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 2. feature maps
# ---------------------------------------------------------------------------
def plot_feature_maps(model, save="task1_feature_maps.png"):
    loader = get_cifar_loader(train=False, regime="plain", batch_size=8, shuffle=False)
    x, _ = next(iter(loader))
    img = x[0:1].to(DEVICE)

    acts = {}
    def hook(name):
        return lambda m, i, o: acts.__setitem__(name, o.detach().cpu())
    handles = []
    # hook the first conv and the first block of each ResNet stage if present
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    for idx in [0, len(convs) // 4, len(convs) // 2]:
        handles.append(convs[idx].register_forward_hook(hook(f"conv{idx}")))
    with torch.no_grad():
        model(img)
    for h in handles:
        h.remove()

    keys = list(acts.keys())
    fig, axes = plt.subplots(len(keys), 8, figsize=(10, 1.4 * len(keys)))
    if len(keys) == 1:
        axes = axes[None, :]
    for r, k in enumerate(keys):
        fmap = acts[k][0]
        for c in range(8):
            ax = axes[r, c]
            ax.axis("off")
            if c < fmap.shape[0]:
                ax.imshow(fmap[c].numpy(), cmap="viridis")
        axes[r, 0].set_ylabel(k, rotation=0, labelpad=25, fontsize=8)
    fig.suptitle("Feature maps at successive conv layers")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, save)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 3. training curves
# ---------------------------------------------------------------------------
def plot_training_curves(tag="best_resnet18", save="task1_training_curves.png"):
    rec = load_json(os.path.join(LOG_DIR, f"{tag}.json"))
    h = rec["history"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="val")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    axes[1].plot(epochs, h["train_acc"], label="train")
    axes[1].plot(epochs, h["val_acc"], label="val")
    axes[1].axhline(rec["final_val_acc"], color="gray", ls="--", lw=0.8,
                    label=f"final val {rec['final_val_acc']:.3f}")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{rec['model']} training ({rec['n_params']:,} params)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, save)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 4. confusion matrix
# ---------------------------------------------------------------------------
@torch.no_grad()
def plot_confusion_matrix(model, save="task1_confusion_matrix.png"):
    loader = get_cifar_loader(train=False, regime="plain", batch_size=256, shuffle=False)
    cm = np.zeros((10, 10), dtype=int)
    for x, y in loader:
        pred = model(x.to(DEVICE)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    ax.set(title="Confusion matrix (row-normalized)",
           xlabel="predicted", ylabel="true")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, save)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)
    return cm


# ---------------------------------------------------------------------------
# 5. Grad-CAM
# ---------------------------------------------------------------------------
def grad_cam(model, img, target_layer):
    """Return a Grad-CAM heatmap for `img` (1,3,32,32) at `target_layer`."""
    feats, grads = {}, {}
    h1 = target_layer.register_forward_hook(
        lambda m, i, o: feats.__setitem__("v", o))
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("v", go[0]))
    model.zero_grad()
    logits = model(img)
    cls = logits.argmax(1)
    logits[0, cls].backward()
    h1.remove(); h2.remove()

    a = feats["v"].detach()          # [1,C,h,w]
    g = grads["v"].detach()          # [1,C,h,w]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, int(cls.item())


def plot_grad_cam(model, n=6, save="task1_gradcam.png"):
    # last conv layer for ResNet (layer4) or generic last conv
    target = None
    if hasattr(model, "layer4"):
        target = model.layer4[-1].conv2
    else:
        target = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]

    loader = get_cifar_loader(train=False, regime="plain", batch_size=n, shuffle=True)
    x, y = next(iter(loader))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        img = x[i:i+1].to(DEVICE)
        cam, pred = grad_cam(model, img, target)
        disp = denorm(x[i]).permute(1, 2, 0).numpy()
        axes[0, i].imshow(disp); axes[0, i].axis("off")
        axes[0, i].set_title(f"{CLASSES[y[i]]}", fontsize=8)
        axes[1, i].imshow(disp)
        axes[1, i].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"pred {CLASSES[pred]}", fontsize=8)
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    fig.suptitle("Grad-CAM interpretation")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, save)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# 6. ablation summaries
# ---------------------------------------------------------------------------
def plot_ablations(save_prefix="task1_ablation"):
    def collect(prefix):
        out = {}
        for path in glob.glob(os.path.join(LOG_DIR, f"{prefix}*.json")):
            rec = load_json(path)
            key = rec["tag"].replace(prefix, "").lstrip("_")
            out[key] = rec
        return out

    groups = {
        "width": ("abl_width_", "number of filters (SimpleCNN width)"),
        "loss": ("abl_loss_", "loss function / regularization"),
        "act": ("abl_act_", "activation function"),
    }
    for gname, (prefix, title) in groups.items():
        recs = collect(prefix)
        if not recs:
            continue
        # sort widths numerically
        def sort_key(k):
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)
        keys = sorted(recs.keys(), key=sort_key)
        accs = [recs[k]["final_val_acc"] for k in keys]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(keys, accs, color="tab:blue")
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.3f}",
                    ha="center", fontsize=8)
        ax.set(title=f"Ablation: {title}", ylabel="final val accuracy")
        ax.set_ylim(0, max(accs) * 1.12)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"{save_prefix}_{gname}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out)


def plot_optimizer_comparison(save="task1_optimizer_comparison.png"):
    tags = ["opt_sgd", "opt_adam", "opt_custom_sgd", "opt_custom_adam"]
    colors = {"opt_sgd": "tab:blue", "opt_adam": "tab:orange",
              "opt_custom_sgd": "tab:green", "opt_custom_adam": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for tag in tags:
        p = os.path.join(LOG_DIR, f"{tag}.json")
        if not os.path.exists(p):
            continue
        h = load_json(p)["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], color=colors[tag], label=tag)
        axes[1].plot(ep, h["val_acc"], color=colors[tag], label=tag)
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Optimizer comparison (torch.optim vs. from-scratch)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, save)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def main():
    # ablation/optimizer figures depend only on JSON logs
    plot_ablations()
    plot_optimizer_comparison()
    try:
        model, rec = load_best_model()
    except FileNotFoundError as e:
        print(e)
        return
    plot_training_curves()
    plot_first_layer_filters(model)
    plot_feature_maps(model)
    plot_confusion_matrix(model)
    plot_grad_cam(model)


if __name__ == "__main__":
    main()
