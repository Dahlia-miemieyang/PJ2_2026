"""VGG Loss Landscape — Task-2 reference script.

This is the completed version of the skeleton shipped with the assignment.
It trains VGG-A and VGG-A-with-BatchNorm across several learning rates,
records the per-step training loss, and draws the loss-landscape band
(max_curve / min_curve filled with ``plt.fill_between``) that visualises how
Batch Normalization smooths the optimization landscape.

The heavier, fully-featured experiment driver (loss landscape + gradient
predictiveness + beta-smoothness + BN/no-BN accuracy comparison) lives in
``vgg_bn_experiments.py``; this file keeps the structure of the original
template for clarity and grading.

Run:
    python VGG_Loss_Landscape.py
"""
import os
import random

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
num_workers = 4
batch_size = 128

module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, os.pardir, "reports", "figures")
figures_path = os.path.normpath(figures_path)
os.makedirs(figures_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Data loaders.
train_loader = get_cifar_loader(train=True, batch_size=batch_size)
val_loader = get_cifar_loader(train=False, batch_size=256, shuffle=False)
for X, y in train_loader:
    print("sample batch:", X.shape, y.shape)
    break


@torch.no_grad()
def get_accuracy(model, loader=val_loader):
    """Accuracy of `model` over `loader`."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def set_random_seeds(seed_value=0, device="cpu"):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, epochs_n=100, max_steps=None):
    """Train and record the per-step training loss (for the loss landscape).

    Returns
    -------
    losses_step : list[float]   training loss at every optimization step
    grads       : list[float]   l2 norm of the classifier-layer gradient/step
    """
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    losses_step = []
    grads = []
    step = 0

    for epoch in tqdm(range(epochs_n), unit="epoch"):
        if scheduler is not None:
            scheduler.step()
        model.train()
        learning_curve[epoch] = 0

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            # Record per-step loss and the gradient of the last classifier
            # weight (model.classifier[-1] is the final Linear layer).
            losses_step.append(loss.item())
            grad = model.classifier[-1].weight.grad.detach().clone()
            grads.append(float(torch.norm(grad)))
            learning_curve[epoch] += loss.item()

            optimizer.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                return losses_step, grads
    return losses_step, grads


def train_multi_lr(model_ctor, learning_rates, epochs_n=2, max_steps=None):
    """Train `model_ctor` once per learning rate; collect per-step losses."""
    all_losses = []
    for lr in learning_rates:
        set_random_seeds(2026, device)
        model = model_ctor()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses, _ = train(model, optimizer, criterion, train_loader,
                          val_loader, epochs_n=epochs_n, max_steps=max_steps)
        all_losses.append(losses)
        print(f"  {model_ctor.__name__} lr={lr}: {len(losses)} steps, "
              f"final loss {losses[-1]:.3f}")
    return all_losses


def min_max_curves(all_losses):
    """Maintain max_curve / min_curve: the per-step max and min loss across
    all learning-rate runs."""
    n = min(len(l) for l in all_losses)
    mat = np.array([l[:n] for l in all_losses])
    return mat.max(axis=0), mat.min(axis=0)


def plot_loss_landscape(std_losses, bn_losses, save_name="loss_landscape.png"):
    """Plot the loss landscape band for VGG-A with and without BN."""
    std_max, std_min = min_max_curves(std_losses)
    bn_max, bn_min = min_max_curves(bn_losses)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs_std = np.arange(len(std_max))
    xs_bn = np.arange(len(bn_max))
    ax.fill_between(xs_std, std_min, std_max, color="tab:red", alpha=0.3,
                    label="Standard VGG")
    ax.fill_between(xs_bn, bn_min, bn_max, color="tab:blue", alpha=0.3,
                    label="Standard VGG + BatchNorm")
    ax.plot(xs_std, std_max, color="tab:red", lw=0.7)
    ax.plot(xs_std, std_min, color="tab:red", lw=0.7)
    ax.plot(xs_bn, bn_max, color="tab:blue", lw=0.7)
    ax.plot(xs_bn, bn_min, color="tab:blue", lw=0.7)
    ax.set(title="Loss Landscape", xlabel="Steps", ylabel="Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figures_path, save_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    # Learning rates as suggested by the assignment.
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    epo = 2  # epochs per learning rate (increase for smoother curves)

    print("Training standard VGG-A across learning rates ...")
    std_losses = train_multi_lr(VGG_A, learning_rates, epochs_n=epo)
    print("Training VGG-A + BatchNorm across learning rates ...")
    bn_losses = train_multi_lr(VGG_A_BatchNorm, learning_rates, epochs_n=epo)

    plot_loss_landscape(std_losses, bn_losses)
