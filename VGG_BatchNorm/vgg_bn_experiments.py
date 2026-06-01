"""Task-2 Batch-Normalization experiments.

Implements both required sub-parts of the assignment:

  Part A (VGG-A with / without BN, 15%):
      train VGG_A and VGG_A_BatchNorm under identical settings and compare
      their training-loss / validation-accuracy curves.

  Part B (How does BN help optimization?, 15%):
      for several learning rates, record the *per-step* training loss and the
      gradient of a chosen layer, then measure
        1. Loss landscape       -> max/min loss band over steps (fill_between)
        2. Gradient predictiveness -> L2 distance between consecutive gradients
        3. "effective beta-smoothness" -> max gradient-difference / step length

This reproduces the analysis of Santurkar et al., "How Does Batch
Normalization Help Optimization?" (NeurIPS 2018).

Run:
    python vgg_bn_experiments.py --part compare   # Part A
    python vgg_bn_experiments.py --part landscape  # Part B
    python vgg_bn_experiments.py --part all
"""
import argparse
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Make package-relative imports work regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from data.loaders import get_cifar_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))
FIG_DIR = os.path.join(REPO_ROOT, "reports", "figures")
LOG_DIR = os.path.join(REPO_ROOT, "reports", "logs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def set_random_seeds(seed_value=2026, device="cpu"):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    import random
    random.seed(seed_value)
    if device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def get_accuracy(model, loader, device):
    """Classification accuracy over a data loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def _grad_of_last_conv(model):
    """Flattened gradient of the last conv layer's weight (proxy for the
    optimization direction, following the BN paper's per-layer analysis)."""
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None or last_conv.weight.grad is None:
        return None
    return last_conv.weight.grad.detach().flatten().cpu()


def train_record(model, optimizer, criterion, train_loader, device,
                 epochs=1, max_steps=None, val_loader=None):
    """Train while recording per-step loss and per-step gradient.

    Returns dict with:
        loss_steps : list[float]      training loss at every optimization step
        grads      : list[Tensor]     last-conv gradient at every step
        val_acc    : list[float]      validation accuracy per epoch (optional)
    """
    model.to(device)
    loss_steps, grads, val_acc = [], [], []
    step = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            loss_steps.append(loss.item())
            g = _grad_of_last_conv(model)
            if g is not None:
                grads.append(g)
            optimizer.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if val_loader is not None:
            val_acc.append(get_accuracy(model, val_loader, device))
        if max_steps is not None and step >= max_steps:
            break
    return {"loss_steps": loss_steps, "grads": grads, "val_acc": val_acc}


# ===========================================================================
# Part A: VGG-A with and without BN
# ===========================================================================
def part_compare(epochs=20, lr=1e-3, batch_size=128):
    train_loader = get_cifar_loader(train=True, batch_size=batch_size)
    val_loader = get_cifar_loader(train=False, batch_size=256, shuffle=False)

    results = {}
    for name, ctor in [("VGG_A", VGG_A), ("VGG_A_BatchNorm", VGG_A_BatchNorm)]:
        set_random_seeds(2026, DEVICE.type)
        model = ctor().to(DEVICE)
        n_params = get_number_of_parameters(model)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        print(f"\n=== Part A: {name} ({n_params:,} params), lr={lr}, epochs={epochs} ===")

        train_loss_epoch, val_acc_epoch, train_acc_epoch = [], [], []
        for epoch in range(epochs):
            model.train()
            running, correct, total = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                running += loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
            tr_loss = running / total
            tr_acc = correct / total
            va = get_accuracy(model, val_loader, DEVICE)
            train_loss_epoch.append(tr_loss)
            train_acc_epoch.append(tr_acc)
            val_acc_epoch.append(va)
            print(f"[{name}] epoch {epoch+1:2d}/{epochs} "
                  f"train_loss {tr_loss:.4f} train_acc {tr_acc:.4f} val_acc {va:.4f}")

        results[name] = {
            "n_params": n_params,
            "train_loss": train_loss_epoch,
            "train_acc": train_acc_epoch,
            "val_acc": val_acc_epoch,
            "best_val_acc": max(val_acc_epoch),
        }
        torch.save(model.state_dict(),
                   os.path.join(REPO_ROOT, "reports", "models", f"{name}.pth"))

    _plot_compare(results, epochs)
    import json
    with open(os.path.join(LOG_DIR, "task2_compare.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def _plot_compare(results, epochs):
    xs = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, c in [("VGG_A", "tab:red"), ("VGG_A_BatchNorm", "tab:blue")]:
        axes[0].plot(xs, results[name]["train_loss"], color=c, marker="o",
                     ms=3, label=name)
        axes[1].plot(xs, results[name]["val_acc"], color=c, marker="o",
                     ms=3, label=name)
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("VGG-A with vs. without Batch Normalization")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "task2_bn_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ===========================================================================
# Part B: loss landscape, gradient predictiveness, beta-smoothness
# ===========================================================================
def _train_multi_lr(ctor, lrs, epochs, max_steps, batch_size):
    """Train one architecture at several LRs; return per-LR step records.

    num_workers=0: Part B trains 8 models back-to-back. On Windows, persistent
    DataLoader worker processes hold shared-memory mappings that accumulate
    across runs and eventually exhaust the system commit limit (error 1455).
    A single-process loader avoids this and is fast enough for the short runs.
    """
    train_loader = get_cifar_loader(train=True, batch_size=batch_size,
                                    num_workers=0)
    records = []
    for lr in lrs:
        set_random_seeds(2026, DEVICE.type)
        model = ctor().to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        print(f"  training {ctor.__name__} lr={lr} ...")
        rec = train_record(model, opt, criterion, train_loader, DEVICE,
                            epochs=epochs, max_steps=max_steps)
        records.append(rec)
    return records


def _curves_from_records(records):
    """Given per-LR step records, return aligned max/min loss curves and
    gradient-based diagnostics."""
    n_steps = min(len(r["loss_steps"]) for r in records)
    loss_mat = np.array([r["loss_steps"][:n_steps] for r in records])  # [n_lr, steps]
    max_curve = loss_mat.max(axis=0)
    min_curve = loss_mat.min(axis=0)

    # Gradient predictiveness: L2 distance between consecutive-step gradients,
    # aggregated (max/min) across LRs at each step.
    grad_diffs = []  # one list per LR
    beta_per_lr = []  # effective beta-smoothness per LR
    for r in records:
        g = r["grads"][:n_steps]
        d = [float(torch.norm(g[i + 1] - g[i])) for i in range(len(g) - 1)]
        grad_diffs.append(d)
        beta_per_lr.append(d)  # same quantity; "max diff in grad over distance"
    m = min(len(d) for d in grad_diffs) if grad_diffs else 0
    gd_mat = np.array([d[:m] for d in grad_diffs]) if m > 0 else np.zeros((len(records), 0))
    grad_max = gd_mat.max(axis=0) if m > 0 else np.array([])
    grad_min = gd_mat.min(axis=0) if m > 0 else np.array([])
    return {
        "max_curve": max_curve, "min_curve": min_curve,
        "loss_mat": loss_mat,
        "grad_max": grad_max, "grad_min": grad_min,
        "grad_mat": gd_mat,
    }


def part_landscape(lrs=(1e-3, 2e-3, 1e-4, 5e-4), epochs=2, max_steps=None,
                   batch_size=128):
    print(f"\n=== Part B: loss landscape, LRs={lrs}, epochs={epochs} ===")
    std = _train_multi_lr(VGG_A, lrs, epochs, max_steps, batch_size)
    bn = _train_multi_lr(VGG_A_BatchNorm, lrs, epochs, max_steps, batch_size)
    std_c = _curves_from_records(std)
    bn_c = _curves_from_records(bn)

    _plot_loss_landscape(std_c, bn_c)
    _plot_gradient_predictiveness(std_c, bn_c)
    _plot_beta_smoothness(std_c, bn_c)

    # Persist the raw curves (downsampled) for the report.
    import json
    def _ds(a, k=500):
        a = np.asarray(a)
        if a.size <= k:
            return a.tolist()
        idx = np.linspace(0, a.size - 1, k).astype(int)
        return a[idx].tolist()
    summary = {
        "lrs": list(lrs), "epochs": epochs,
        "standard": {"max_curve": _ds(std_c["max_curve"]),
                     "min_curve": _ds(std_c["min_curve"]),
                     "grad_max": _ds(std_c["grad_max"]),
                     "grad_min": _ds(std_c["grad_min"])},
        "batchnorm": {"max_curve": _ds(bn_c["max_curve"]),
                      "min_curve": _ds(bn_c["min_curve"]),
                      "grad_max": _ds(bn_c["grad_max"]),
                      "grad_min": _ds(bn_c["grad_min"])},
    }
    with open(os.path.join(LOG_DIR, "task2_landscape.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return std_c, bn_c


def _plot_loss_landscape(std_c, bn_c):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs_std = np.arange(len(std_c["max_curve"]))
    xs_bn = np.arange(len(bn_c["max_curve"]))
    ax.fill_between(xs_std, std_c["min_curve"], std_c["max_curve"],
                    color="tab:red", alpha=0.3, label="Standard VGG")
    ax.fill_between(xs_bn, bn_c["min_curve"], bn_c["max_curve"],
                    color="tab:blue", alpha=0.3, label="Standard VGG + BatchNorm")
    ax.plot(xs_std, std_c["max_curve"], color="tab:red", lw=0.8)
    ax.plot(xs_std, std_c["min_curve"], color="tab:red", lw=0.8)
    ax.plot(xs_bn, bn_c["max_curve"], color="tab:blue", lw=0.8)
    ax.plot(xs_bn, bn_c["min_curve"], color="tab:blue", lw=0.8)
    ax.set(title="Loss Landscape", xlabel="Steps", ylabel="Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "task2_loss_landscape.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_gradient_predictiveness(std_c, bn_c):
    if std_c["grad_max"].size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    xs_std = np.arange(len(std_c["grad_max"]))
    xs_bn = np.arange(len(bn_c["grad_max"]))
    ax.fill_between(xs_std, std_c["grad_min"], std_c["grad_max"],
                    color="tab:red", alpha=0.3, label="Standard VGG")
    ax.fill_between(xs_bn, bn_c["grad_min"], bn_c["grad_max"],
                    color="tab:blue", alpha=0.3, label="Standard VGG + BatchNorm")
    ax.set(title="Gradient Predictiveness",
           xlabel="Steps", ylabel=r"$\ell_2$ change of gradient")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "task2_gradient_predictiveness.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_beta_smoothness(std_c, bn_c):
    if std_c["grad_mat"].size == 0:
        return
    # "Maximum difference in gradient over distance": running max of the
    # per-step gradient change, the practical beta-smoothness estimate.
    fig, ax = plt.subplots(figsize=(8, 5))
    std_beta = np.maximum.accumulate(std_c["grad_mat"].max(axis=0))
    bn_beta = np.maximum.accumulate(bn_c["grad_mat"].max(axis=0))
    ax.plot(std_beta, color="tab:red", label="Standard VGG")
    ax.plot(bn_beta, color="tab:blue", label="Standard VGG + BatchNorm")
    ax.set(title=r'"Effective $\beta$-smoothness" (max gradient difference)',
           xlabel="Steps", ylabel="max gradient difference")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "task2_beta_smoothness.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["compare", "landscape", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=20,
                        help="epochs for Part A")
    parser.add_argument("--landscape-epochs", type=int, default=3,
                        help="epochs per LR for Part B")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap optimization steps per LR in Part B")
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    if args.part in ("compare", "all"):
        part_compare(epochs=args.epochs)
    if args.part in ("landscape", "all"):
        part_landscape(epochs=args.landscape_epochs, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
