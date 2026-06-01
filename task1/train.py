"""Task-1 experiment driver: train CIFAR-10 models, run ablations, compare
optimizers, and train the best model to completion.

Usage examples
--------------
    python task1/train.py --exp best          # train the strong ResNet-18 model
    python task1/train.py --exp ablation       # filters / loss / activation sweeps
    python task1/train.py --exp optimizer      # torch.optim vs custom optimizers
    python task1/train.py --exp baseline       # the minimal SimpleCNN

All runs write a JSON history to reports/logs and the best weights to
reports/models. Figures are produced separately by visualize.py.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import get_cifar_loader
from common.utils import (set_random_seeds, get_number_of_parameters,
                          train_model, evaluate, save_json)
from task1.models import build_model
from task1.optimizers import SGDMomentum, Adam as CustomAdam

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "reports", "logs")
MODEL_DIR = os.path.join(ROOT, "reports", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Loss functions with different regularization (assignment 3b).
# ---------------------------------------------------------------------------
class L1RegLoss(nn.Module):
    """Cross-entropy + explicit L1 penalty on model weights."""

    def __init__(self, model, l1_lambda=1e-5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.model = model
        self.l1_lambda = l1_lambda

    def forward(self, logits, target):
        loss = self.ce(logits, target)
        l1 = sum(p.abs().sum() for p in self.model.parameters())
        return loss + self.l1_lambda * l1


def make_criterion(kind, model=None):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if kind == "l1":
        return L1RegLoss(model, l1_lambda=1e-5)
    raise ValueError(f"unknown loss '{kind}'")


def make_optimizer(kind, model, lr=None, weight_decay=5e-4):
    """Built-in (torch.optim) and from-scratch optimizers."""
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr or 0.1, momentum=0.9,
                               weight_decay=weight_decay, nesterov=True)
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr or 1e-3,
                                weight_decay=weight_decay)
    if kind == "custom_sgd":
        return SGDMomentum(model.parameters(), lr=lr or 0.1, momentum=0.9,
                           weight_decay=weight_decay, nesterov=True)
    if kind == "custom_adam":
        return CustomAdam(model.parameters(), lr=lr or 1e-3,
                          weight_decay=weight_decay, decoupled_wd=True)
    raise ValueError(f"unknown optimizer '{kind}'")


def make_loaders(regime="augment", batch_size=128, num_workers=2):
    train_loader = get_cifar_loader(train=True, regime=regime,
                                    batch_size=batch_size, num_workers=num_workers)
    val_loader = get_cifar_loader(train=False, regime="plain" if regime == "plain"
                                  else "augment", batch_size=256,
                                  num_workers=num_workers, shuffle=False)
    return train_loader, val_loader


def run_one(tag, model_name, model_kwargs, optimizer, loss_kind, epochs,
            lr=None, use_cosine=True, batch_size=128, regime="augment",
            save_best=False, use_amp=True):
    """Train a single configuration and persist its history."""
    set_random_seeds(2026, device=DEVICE.type)
    model = build_model(model_name, **model_kwargs).to(DEVICE)
    n_params = get_number_of_parameters(model)

    train_loader, val_loader = make_loaders(regime, batch_size)
    criterion = make_criterion(loss_kind, model)
    opt = make_optimizer(optimizer, model, lr=lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                 if use_cosine else None)

    best_path = os.path.join(MODEL_DIR, f"{tag}.pth") if save_best else None
    print(f"\n=== {tag} | {model_name} params={n_params:,} | opt={optimizer} "
          f"loss={loss_kind} epochs={epochs} ===")
    # L1 regularization loss reads model weights in fp32; disable AMP for it.
    amp = use_amp and loss_kind != "l1"
    history = train_model(model, opt, criterion, train_loader, val_loader,
                          DEVICE, epochs=epochs, scheduler=scheduler,
                          best_model_path=best_path, verbose=True, use_amp=amp)
    final_acc, final_loss = evaluate(model, val_loader, DEVICE,
                                     nn.CrossEntropyLoss())
    record = {
        "tag": tag, "model": model_name, "model_kwargs": model_kwargs,
        "optimizer": optimizer, "loss": loss_kind, "epochs": epochs,
        "lr": lr, "regime": regime, "batch_size": batch_size,
        "n_params": n_params, "final_val_acc": final_acc,
        "final_val_loss": final_loss, "history": history,
    }
    save_json(record, os.path.join(LOG_DIR, f"{tag}.json"))
    print(f"--- {tag}: best_val_acc={history['best_val_acc']:.4f} "
          f"final_val_acc={final_acc:.4f} time={history['total_time']:.0f}s")
    return record


# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------
def exp_baseline(epochs):
    return [run_one("baseline_simplecnn", "simplecnn",
                    dict(width=64, activation="relu", use_bn=False, dropout=0.0),
                    optimizer="adam", loss_kind="ce", epochs=epochs)]


def exp_best(epochs):
    return [run_one("best_resnet18", "resnet18",
                    dict(base_width=64, activation="relu", dropout=0.0),
                    optimizer="sgd", loss_kind="label_smoothing", epochs=epochs,
                    lr=0.1, save_best=True)]


def exp_ablation(epochs):
    records = []
    # (3a) different number of filters/neurons — SimpleCNN width sweep.
    for width in [16, 32, 64, 128]:
        records.append(run_one(
            f"abl_width_{width}", "simplecnn",
            dict(width=width, activation="relu", use_bn=True, dropout=0.2),
            optimizer="adam", loss_kind="ce", epochs=epochs))
    # (3b) different loss functions / regularization.
    for loss_kind in ["ce", "label_smoothing", "l1"]:
        records.append(run_one(
            f"abl_loss_{loss_kind}", "simplecnn",
            dict(width=64, activation="relu", use_bn=True, dropout=0.2),
            optimizer="adam", loss_kind=loss_kind, epochs=epochs))
    # (3c) different activations.
    for act in ["relu", "leaky_relu", "gelu", "mish", "tanh"]:
        records.append(run_one(
            f"abl_act_{act}", "simplecnn",
            dict(width=64, activation=act, use_bn=True, dropout=0.2),
            optimizer="adam", loss_kind="ce", epochs=epochs))
    return records


def exp_optimizer(epochs):
    records = []
    cfgs = [
        ("opt_sgd", "sgd", 0.1),
        ("opt_adam", "adam", 1e-3),
        ("opt_custom_sgd", "custom_sgd", 0.1),
        ("opt_custom_adam", "custom_adam", 1e-3),
    ]
    for tag, opt, lr in cfgs:
        records.append(run_one(
            tag, "resnet18",
            dict(base_width=64, activation="relu", dropout=0.0),
            optimizer=opt, loss_kind="ce", epochs=epochs, lr=lr))
    return records


EXPERIMENTS = {
    "baseline": exp_baseline,
    "best": exp_best,
    "ablation": exp_ablation,
    "optimizer": exp_optimizer,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="override default epochs for the chosen experiment")
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    # Sensible default epoch budgets per experiment group.
    default_epochs = {"baseline": 30, "best": 80, "ablation": 20, "optimizer": 25}

    if args.exp == "all":
        groups = ["best", "ablation", "optimizer", "baseline"]
    else:
        groups = [args.exp]

    for g in groups:
        ep = args.epochs or default_epochs[g]
        EXPERIMENTS[g](ep)


if __name__ == "__main__":
    main()
