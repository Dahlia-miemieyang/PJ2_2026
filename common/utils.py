"""Shared helpers: seeding, accuracy, parameter counting, training loop, logging."""
import json
import os
import random
import time

import numpy as np
import torch


def set_random_seeds(seed_value=2026, device="cpu"):
    """Seed python / numpy / torch for reproducible runs."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_number_of_parameters(model):
    """Total number of trainable scalar parameters."""
    return sum(int(np.prod(p.shape)) for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """Return (accuracy, mean_loss) over ``loader``."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total
    mean_loss = loss_sum / total if criterion is not None else float("nan")
    return acc, mean_loss


def train_model(model, optimizer, criterion, train_loader, val_loader, device,
                epochs=30, scheduler=None, best_model_path=None, log_every=0,
                verbose=True, use_amp=False):
    """Generic training loop.

    Returns a history dict with per-epoch train loss/acc and val loss/acc,
    plus best validation accuracy and wall-clock time.

    use_amp enables automatic mixed precision (float16) on CUDA, which roughly
    halves the per-epoch time on the RTX 4060. It works transparently with the
    from-scratch optimizers because they are standard Optimizer subclasses.
    """
    model.to(device)
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "epoch_time": [],
    }
    best_val_acc, best_epoch = 0.0, -1
    amp_on = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        ep_t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp_on):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            if log_every and step % log_every == 0 and verbose:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.4f}")

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(time.time() - ep_t0)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            if best_model_path is not None:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)

        if verbose:
            print(f"[epoch {epoch+1:3d}/{epochs}] "
                  f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
                  f"{history['epoch_time'][-1]:.1f}s")

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["total_time"] = time.time() - t_start
    return history


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
