# Project 2 
CIFAR-10 image classification and a study of how Batch Normalization helps optimization, for the *Neural Network and Deep Learning* course (PJ2, 2026).

## Contents

| Path | Description |
|------|-------------|
| `codes/common/` | shared CIFAR-10 data loaders, training loop, utilities |
| `codes/task1/` | Task-1: model zoo, from-scratch optimizers, training driver, visualizations |
| `codes/VGG_BatchNorm/` | Task-2: VGG-A / VGG-A-BN models and Batch-Norm experiments |
| `reports/figures/` | all generated figures |
| `reports/logs/` | per-experiment JSON histories |

## Environment

```
Python 3.10+, PyTorch 2.x (CUDA build recommended), torchvision, numpy,
matplotlib, tqdm.
```

Install (conda):

```bash
conda create -n torch_env python=3.10
conda activate torch_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib tqdm
```

CIFAR-10 downloads automatically into `codes/data/` on first run.

## Reproducing the results

All commands are run from the `codes/` directory with `torch_env` active.

### Task 1 — CIFAR-10 classification (60%)

```bash
# train everything: best ResNet-18 + ablations + optimizer comparison + baseline
python task1/train.py --exp all

# or run a single group
python task1/train.py --exp best        # strong ResNet-18 (saves best weights)
python task1/train.py --exp ablation     # filters / loss / activation sweeps
python task1/train.py --exp optimizer    # torch.optim vs. from-scratch optimizers
python task1/train.py --exp baseline     # minimal SimpleCNN

# generate all Task-1 figures (filters, feature maps, Grad-CAM, curves, ...)
python task1/visualize.py
```

### Task 2 — Batch Normalization (30%)

```bash
cd VGG_BatchNorm
python vgg_bn_experiments.py --part compare    # VGG-A with vs. without BN
python vgg_bn_experiments.py --part landscape   # loss landscape + gradient analysis
python vgg_bn_experiments.py --part all

# the assignment-referenced reference script (loss landscape only):
python VGG_Loss_Landscape.py
```

## Notes

- The CIFAR-10 archive bundled with the assignment template was truncated, so the loaders download a fresh copy into `codes/data/`.
- Training uses automatic mixed precision (AMP) on CUDA to roughly halve the per-epoch time; it is disabled automatically for the explicit-L1 loss run.
- All runs are seeded (`set_random_seeds`) for reproducibility.
