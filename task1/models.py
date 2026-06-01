"""Model zoo for Task-1 (CIFAR-10 classification).

Every model exposes the four *required* components of the assignment:
  (a) Fully-connected layer   (b) 2D convolution
  (c) 2D pooling              (d) non-linear activation

and the bigger models add the *optional* components:
  Batch-Norm, Dropout, Residual connections.

`ACTIVATIONS` lets the ablation study swap the non-linearity without touching
the architecture code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Activation registry — used by the "different activations" ablation.
# ---------------------------------------------------------------------------
ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
    "elu": lambda: nn.ELU(inplace=True),
    "gelu": lambda: nn.GELU(),
    "mish": lambda: nn.Mish(inplace=True),
    "tanh": lambda: nn.Tanh(),
}


def make_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation '{name}', choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# Baseline CNN — minimal model satisfying the four required components.
# ~3 conv blocks + 2 FC layers. Width controlled by `width` for the
# "different number of filters" ablation.
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, width=64, activation="relu",
                 use_bn=False, dropout=0.0):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4

        def conv_block(cin, cout):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(make_activation(activation))
            layers.append(nn.MaxPool2d(2, 2))   # (c) 2D pooling
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, c1),     # 32 -> 16
            conv_block(c1, c2),    # 16 -> 8
            conv_block(c2, c3),    # 8  -> 4
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(c3 * 4 * 4, 256),   # (a) fully-connected
            make_activation(activation),
            self.dropout,
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# ResNet for CIFAR-10 — the "improved" model.
# BasicBlock with BN + residual connection; optional dropout before the FC head.
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = make_activation(activation)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            # Projection shortcut to match dimensions (residual connection).
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)     # residual connection
        return self.act(out)


class ResNetCIFAR(nn.Module):
    """ResNet for 32x32 CIFAR images.

    num_blocks=[2,2,2,2] with base_width=64 reproduces ResNet-18.
    base_width is exposed for the "different number of filters" ablation.
    """

    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, base_width=64,
                 activation="relu", dropout=0.0):
        super().__init__()
        self.in_planes = base_width
        self.activation = activation
        w = base_width

        self.conv1 = nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.act = make_activation(activation)

        self.layer1 = self._make_layer(w,     num_blocks[0], stride=1)
        self.layer2 = self._make_layer(w * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(w * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(w * 8, num_blocks[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)   # 2D pooling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(w * 8 * BasicBlock.expansion, num_classes)  # FC head

    def _make_layer(self, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.activation))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.fc(out)


def resnet18_cifar(num_classes=10, base_width=64, activation="relu", dropout=0.0):
    return ResNetCIFAR((2, 2, 2, 2), num_classes, base_width, activation, dropout)


def build_model(name, **kwargs):
    """Factory used by the experiment scripts."""
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(**kwargs)
    if name in ("resnet18", "resnet18_cifar", "resnet"):
        return resnet18_cifar(**kwargs)
    raise ValueError(f"unknown model '{name}'")


if __name__ == "__main__":
    import numpy as np
    for ctor in [SimpleCNN(), SimpleCNN(use_bn=True, dropout=0.3), resnet18_cifar()]:
        n = sum(int(np.prod(p.shape)) for p in ctor.parameters())
        y = ctor(torch.randn(2, 3, 32, 32))
        print(f"{ctor.__class__.__name__:12s} params={n:,} out={tuple(y.shape)}")
