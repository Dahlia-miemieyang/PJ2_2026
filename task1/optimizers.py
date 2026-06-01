"""From-scratch optimizers implemented as ``torch.optim.Optimizer`` subclasses.

These satisfy assignment task 4(b): "Implement an optimizer ... by yourself ...
You may use things like torch.matmul." We only use elementary tensor ops
(add/mul/addcmul/sqrt) inside the update — no torch.optim update internals.

Two optimizers are provided:
  * SGDMomentum  — SGD with (Nesterov) momentum + weight decay
  * Adam         — Adam with bias-corrected first/second moments

Both are verified against their torch.optim counterparts in the __main__ block.
"""
import math

import torch
from torch.optim.optimizer import Optimizer


class SGDMomentum(Optimizer):
    """SGD with momentum and optional Nesterov acceleration, written from scratch.

    Update (per parameter p with gradient g):
        g      <- g + weight_decay * p
        buf    <- momentum * buf + g
        d      <- g + momentum * buf      (if nesterov else buf)
        p      <- p - lr * d
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0,
                 nesterov=False):
        if lr < 0.0:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(g).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(g)
                    d = g.add(buf, alpha=momentum) if nesterov else buf
                else:
                    d = g
                p.add_(d, alpha=-lr)
        return loss


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2014) implemented from scratch.

        m   <- b1*m + (1-b1)*g
        v   <- b2*v + (1-b2)*g^2
        m^  <- m / (1 - b1^t)
        v^  <- v / (1 - b2^t)
        p   <- p - lr * m^ / (sqrt(v^) + eps)
    Decoupled weight decay (AdamW-style) optional via `decoupled_wd`.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled_wd=False):
        if lr < 0.0:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        decoupled_wd=decoupled_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            decoupled = group["decoupled_wd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0 and not decoupled:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                bias_c1 = 1 - b1 ** t
                bias_c2 = 1 - b2 ** t
                denom = (v.sqrt() / math.sqrt(bias_c2)).add_(eps)
                step_size = lr / bias_c1

                if wd != 0 and decoupled:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)
        return loss


if __name__ == "__main__":
    # Sanity check: our optimizers should closely track torch.optim on a
    # tiny convex problem (linear regression).
    torch.manual_seed(0)
    X = torch.randn(256, 10)
    w_true = torch.randn(10, 1)
    Y = X @ w_true + 0.01 * torch.randn(256, 1)

    def run(opt_ctor, ref=False):
        torch.manual_seed(1)
        w = torch.zeros(10, 1, requires_grad=True)
        opt = opt_ctor([w])
        for _ in range(200):
            opt.zero_grad()
            loss = ((X @ w - Y) ** 2).mean()
            loss.backward()
            opt.step()
        return loss.item()

    import torch.optim as O
    print("custom SGDMomentum loss:", run(lambda p: SGDMomentum(p, lr=0.05, momentum=0.9)))
    print("torch   SGD       loss:", run(lambda p: O.SGD(p, lr=0.05, momentum=0.9)))
    print("custom Adam       loss:", run(lambda p: Adam(p, lr=0.05)))
    print("torch  Adam       loss:", run(lambda p: O.Adam(p, lr=0.05)))
