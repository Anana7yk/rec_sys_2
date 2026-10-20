"""GPT-2 causal across timestamp groups; permutation-invariant tied readout."""

import sys, math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


from models import Block, seed_all, freeze, categorical_kl


class GroupGPT(nn.Module):
    def __init__(self, n, context=64, d=128, layers=2, heads=4):
        super().__init__()
        self.spec = dict(n=n, context=context, d=d, layers=layers, heads=heads)
        self.item = nn.Embedding(n + 3, d, padding_idx=0)
        self.pos = nn.Embedding(context, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d, eps=1e-5)
        self.apply(self.init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x, g):
        h = self.item(x) + self.pos((g - 1).clamp_min(0))
        allow = (g[:, None, None, :] <= g[:, None, :, None]) & (
            x[:, None, None, :] != 0
        )
        for block in self.blocks:
            h = block(h, allow)
        h = self.ln(h)
        last = (x != 0).sum(1) - 1
        return F.linear(
            h[torch.arange(len(x), device=x.device), last], self.item.weight[3:]
        )


def logp(m, x, g, used=None):
    logits = m(x, g)
    if used is not None:
        logits = logits.masked_fill(used, -torch.inf)
    return logits.log_softmax(-1)


def append(x, g, a):
    b, l = x.shape
    rows = torch.arange(b, device=x.device)
    n = (x != 0).sum(1)
    full = n == l
    old = x.clone()
    old[rows, n - 1] = a + 3
    lastg = g[rows, n - 1]
    keep = (x != 0) & ~(full[:, None] & (g == 1))
    pos = (keep.long().cumsum(1) - 1).clamp_min(0)
    z = torch.zeros_like(x).scatter_add_(1, pos, old * keep)
    gg = torch.zeros_like(g).scatter_add_(
        1, pos, (g - full[:, None].long()).clamp_min(0) * keep
    )
    count = keep.sum(1)
    z[rows, count] = 1
    gg[rows, count] = lastg - full.long() + 1
    return z, gg


def save(p, m, **extra):
    torch.save(dict(spec=m.spec, state=m.state_dict(), **extra), p)


def load(p, device):
    q = torch.load(p, map_location="cpu", weights_only=False)
    m = HybridGPT(**q["spec"])
    m.load_state_dict(q["state"])
    return m.to(device)


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


class HybridGPT(GroupGPT):
    """Fixed mixture gate; gradients update only GPT parameters."""

    def __init__(self, n, context=64, d=128, layers=2, heads=4, alpha=0.0):
        super().__init__(n, context, d, layers, heads)
        self.alpha = alpha
        self.spec["alpha"] = alpha
        self.register_buffer("popularity", torch.full((n,), 1 / n))

    def pure(self, x, g):
        return super().forward(x, g)

    def count(self, x):
        c = torch.zeros((len(x), self.spec["n"]), device=x.device)
        c.scatter_add_(1, (x - 3).clamp_min(0), (x >= 3).float())
        total = c.sum(1, keepdim=True)
        return torch.where(
            total > 0,
            0.999 * c / total.clamp_min(1) + 0.001 * self.popularity,
            self.popularity,
        )

    def forward(self, x, g):
        z = self.pure(x, g)
        if self.alpha == 0:
            return z
        return torch.logaddexp(
            z.log_softmax(-1) + math.log1p(-self.alpha),
            self.count(x).log() + math.log(self.alpha),
        )
