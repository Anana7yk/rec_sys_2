"""GPT-2 decoder trained from scratch, and response regressors."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d)
        )

    def forward(self, x, allow):
        b, l, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = [
            a.view(b, l, self.heads, d // self.heads).transpose(1, 2) for a in [q, k, v]
        ]
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=0.0)
        x = x + self.proj(z.transpose(1, 2).reshape(b, l, d))
        return x + self.mlp(self.ln2(x))


class GPT2Rec(nn.Module):
    def __init__(self, n, context, d=96, layers=2, heads=4):
        super().__init__()
        self.spec = dict(n=n, context=context, d=d, layers=layers, heads=heads)
        self.item = nn.Embedding(n + 3, d, padding_idx=0)
        self.pos = nn.Embedding(context, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d, eps=1e-5)
        self.apply(self.init)
        # GPT-2 residual projection scaling.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden(self, tokens):
        b, l = tokens.shape
        valid = tokens != 0
        pos = (valid.long().cumsum(1) - 1).clamp_min(0)
        x = self.item(tokens) + self.pos(pos)
        causal = torch.ones((l, l), dtype=torch.bool, device=tokens.device).tril()
        allow = causal[None, None, :, :] & valid[:, None, None, :]
        for block in self.blocks:
            x = block(x, allow)
        return self.ln(x)

    def forward(self, tokens):
        h = self.hidden(tokens)
        last = (tokens != 0).long().sum(1) - 1
        return F.linear(
            h[torch.arange(len(tokens), device=tokens.device), last],
            self.item.weight[3:],
        )


def append_action(x, action):
    # Preserve BOS; drop oldest context item when full. No target response appended.
    n = (x != 0).long().sum(1)
    full = n == x.shape[1]
    z = x.clone()
    z[full, 1:-1] = x[full, 2:]
    z[full, -1] = action[full] + 3
    rows = torch.arange(len(x), device=x.device)[~full]
    z[rows, n[~full]] = action[~full] + 3
    return z


def policy_logp(model, x, temperature=1.0, used=None):
    logits = model(x) / temperature
    if used is not None:
        logits = logits.masked_fill(used, -torch.inf)
    return F.log_softmax(logits, -1)


def categorical_kl(lp, lref):
    # Masked actions have probability zero in BOTH policies. Avoid 0 * NaN.
    delta = torch.where(torch.isfinite(lp), lp - lref, torch.zeros_like(lp))
    return (lp.exp() * delta).sum(-1)


class ResponseModel(nn.Module):
    def __init__(self, n, meta, personal=True, d=32):
        super().__init__()
        self.personal = personal
        self.spec = dict(n=n, personal=personal, d=d)
        self.register_buffer("meta", torch.as_tensor(meta, dtype=torch.float32))
        self.item = nn.Embedding(n + 3, d, padding_idx=0)
        inp = d + 2 + (2 * d + 2 if personal else 0)
        self.net = nn.Sequential(nn.Linear(inp, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, x, a):
        # Actions are 0-based catalog; context tokens use offset 3.
        v = self.item(a + 3)
        feats = [v, self.meta[a]]
        if self.personal:
            valid = x >= 3
            h = (self.item(x) * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[
                :, None
            ]
            rep = (x == a[:, None] + 3).float().sum(1, keepdim=True) / x.shape[1]
            known = valid.float().mean(1, keepdim=True)
            feats.extend([h, h * v, rep, known])
        return torch.cat(feats, -1)

    def forward(self, x, a):
        return self.net(self.encode(x, a)).squeeze(-1).sigmoid()


def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
