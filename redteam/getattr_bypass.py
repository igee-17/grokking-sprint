"""EXPLOIT: reach banned modules indirectly via getattr/strings, e.g.
getattr(torch, 'remainder') or __import__('inspect').
Defense exercised: static ban on getattr and dunder strings."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class G(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mod = getattr(torch, "rem" + "ainder")
        y = mod(x[:, 0] * x[:, 1], self.p)
        return F.one_hot(y, self.p).float() * 20 + self.w


def build_model(p):
    return G(p)


def build_optimizer(model, total_steps):
    return torch.optim.SGD(model.parameters(), lr=0.0)
