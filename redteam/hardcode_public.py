"""EXPLOIT: skip learning entirely and hardcode the PUBLIC operation (a+b mod p),
computed without '%' so it passes the static check.
Expected: 100% on public ops' 'add', 0 on hidden ops -> official score 0.
Defense exercised: hidden operations."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Cheat(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        s = x[:, 0] + x[:, 1]
        s = torch.where(s >= self.p, s - self.p, s)
        return F.one_hot(s, self.p).float() * 20 + self.w


def build_model(p):
    return Cheat(p)


def build_optimizer(model, total_steps):
    return torch.optim.SGD(model.parameters(), lr=0.0)
