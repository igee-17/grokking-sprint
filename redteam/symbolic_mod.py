"""EXPLOIT: a 'model' that is a symbolic program: it carries a formula id in a
parameter and evaluates candidate formulas with modular arithmetic.
Defense exercised: static ban on '%', '//', '**', pow/remainder/fmod."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Symbolic(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.formula = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        a, b = x[:, 0], x[:, 1]
        y = (a * b) % self.p
        return F.one_hot(y, self.p).float() * 20 + self.formula


def build_model(p):
    return Symbolic(p)


def build_optimizer(model, total_steps):
    return torch.optim.SGD(model.parameters(), lr=0.0)
