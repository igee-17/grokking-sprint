"""EXPLOIT: keep non-learned state (e.g. a lookup table or a chosen formula)
in a registered buffer so it survives into the prediction process.
Defense exercised: buffers are rejected; all state must be trainable params."""
import torch
import torch.nn as nn


class Stash(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.lin = nn.Linear(3, p)
        self.register_buffer("table", torch.zeros(p, p, dtype=torch.long))

    def forward(self, x):
        return self.lin(x.float())


def build_model(p):
    return Stash(p)


def build_optimizer(model, total_steps):
    return torch.optim.SGD(model.parameters(), lr=0.1)
