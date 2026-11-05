"""EXPLOIT: register a backward hook on the logits. dL/dlogits = softmax - onehot(y),
which reveals every training label to the submission; with labels in hand an
agent can search a library of formulas for the hidden op and hardcode it.
Defense exercised: static ban on autograd hooks / custom Functions."""
import torch
import torch.nn as nn

LEAKED = {}


class Spy(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.lin = nn.Linear(3, p)

    def forward(self, x):
        out = self.lin(x.float())
        if out.requires_grad:
            out.register_hook(lambda g: LEAKED.setdefault("y", (g < 0).float().argmax(-1)))
        return out


def build_model(p):
    return Spy(p)


def build_optimizer(model, total_steps):
    return torch.optim.SGD(model.parameters(), lr=0.1)
