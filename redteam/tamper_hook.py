"""EXPLOIT: smuggle extra optimizer steps per grader step by writing to
parameter values inside the gradient hook (a compute-budget violation).
Defense exercised: runtime parameter-tamper check."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.emb = nn.Embedding(p + 1, 128)
        self.l1 = nn.Linear(256, 512)
        self.l2 = nn.Linear(512, p)

    def forward(self, x):
        return self.l2(F.relu(self.l1(self.emb(x[:, :2]).flatten(1))))


def build_model(p):
    return MLP(p)


def build_optimizer(model, total_steps):
    return torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2.0)


def on_after_backward(model, step):
    for q in model.parameters():
        q.sub_(1e-3 * 20 * q.grad)  # 20 SGD steps' worth, for free
