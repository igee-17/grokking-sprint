"""Starter solution: embedding + 1-hidden-layer MLP trained with AdamW.

This is the standard grokking setup (Power et al., 2022; Liu et al., 2022;
Gromov, 2023). It memorizes the training set quickly and does generalize --
eventually. It does NOT generalize within this task's step budget.

Edit anything in this file. Keep the three function signatures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, p, d=128, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(p + 1, d)
        self.l1 = nn.Linear(2 * d, hidden)
        self.l2 = nn.Linear(hidden, p)

    def forward(self, x):  # x: (N, 3) long tokens [a, b, "="]
        z = self.emb(x[:, :2]).flatten(1)
        return self.l2(F.relu(self.l1(z)))  # (N, p) logits


def build_model(p):
    return MLP(p)


def build_optimizer(model, total_steps):
    # Must return an object with .zero_grad() and .step().
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))


def on_after_backward(model, step):
    # Optional. Runs after loss.backward(), before optimizer.step(), under no_grad.
    # You may read and modify q.grad for q in model.parameters().
    # You may NOT modify parameter values (the grader checks).
    pass
