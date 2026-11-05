"""Reference solution (kept by the task author; not shown to agents).

Same architecture as the starter. The change is the optimizer:
higher learning rate (1e-3 -> 5e-3) and stronger decoupled weight decay
(1.0 -> 2.0).

Why: Liu et al. (2022, "Omnigrok") tie grokking to weight norm. Memorizing
solutions sit at large norm; generalizing ones at small norm. Weight decay
pulls the network toward the small-norm region, and the effective rate of
that pull scales with lr * wd. Raising both shortens the plateau ~2-4x here.

Negative result worth recording: Grokfast-EMA (Lee et al., 2024; alpha=0.98,
lambda in {2, 5}) did NOT help in this regime. At lr=1e-3 it failed to
generalize within 1000-3000 steps, and at lr=3e-3 it delayed generalization
(add: 1250 -> 2250 steps). Amplifying the slow gradient component seems to
interact badly with an already-large effective step size.
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

    def forward(self, x):
        z = self.emb(x[:, :2]).flatten(1)
        return self.l2(F.relu(self.l1(z)))


def build_model(p):
    return MLP(p)


def build_optimizer(model, total_steps):
    return torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=2.0, betas=(0.9, 0.98))
