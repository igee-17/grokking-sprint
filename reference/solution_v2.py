"""Reference solution for v2 (500 steps, 30% train split). Not shown to agents.

v2 is calibrated so that no single idea passes:

  * Optimizer only (Omnigrok-style lr x wd increase): the v1 reference
    (lr 5e-3, wd 2) gets ~2% held-out. Pushing to lr 1e-2, wd 3-4 passes
    mul/div but stalls on aff at 28-46%.
  * Architecture only (quadratic activation, Gromov 2023) at lr 5e-3:
    67-94% held-out, below the bar.
  * Both together: >=95% by step ~250-375 on every hidden op and seed.

Why the quadratic activation helps: Gromov (2023) shows a 2-layer network
with x -> x^2 activations can represent the modular-arithmetic solution
exactly with Fourier-feature embeddings (cos/sin of 2*pi*k*a/p). Products of
embedding sums give the cos(w(a+b)) terms directly, so the generalizing
circuit is much "closer" to initialization than with ReLU.
"""
import torch
import torch.nn as nn


class QuadMLP(nn.Module):
    def __init__(self, p, d=128, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(p + 1, d)
        self.l1 = nn.Linear(2 * d, hidden)
        self.l2 = nn.Linear(hidden, p)

    def forward(self, x):
        z = self.l1(self.emb(x[:, :2]).flatten(1))
        return self.l2(z * z)


def build_model(p):
    return QuadMLP(p)


def build_optimizer(model, total_steps):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=3.0, betas=(0.9, 0.98))
