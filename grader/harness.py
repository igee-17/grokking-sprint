"""Task definition: operations and deterministic train/held-out splits.

Public ops live here. Hidden ops live in hidden_ops.py, which is NOT shipped
to the agent workspace (see scripts/make_agent_workspace.py).
"""
import torch

OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
}
PUBLIC_OPS = ["add", "sub"]

try:
    from hidden_ops import HIDDEN  # noqa: E402
    OPS.update(HIDDEN)
    HIDDEN_OPS = list(HIDDEN)
except ImportError:  # agent workspace
    HIDDEN_OPS = []

# Versions differ only in budget and data. Everything else is shared.
VERSIONS = {
    "v1": {"steps": 1000, "train_frac": 0.4},
    "v2": {"steps": 500, "train_frac": 0.3},
}


def make_data(p, op, seed, train_frac):
    f = OPS[op]
    xs, ys = [], []
    for a in range(p):
        for b in range(p):
            y = f(a, b, p)
            if y is None:
                continue
            xs.append([a, b, p])  # token p is "="
            ys.append(int(y))
    x = torch.tensor(xs)
    y = torch.tensor(ys)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    n_train = int(train_frac * len(x))
    tr, te = perm[:n_train], perm[n_train:]
    return (x[tr], y[tr]), (x[te], y[te])
