"""Task definition: operations and deterministic train/held-out splits.

Only the grader imports this. Submissions never see the op or the split.
"""
import torch

OPS = {
    # public (described in problem.md)
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    # hidden (only the grader knows these)
    "mul": lambda a, b, p: (a * b) % p,
    "aff": lambda a, b, p: (a + 5 * b) % p,
    "quad": lambda a, b, p: (a * a + a * b + b * b) % p,  # too hard in budget; kept for reference
    "div": lambda a, b, p: (a * pow(int(b), p - 2, p)) % p if b != 0 else None,
}
PUBLIC_OPS = ["add", "sub"]
HIDDEN_OPS = ["mul", "aff", "div"]


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
