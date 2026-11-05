"""Author tool: train starter and reference on one op, log accuracy every 20 steps,
and plot the grokking curves. Not used for grading.

  python analysis/plot_curves.py [op] [steps]
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "grader"))
import harness  # noqa: E402

torch.set_num_threads(2)
op = sys.argv[1] if len(sys.argv) > 1 else "mul"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000


def load(path):
    s = importlib.util.spec_from_file_location("s", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def curve(path, seed=11):
    sub = load(path)
    torch.manual_seed(seed)
    (xtr, ytr), (xte, yte) = harness.make_data(97, op, seed, 0.4)
    m = sub.build_model(97)
    o = sub.build_optimizer(m, steps)
    xs, tr, te = [], [], []
    for s in range(steps):
        loss = F.cross_entropy(m(xtr), ytr)
        o.zero_grad()
        loss.backward()
        o.step()
        if s % 20 == 0:
            with torch.no_grad():
                xs.append(s)
                tr.append((m(xtr).argmax(-1) == ytr).float().mean().item())
                te.append((m(xte).argmax(-1) == yte).float().mean().item())
    return xs, tr, te


runs = {"starter (lr 1e-3, wd 1)": "task/starter/solution.py", "reference (lr 5e-3, wd 2)": "reference/solution.py"}
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {"starter (lr 1e-3, wd 1)": "#9a9a9a", "reference (lr 5e-3, wd 2)": "#2a6fdb"}
for label, path in runs.items():
    xs, tr, te = curve(os.path.join(ROOT, path))
    ax.plot(xs, tr, ls="--", lw=1.2, color=colors[label], alpha=0.7)
    ax.plot(xs, te, lw=2, color=colors[label], label=f"{label}: held-out")
ax.axvline(1000, color="#c0392b", lw=1, ls=":")
ax.text(1010, 0.05, "step budget", color="#c0392b", fontsize=9)
ax.axhline(0.95, color="#555", lw=0.8, ls=":")
ax.set_xlabel("full-batch training step")
ax.set_ylabel("accuracy")
ax.set_title(f"Grokking on (a {op} b) mod 97, 40% train (dashed = train acc)")
ax.legend(loc="center right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(ROOT, "results", f"curves_{op}.png")
fig.savefig(out, dpi=150)
print(out)
