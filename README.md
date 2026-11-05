# Grokking Sprint

**An ML research environment for evaluating AI agents: can the agent read the
grokking literature and turn it into a training change that makes a network
generalize ~2–4x sooner, on tasks it has never seen?**

![Grokking curves](results/curves_mul.png)

*Same network, same data, same 1000-step budget. The starter memorizes in ~50
steps but doesn't start generalizing until ~1500. A change motivated by the
weight-norm account of grokking (Liu et al., 2022) generalizes by step ~450.*

---

## Why this task

Grokking (Power et al., 2022) is a well-studied, cheap-to-run phenomenon with
several competing explanations and proposed fixes: weight decay and weight
norm (Liu et al., 2022), representation learning (Nanda et al., 2023;
Gromov, 2023), and gradient filtering (Lee et al., 2024). That makes it a good
testbed for **research skill** rather than recall:

- The agent has to form a hypothesis about *why* generalization is slow and act on it.
- Several literature-backed ideas don't transfer to this regime (see
  [Findings](#findings)), so copying a paper's headline method isn't enough.
- It runs on 2 CPU cores in ~3 minutes per grade, so it can be run many times.

## The task (what the agent sees)

See [`task/problem.md`](task/problem.md). In short:

- Modular arithmetic over `Z_97`. 40% of the `(a, b)` table for training, 60% held out.
- The agent edits one file defining `build_model`, `build_optimizer`, and an
  optional gradient hook `on_after_backward`.
- **The grader owns the training loop:** exactly 1000 full-batch steps.
- **Graded on hidden operations** (multiplication, an affine map, and division
  mod 97) across 2 seeds each. A run passes at ≥95% held-out accuracy. Score =
  fraction of passing runs.

**Two difficulty levels** (same code, `--version` flag):

| | Steps | Train split | What it takes to pass |
|---|---|---|---|
| **v1** | 1000 | 40% | One idea: increase the weight-decay pull (lr × wd) |
| **v2** | 500 | 30% | Two ideas combined: an architecture change *and* an optimizer change |

## Results

| Submission | v1 score | v2 score |
|---|---|---|
| Starter (ReLU MLP, AdamW lr 1e-3, wd 1.0) | **0 / 6** | **0 / 6** |
| v1 reference (ReLU, lr 5e-3, wd 2.0) | **6 / 6** | **0 / 6** (1.2% held-out) |
| v2 reference (quadratic activation, lr 1e-2, wd 3.0) | **6 / 6** | **6 / 6** |
| Exploit: hardcode the public op | **0 / 6** | — |

Raw grader output is in [`results/`](results/). A v1 run takes ~30 s and a v2 run
~10 s on 2 CPU threads.

### How v2 was calibrated

v2 had to be hard enough that one trick doesn't solve it. I swept ideas from
each paper on the hidden ops (time-to-95% checked every 25 steps):

| Idea alone | Result at 500 steps / 30% |
|---|---|
| Optimizer only (lr 1e-2, wd 3–4, ReLU) | Passes `mul`/`div`, **stalls on `aff` at 28–46%** |
| Smaller init (Omnigrok) + optimizer | Same: `aff` stuck at 28–37% |
| Quadratic activation only (lr 5e-3) | 67–94%, **below the bar** |
| Cosine schedule with warmup | Worse: ~2% |
| **Quadratic activation + lr 1e-2, wd 3** | **≥95% by step ~250–375 on every op and seed** |

The quadratic activation works because a network with x → x² can represent
the modular-arithmetic solution exactly using Fourier-feature embeddings
(Gromov, 2023). The circuit that generalizes starts out much closer to
initialization than it does with ReLU. The optimizer change then supplies
the weight-norm pressure that makes the network find it.

## Threat model: how an agent could cheat, and what stops it

Writing the grader is half of an environment. Every defense below has a working
exploit in [`redteam/`](redteam/), and `python redteam/run_redteam.py` asserts
each one scores 0.

| Exploit | What it tries | Defense | Layer |
|---|---|---|---|
| `hardcode_public.py` | Skip learning; compute `a+b mod p` without `%` | Hidden ops (scores 100% on public `add`, 0% on hidden) | Task design |
| `tamper_hook.py` | Take ~20 extra SGD steps per grader step inside the gradient hook | Parameter snapshot before/after the hook | Runtime |
| `buffer_stash.py` | Keep non-learned state (lookup table, formula id) in a buffer | Buffers rejected | Runtime |
| `peek_frames.py` | Walk the stack to find held-out data or labels | Static ban + **process isolation**: the training process never holds test data, and the prediction process never holds labels | Static + architecture |
| `label_leak.py` | Hook `dL/dlogits` (= softmax − onehot) to read every training label, then search formulas | Static ban on autograd hooks / custom `Function` | Static |
| `symbolic_mod.py` | Model is secretly a symbolic program using `%` | Static ban on `%`, `//`, `**`, `pow`, `remainder`, `fmod` | Static |
| `getattr_bypass.py` | Reach banned functions via `getattr(torch, "rem"+"ainder")` | Static ban on `getattr`/dynamic access | Static |

**Training and prediction run in separate processes.** Prediction builds a
*fresh* model and loads the `state_dict` strictly, so module globals, closures
and stashed Python attributes from training don't survive. Anything the
model "knows" has to be in its trainable parameters.

**Checking for false positives.** A grader that rejects legitimate solutions
is as broken as one that accepts cheats. The red-team runner also asserts the
starter and reference pass the static check. Red-teaming caught one false positive in my
first version of the checker: it flagged a *docstring* that mentioned a dunder
name. Bare string statements are now ignored.

## Findings

These came out of calibrating the task, and they double as reasons it's a
real research task and not a lookup:

1. **The weight-norm account predicts the fix.** Raising lr × wd (1e-3×1.0 →
   5e-3×2.0) cuts time-to-95% from >1500 steps to ~450–750 across ops. That's
   consistent with Omnigrok's claim that generalization comes from contracting
   toward a small-norm region.
2. **Grokfast-EMA did not transfer.** With α=0.98, λ∈{2,5}: at lr 1e-3 it didn't
   reach 95% within 1000 steps (or 3000 on `add`). At lr 3e-3 it *slowed*
   generalization (`add`: 1250 → 2250 steps). An agent that pattern-matches
   "grokking → Grokfast" without testing fails.
3. **Not every op is learnable in budget.** `a² + ab + b² mod p` never generalized
   in 1000 steps under any config tried, so I left it out of the hidden set. A
   hidden op that no reference can solve only measures luck.
4. **Architecture matters for compute.** A 1-layer pre-norm transformer plateaued
   at ~38% held-out for 8000 steps on `add` (p=59), which was too slow to calibrate
   on CPU. The embedding-MLP setup groks reliably.

## Known limitations

- **The static check is a filter, not a sandbox.** Hidden ops live in
  `grader/hidden_ops.py`, which the workspace exporter never copies, and agent
  runs happen in a Docker container that only mounts the exported workspace.
  A production setup would also want network and resource limits.
- **A subtle residual leak.** Parameter gradients carry aggregate label
  information, such as the output-bias gradient = per-class counts. I don't see a
  path from that to recovering the op under the static rules, but I haven't proven it.
- **Two seeds per op.** Enough to separate the reference from the baseline, but
  agent scores near the threshold would need more seeds to be trusted.
- **v2 reference margin.** The hardest case (`aff`, seed 11) passes at step
  ~375 of 500. That's comfortable, but a different PyTorch build could shift it.

## Run it

```bash
pip install -r requirements.txt
python grader/grade.py task/starter/solution.py --public   # what an agent can run
python grader/grade.py reference/solution.py               # official hidden-op score (v1)
python grader/grade.py reference/solution_v2.py --version v2
python redteam/run_redteam.py                              # all exploits must score 0
python analysis/plot_curves.py mul 2000                    # regenerate the figure
```

## Running an agent on it

```bash
ANTHROPIC_API_KEY=sk-... ./agent_runs/run_claude_code.sh v2 opus 1
```

This exports a clean workspace (public ops only) to `~/grokking-agent-runs/`,
runs Claude Code headless inside a Docker container that can see only that
folder, grades the result on hidden ops outside the container, and writes a
`summary.md` with the score, turns, cost, tool usage, the agent's own notes,
and its diff. See [`agent_runs/README.md`](agent_runs/README.md).

## Layout

```
task/            what the agent gets: problem.md + starter/solution.py
grader/          grade.py (scoring), runner.py (isolated child process),
                 static_check.py, harness.py (public ops, splits, versions),
                 hidden_ops.py (never exported to agents)
reference/       solution.py (v1) and solution_v2.py, reasoning in docstrings
scripts/         make_agent_workspace.py: clean export for an agent run
agent_runs/      Docker sandbox, run script, transcript summarizer, results
redteam/         one exploit per defense + run_redteam.py
analysis/        curve plotting
results/         grader outputs and figure
```

## References

- Power et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.*
- Liu, Michaud, Tegmark (2022). *Omnigrok: Grokking Beyond Algorithmic Data.*
- Nanda et al. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.*
- Gromov (2023). *Grokking Modular Arithmetic.*
- Lee et al. (2024). *Grokfast: Accelerated Grokking by Amplifying Slow Gradients.*
