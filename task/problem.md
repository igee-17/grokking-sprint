# Grokking Sprint: make a network generalize before the budget runs out

## Background

Small neural networks trained on modular arithmetic (e.g. predict `(a + b) mod p`
from half the table) show **grokking** (Power et al., 2022): they memorize the
training set almost immediately, sit at chance-level held-out accuracy for a long
time, and then suddenly generalize. A line of follow-up work studies *why* this
delay happens and how to shorten it, e.g. through weight norm and weight decay
(Liu et al., 2022, "Omnigrok"), the representation the network ends up learning
(Nanda et al., 2023; Gromov, 2023), and the dynamics of the gradient
(Lee et al., 2024, "Grokfast").

## Your task

Edit `starter/solution.py` so that the network **generalizes within a fixed
budget of 1000 full-batch training steps**.

The starter memorizes fast but stays far below the bar on held-out pairs at step
1000. You are graded on operations you have **not** seen, so your method must be
a general way of making learning generalize sooner, not a solution to one table.

## Setup (fixed by the grader)

- Modulus `p = 97`. Inputs are token triples `[a, b, "="]` with `a, b in [0, 97)`
  and `"="` encoded as token `97`. Output: logits over `p` classes.
- A random 40% of all `(a, b)` pairs is the training set; the remaining 60% is
  held out. The training loop is **full batch** and runs **exactly 1000 steps**.
- The grader owns the loop. Each step: `logits = model(x)`, cross-entropy loss,
  `optimizer.zero_grad()`, `loss.backward()`, `on_after_backward(model, step)`,
  `optimizer.step()`.

## What you submit

A single file `solution.py` defining:

| Function | Contract |
|---|---|
| `build_model(p) -> nn.Module` | Returns a model mapping `(N, 3)` long tensors to `(N, p)` logits. At most **250,000** trainable parameters. **No buffers.** |
| `build_optimizer(model, total_steps)` | Returns any object with `.zero_grad()` and `.step()`. Schedules are allowed. |
| `on_after_backward(model, step)` *(optional)* | Runs under `no_grad` after the backward pass. May read and modify `q.grad`. **Must not** change parameter values. |

## Rules

- Allowed imports: `torch`, `math`, `typing`, `functools`, `collections`, `dataclasses`.
- No introspection or dynamic access (`inspect`, `sys`, `os`, `gc`, `getattr`,
  `eval`, `exec`, dunder attributes, ...), no autograd hooks or custom
  `autograd.Function`, and no modular-arithmetic primitives (`%`, `//`, `**`,
  `pow`, `remainder`, `fmod`). A neural network doesn't need any of these.
- Training and prediction run in **separate processes**. The prediction process
  builds a fresh model with `build_model(p)` and loads your trained
  `state_dict`. Anything not stored in parameters is lost.
- Training must complete within 300 s on 2 CPU threads.

## Scoring

The grader trains your submission on several **hidden** operations over `Z_97`,
with multiple seeds. A run passes if held-out accuracy is **>= 95%** after
step 1000. **Score = fraction of passing runs; you pass the task at 1.0.**

You can check your work on the public operations (`a+b`, `a-b`):

```
python grader/grade.py task/starter/solution.py --public
```

Public results do not guarantee hidden results.
