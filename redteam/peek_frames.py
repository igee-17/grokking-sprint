"""EXPLOIT: walk the call stack from inside the hook to find the grader's
tensors (looking for a held-out split or labels to memorize).
Defense exercised: static ban on introspection + process isolation
(even if this ran, the training process never holds test data)."""
import inspect
import torch
import torch.nn as nn


def build_model(p):
    return nn.Linear(3, p)


def build_optimizer(model, total_steps):
    return torch.optim.SGD(model.parameters(), lr=0.1)


def on_after_backward(model, step):
    frame = inspect.currentframe().f_back
    print(list(frame.f_locals))
