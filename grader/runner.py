"""Child process that executes submission code in isolation.

  train:   gets ONLY the training tensors (no op name, no held-out split),
           runs the grader-owned loop, writes the trained state_dict.
  predict: gets ONLY held-out inputs (no labels), rebuilds a FRESH model,
           loads the state_dict strictly, writes argmax predictions.

A fresh process for each phase means module-level globals, closures and
Python attributes stashed during training are gone at prediction time:
anything the model "knows" has to live in its parameters.
"""
import importlib.util
import json
import sys

import torch
import torch.nn.functional as F


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("build_model", "build_optimizer"):
        if not callable(getattr(mod, fn, None)):
            raise ValueError(f"submission must define {fn}()")
    return mod


def build_checked(sub, p, max_params):
    model = sub.build_model(p)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("build_model must return an nn.Module")
    n = sum(q.numel() for q in model.parameters())
    if n == 0 or n > max_params:
        raise ValueError(f"model has {n} parameters; must be in [1, {max_params}]")
    if any(True for _ in model.buffers()):
        raise ValueError("buffers are not allowed; all state must be trainable parameters")
    return model, n


def do_train(cfg):
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    sub = load_submission(cfg["submission"])
    data = torch.load(cfg["data"])
    x, y, p, steps = data["x"], data["y"], cfg["p"], cfg["steps"]
    model, n_params = build_checked(sub, p, cfg["max_params"])
    opt = sub.build_optimizer(model, steps)
    hook = getattr(sub, "on_after_backward", None)

    for step in range(steps):
        model.train()
        logits = model(x)
        if not isinstance(logits, torch.Tensor) or logits.shape != (len(x), p):
            raise ValueError(f"forward must return logits of shape {(len(x), p)}")
        loss = F.cross_entropy(logits.float(), y)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        if hook is not None:
            before = [q.detach().clone() for q in model.parameters()]
            with torch.no_grad():
                hook(model, step)
            for b, q in zip(before, model.parameters()):
                if not torch.equal(b, q.detach()):
                    raise ValueError("on_after_backward changed parameter values; only .grad may be modified")
        opt.step()

    with torch.no_grad():
        train_acc = (model(x).argmax(-1) == y).float().mean().item()
    torch.save(model.state_dict(), cfg["out"])
    return {"train_acc": train_acc, "final_loss": loss.item(), "n_params": n_params}


def do_predict(cfg):
    torch.set_num_threads(cfg["threads"])
    sub = load_submission(cfg["submission"])
    model, _ = build_checked(sub, cfg["p"], cfg["max_params"])
    model.load_state_dict(torch.load(cfg["state"]), strict=True)
    model.eval()
    x = torch.load(cfg["data"])["x"]
    with torch.no_grad():
        logits = model(x)
        if logits.shape != (len(x), cfg["p"]):
            raise ValueError("bad output shape at prediction time")
        preds = logits.argmax(-1)
    torch.save(preds, cfg["out"])
    return {}


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    try:
        res = {"ok": True, **(do_train(cfg) if cfg["mode"] == "train" else do_predict(cfg))}
    except Exception as e:  # report, never crash the grader
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print("RESULT " + json.dumps(res))
